#!/usr/bin/env python3
"""V10: validate LLM rules with normal data and fuse only co-located evidence."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from neurosymbolic_fsad.augmentations import build_calibration_views
from neurosymbolic_fsad.data import (
    collect_test_samples,
    normal_train_images,
    select_inpformer_support,
)
from neurosymbolic_fsad.evidence_consistency import (
    EvidenceConsistentBridge,
    top_fraction_mean,
)
from neurosymbolic_fsad.experiment_logic import (
    _heat_overlay,
    _label_panel,
    _mask_for_sample,
)
from neurosymbolic_fsad.inpformer_adapter import (
    DEFAULT_INPFORMER_ROOT,
    INPFormerAdapter,
    load_prediction_cache,
    save_prediction_cache,
)
from neurosymbolic_fsad.transistor_logic import (
    TransistorLogicEngine,
    execute_grounded_neural_symbolic,
    preprocess_rgb,
)


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _read_cache(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "paths": payload["paths"].astype(str),
            "maps": payload["maps"].astype(np.float32),
            "scores": payload["scores"].astype(np.float64),
            "metadata": json.loads(str(payload["metadata_json"].item())),
        }


def _metrics(labels, scores, masks, maps) -> Dict[str, float]:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    masks = np.asarray(masks).reshape(-1)
    maps = np.asarray(maps).reshape(-1)
    return {
        "i_auroc": float(roc_auc_score(labels, scores)) if np.ptp(scores) else 0.5,
        "i_aupr": float(average_precision_score(labels, scores)),
        "p_auroc": float(roc_auc_score(masks, maps)) if np.ptp(maps) else 0.5,
        "p_aupr": float(average_precision_score(masks, maps)),
    }


def _delta(candidate: Mapping[str, float], baseline: Mapping[str, float]):
    return {name: float(candidate[name] - baseline[name]) for name in baseline}


def _select_remaining_normals(
    class_root: Path,
    excluded: Sequence[str],
    count: int,
    seed: int,
) -> list[Path]:
    excluded_set = {str(Path(item).resolve()) for item in excluded}
    candidates = [
        path for path in normal_train_images(class_root)
        if str(path.resolve()) not in excluded_set
    ]
    if count < 1 or count > len(candidates):
        raise ValueError(
            f"validation_count={count}; only {len(candidates)} disjoint normals exist"
        )
    return sorted(random.Random(seed).sample(candidates, count), key=lambda p: p.name)


def _save_v10_visualization(
    directory: Path,
    sample: Mapping[str, Any],
    mask: np.ndarray,
    baseline_map: np.ndarray,
    raw_symbolic_map: np.ndarray,
    confirmed_symbolic_map: np.ndarray,
    fused_map: np.ndarray,
    index: int,
) -> Dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    image_path = Path(sample["path"])
    original = preprocess_rgb(image_path)
    gt = original.copy()
    if np.any(mask):
        contours = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        cv2.drawContours(gt, contours, -1, (255, 0, 0), 2)
    panels = {
        "baseline": _heat_overlay(original, baseline_map),
        "raw_symbolic": _heat_overlay(original, raw_symbolic_map),
        "confirmed_symbolic": _heat_overlay(original, confirmed_symbolic_map),
        "fused": _heat_overlay(original, fused_map),
    }
    prefix = f"{index:03d}_{image_path.parent.name}_{image_path.stem}"
    outputs = {
        "original": directory / f"{prefix}_original.jpg",
        "gt": directory / f"{prefix}_gt.jpg",
        "baseline": directory / f"{prefix}_baseline_INPFormer.jpg",
        "raw_symbolic": directory / f"{prefix}_raw_symbolic.jpg",
        "confirmed_symbolic": directory / f"{prefix}_confirmed_symbolic.jpg",
        "fused": directory / f"{prefix}_fused_evidence_consistent.jpg",
        "comparison": directory / f"{prefix}_comparison.jpg",
    }
    Image.fromarray(original).save(outputs["original"], quality=95)
    Image.fromarray(gt).save(outputs["gt"], quality=95)
    for name, panel in panels.items():
        Image.fromarray(panel).save(outputs[name], quality=95)
    comparison = np.concatenate([
        _label_panel(original, "Original"),
        _label_panel(gt, "Pixel GT (evaluation only)"),
        _label_panel(panels["baseline"], "INP-Former baseline"),
        _label_panel(panels["raw_symbolic"], "Raw rule violation"),
        _label_panel(panels["confirmed_symbolic"], "Rule AND co-located INP evidence"),
        _label_panel(panels["fused"], "Final OR fusion"),
    ], axis=1)
    Image.fromarray(comparison).save(outputs["comparison"], quality=95)
    return {name: str(path) for name, path in outputs.items()}


def _per_defect(
    labels, defect_types, candidate_scores, baseline_scores,
    masks, candidate_maps, baseline_maps,
):
    result = {}
    for defect in sorted(set(defect_types) - {"good", "ok"}):
        selected = np.isin(defect_types, ["good", "ok", defect])
        anomaly = defect_types == defect
        result[defect] = {
            "anomaly_count": int(np.sum(anomaly)),
            "baseline_i_auroc": float(roc_auc_score(labels[selected], baseline_scores[selected])),
            "v10_i_auroc": float(roc_auc_score(labels[selected], candidate_scores[selected])),
            "baseline_p_auroc": float(roc_auc_score(
                masks[anomaly].reshape(-1), baseline_maps[anomaly].reshape(-1)
            )),
            "v10_p_auroc": float(roc_auc_score(
                masks[anomaly].reshape(-1), candidate_maps[anomaly].reshape(-1)
            )),
            "baseline_p_aupr": float(average_precision_score(
                masks[anomaly].reshape(-1), baseline_maps[anomaly].reshape(-1)
            )),
            "v10_p_aupr": float(average_precision_score(
                masks[anomaly].reshape(-1), candidate_maps[anomaly].reshape(-1)
            )),
        }
    return result


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).expanduser().resolve()
    class_root = data_root / args.class_name
    library_path = Path(args.rule_library).expanduser().resolve()
    library = json.loads(library_path.read_text(encoding="utf-8"))
    if library.get("dataset") != "mvtec" or library.get("class_name") != "transistor":
        raise ValueError("The controlled V10 implementation currently targets MVTec/transistor")
    engine = TransistorLogicEngine(library)

    cache_root = Path(args.prediction_cache_dir).expanduser().resolve()
    if (cache_root / "cache").is_dir():
        cache_root /= "cache"
    query_cache = _read_cache(cache_root / "test_inp_v4.npz")
    baseline_support = select_inpformer_support(
        data_root, "mvtec", args.class_name,
        int(query_cache["metadata"].get("shot", 4)),
        int(query_cache["metadata"].get("support_seed", 1)),
    )

    if args.validation_source == "remaining_train":
        validation_paths = _select_remaining_normals(
            class_root,
            list(library["support"]) + [str(path) for path in baseline_support],
            args.validation_count,
            args.validation_seed,
        )
        validation_ids = [str(path.resolve()) for path in validation_paths]
        validation_images: Sequence[Any] = validation_paths
        validation_cache_path = output / "cache" / "normal_validation_inp_v10.npz"
        expected_metadata = {
            **query_cache["metadata"],
            "purpose": "disjoint_remaining_train_normal_validation_v10",
            "validation_seed": int(args.validation_seed),
            "validation_count": int(args.validation_count),
        }
        validation_predictions = load_prediction_cache(
            validation_cache_path, validation_ids, expected_metadata=expected_metadata
        )
        if validation_predictions is None:
            adapter = INPFormerAdapter(
                checkpoint=args.checkpoint,
                device=args.device,
                inpformer_root=args.inpformer_root,
                encoder_name=args.encoder,
                inp_num=args.inp_num,
                input_size=args.input_size,
                crop_size=args.crop_size,
                map_size=args.map_size,
            )
            validation_predictions = adapter.infer(
                validation_paths, batch_size=args.batch_size
            )
            save_prediction_cache(
                validation_cache_path, validation_predictions,
                metadata=expected_metadata,
            )
            adapter.close()
            del adapter
            gc.collect()
        protocol_name = (
            f"FSAD-NV: {query_cache['metadata'].get('shot', 4)}-shot INP baseline + "
            f"{len(library['support'])} normal rule-discovery images + "
            f"{len(validation_paths)} disjoint normal validation images"
        )
        strict_k_shot = False
    else:
        views = build_calibration_views(baseline_support, seed=args.validation_seed)
        normal_indices = [i for i, view in enumerate(views) if not view.is_pseudo_anomaly]
        validation_images = [views[i].image for i in normal_indices]
        augmentation_cache = _read_cache(
            Path(args.support_augmentation_cache).expanduser().resolve()
        )
        expected_all = np.asarray([
            f"{'pseudo' if view.is_pseudo_anomaly else 'normal'}://"
            f"{view.source_index}/{view.transform}" for view in views
        ])
        if not np.array_equal(augmentation_cache["paths"], expected_all):
            raise ValueError("support augmentation cache does not match deterministic views")
        validation_predictions = {
            "maps": augmentation_cache["maps"][normal_indices],
            "scores": augmentation_cache["scores"][normal_indices],
        }
        validation_ids = expected_all[normal_indices].tolist()
        validation_paths = []
        validation_cache_path = Path(args.support_augmentation_cache).resolve()
        protocol_name = "support-only nuisance augmentation calibration (no extra real normals)"
        strict_k_shot = len(library["support"]) <= int(query_cache["metadata"].get("shot", 4))

    bridge = EvidenceConsistentBridge.calibrate(
        engine,
        validation_images,
        validation_predictions["maps"],
        normal_identifiers=validation_ids,
        protocol=protocol_name,
        fact_quantile=args.fact_quantile,
        evidence_quantile=args.evidence_quantile,
        maximum_validation_fpr=args.maximum_validation_fpr,
        max_threshold_shift_scales=args.max_threshold_shift_scales,
    )
    _dump(output / "bridge_calibration_v10.json", bridge.calibration)
    effective_library = json.loads(json.dumps(library))
    effective_library.update({
        "schema_version": "10.0-evidence-consistent-static-rule-library",
        "base_rule_library": str(library_path),
        "evidence_consistency_calibration": bridge.calibration,
        "runtime_requires_llm": False,
    })
    _dump(output / "normality_rules_v10_evidence_consistent.json", effective_library)

    samples = collect_test_samples(class_root)
    expected_paths = np.asarray([str(Path(row["path"]).resolve()) for row in samples])
    if not np.array_equal(query_cache["paths"], expected_paths):
        raise ValueError("frozen INP query cache path/order mismatch")
    calibration_set = set(validation_ids)
    if calibration_set.intersection(set(expected_paths.tolist())):
        raise ValueError("normal calibration overlaps query/test data")

    labels = np.asarray([int(row["label"]) for row in samples], dtype=np.uint8)
    defect_types = np.asarray([str(row["defect_type"]) for row in samples])
    baseline_maps = np.clip(query_cache["maps"], 0.0, 1.0)
    baseline_scores = np.asarray([
        top_fraction_mean(score_map) for score_map in baseline_maps
    ], dtype=np.float64)
    if not np.allclose(baseline_scores, query_cache["scores"], atol=1e-6):
        raise ValueError("cached INP scores do not use the expected top-1% pooling")

    masks, raw_scores, confirmed_scores, fused_scores, legacy_scores = [], [], [], [], []
    raw_maps, confirmed_maps, fused_maps, legacy_maps = [], [], [], []
    rows, traces, visual_manifest = [], [], []
    explanations_dir = output / "query_explanations"
    explanations_dir.mkdir(parents=True, exist_ok=True)
    for index, (sample, baseline_score, baseline_map) in enumerate(
        zip(samples, baseline_scores, baseline_maps)
    ):
        raw_result = engine.evaluate(sample["path"])
        decision = bridge.infer(
            sample["path"], baseline_map, logic_result=raw_result
        )
        legacy_map, _ = execute_grounded_neural_symbolic(
            baseline_map, raw_result.symbolic_map
        )
        legacy_map = np.asarray(legacy_map, dtype=np.float32)
        mask = _mask_for_sample(sample, class_root)
        masks.append(mask)
        raw_scores.append(float(raw_result.symbolic_anomaly))
        confirmed_scores.append(float(decision.confirmed_symbolic_anomaly))
        fused_scores.append(float(decision.fused_score))
        legacy_scores.append(top_fraction_mean(legacy_map))
        raw_maps.append(raw_result.symbolic_map)
        confirmed_maps.append(decision.confirmed_symbolic_map)
        fused_maps.append(decision.fused_map)
        legacy_maps.append(legacy_map)

        prefix = f"{index:03d}_{Path(sample['path']).parent.name}_{Path(sample['path']).stem}"
        explanation_path = explanations_dir / f"{prefix}.txt"
        text_lines = [
            f"Query: {Path(sample['path']).resolve()}",
            f"INP-Former score: {baseline_score:.6f}",
            f"Raw symbolic score: {raw_result.symbolic_anomaly:.6f}",
            f"Evidence-confirmed symbolic score: {decision.confirmed_symbolic_anomaly:.6f}",
            f"Final fused score: {decision.fused_score:.6f}",
            f"Conclusion: {decision.explanation}",
            "Rules:",
        ]
        for name, evidence in decision.rule_evidence.items():
            text_lines.append(
                f"- {name}: status={evidence['status']}; "
                f"{evidence['fact']}={evidence['fact_value']:.6f}; "
                f"normal_upper={evidence['normal_upper']:.6f}; "
                f"violation={evidence['rule_violation']:.6f}; "
                f"INP_region_evidence={evidence['localized_inp_evidence']:.6f}; "
                f"INP_normal_upper={evidence['normal_inp_evidence_threshold']:.6f}; "
                f"agreement_gate={evidence['neural_confirmation_gate']:.6f}"
            )
        explanation_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

        trace = {
            "path": str(Path(sample["path"]).resolve()),
            "defect_type": str(sample["defect_type"]),
            "baseline_score": float(baseline_score),
            "raw_symbolic_score": float(raw_result.symbolic_anomaly),
            "confirmed_symbolic_score": float(decision.confirmed_symbolic_anomaly),
            "final_fused_score": float(decision.fused_score),
            "facts": raw_result.facts,
            "raw_atoms": raw_result.atoms,
            "rule_evidence": decision.rule_evidence,
            "explanation": decision.explanation,
            "explanation_file": str(explanation_path),
        }
        traces.append(trace)
        rows.append({
            "path": trace["path"],
            "defect_type": trace["defect_type"],
            "label": int(sample["label"]),
            "baseline_score": float(baseline_score),
            "raw_symbolic_score": float(raw_result.symbolic_anomaly),
            "confirmed_symbolic_score": float(decision.confirmed_symbolic_anomaly),
            "fused_score": float(decision.fused_score),
            "confirmed_rules": "|".join(
                name for name, evidence in decision.rule_evidence.items()
                if evidence["status"] == "neural_symbolic_confirmed"
            ) or "none",
            "explanation": str(explanation_path),
        })
        if args.max_visualizations <= 0 or index < args.max_visualizations:
            visual_manifest.append(_save_v10_visualization(
                output / "visualizations", sample, mask, baseline_map,
                raw_result.symbolic_map, decision.confirmed_symbolic_map,
                decision.fused_map, index,
            ))

    masks = np.asarray(masks, dtype=bool)
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    confirmed_scores = np.asarray(confirmed_scores, dtype=np.float64)
    fused_scores = np.asarray(fused_scores, dtype=np.float64)
    legacy_scores = np.asarray(legacy_scores, dtype=np.float64)
    raw_maps = np.asarray(raw_maps, dtype=np.float32)
    confirmed_maps = np.asarray(confirmed_maps, dtype=np.float32)
    fused_maps = np.asarray(fused_maps, dtype=np.float32)
    legacy_maps = np.asarray(legacy_maps, dtype=np.float32)

    baseline_metrics = _metrics(labels, baseline_scores, masks, baseline_maps)
    raw_metrics = _metrics(labels, raw_scores, masks, raw_maps)
    confirmed_metrics = _metrics(labels, confirmed_scores, masks, confirmed_maps)
    legacy_metrics = _metrics(labels, legacy_scores, masks, legacy_maps)
    fused_metrics = _metrics(labels, fused_scores, masks, fused_maps)
    metrics = {
        "schema_version": "10.0-evidence-consistent-controlled-experiment",
        "dataset": "mvtec",
        "class_name": "transistor",
        "protocol": {
            "name": protocol_name,
            "strict_k_shot": bool(strict_k_shot),
            "baseline_shot": int(query_cache["metadata"].get("shot", 4)),
            "rule_discovery_normal_count": int(len(library["support"])),
            "normal_validation_count": int(len(validation_ids)),
            "validation_source": args.validation_source,
            "validation_cache": str(validation_cache_path),
            "query_or_test_used_for_calibration": False,
            "test_labels_used_for_selection": False,
        },
        "llm": library.get("llm", {}),
        "symbolic_ast": library["logic"]["symbolic_anomaly_ast"],
        "active_rules": engine.active_ids,
        "accepted_after_normal_validation": [
            name for name, state in bridge.rule_calibration.items()
            if state["accepted"]
        ],
        "abstained_after_normal_validation": [
            name for name, state in bridge.rule_calibration.items()
            if not state["accepted"]
        ],
        "baseline": baseline_metrics,
        "raw_symbolic_only": raw_metrics,
        "confirmed_symbolic_only": confirmed_metrics,
        "legacy_same-map_multiplication": legacy_metrics,
        "evidence_consistent_neuro_symbolic": fused_metrics,
        "delta_v10_minus_baseline": _delta(fused_metrics, baseline_metrics),
        "delta_v10_minus_legacy": _delta(fused_metrics, legacy_metrics),
        "per_defect_type": _per_defect(
            labels, defect_types, fused_scores, baseline_scores,
            masks, fused_maps, baseline_maps,
        ),
        "normal_confirmed_rule_rate": float(np.mean(confirmed_scores[labels == 0] > 1e-8)),
        "anomaly_confirmed_rule_rate": float(np.mean(confirmed_scores[labels == 1] > 1e-8)),
        "base_rule_library": str(library_path),
        "bridge_calibration": str(output / "bridge_calibration_v10.json"),
        "runtime_requires_llm": False,
    }
    with (output / "predictions_v10.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _dump(output / "rule_evidence_v10.json", traces)
    _dump(output / "visualization_manifest_v10.json", visual_manifest)
    np.savez_compressed(
        output / "pixel_predictions_v10.npz",
        paths=expected_paths,
        labels=labels,
        masks=masks,
        baseline_maps=baseline_maps,
        raw_symbolic_maps=raw_maps,
        confirmed_symbolic_maps=confirmed_maps,
        legacy_maps=legacy_maps,
        fused_maps=fused_maps,
    )
    _dump(output / "metrics_v10.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normal-validated evidence-consistent neuro-symbolic FSAD"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--class-name", default="transistor")
    parser.add_argument("--rule-library", required=True)
    parser.add_argument("--prediction-cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--validation-source", choices=["remaining_train", "support_augment"],
        default="remaining_train",
    )
    parser.add_argument("--validation-count", type=int, default=32)
    parser.add_argument("--validation-seed", type=int, default=20261019)
    parser.add_argument("--support-augmentation-cache")
    parser.add_argument("--fact-quantile", type=float, default=0.95)
    parser.add_argument("--evidence-quantile", type=float, default=0.95)
    parser.add_argument("--maximum-validation-fpr", type=float, default=0.10)
    parser.add_argument("--max-threshold-shift-scales", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inpformer-root", default=DEFAULT_INPFORMER_ROOT)
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--map-size", type=int, default=256)
    parser.add_argument("--max-visualizations", type=int, default=100)
    return parser


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
