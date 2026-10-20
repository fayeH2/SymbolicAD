"""Controlled executable-logic experiment on MVTec transistor.

The experiment consumes frozen INP-Former predictions, compiles a static rule
library from normal supports only, executes every accepted rule through an
AND/OR/NOT AST, and evaluates image- and pixel-level AUROC/AUPR.  Test labels
are read only after compilation and are never used for thresholds or rule
selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import collect_test_samples, select_inpformer_support
from .transistor_logic import (
    FINAL_FUSION_AST,
    TransistorLogicEngine,
    compile_transistor_logic_library,
    execute_grounded_neural_symbolic,
    preprocess_mask,
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
            "scores": payload["scores"].astype(np.float64),
            "maps": payload["maps"].astype(np.float32),
            "metadata": json.loads(str(payload["metadata_json"].item())),
        }


def _metric_pair(labels: np.ndarray, scores: np.ndarray, prefix: str) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return {
        f"{prefix}_auroc": float(roc_auc_score(labels, scores)),
        f"{prefix}_aupr": float(average_precision_score(labels, scores)),
    }


def _all_metrics(
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    pixel_labels: np.ndarray,
    pixel_scores: np.ndarray,
) -> Dict[str, float]:
    return {
        **_metric_pair(image_labels, image_scores, "i"),
        **_metric_pair(pixel_labels, pixel_scores, "p"),
    }


def _delta(candidate: Mapping[str, float], baseline: Mapping[str, float]) -> Dict[str, float]:
    return {key: float(candidate[key] - baseline[key]) for key in baseline}


def _mask_for_sample(sample: Mapping[str, Any], class_root: Path) -> np.ndarray:
    if int(sample["label"]) == 0:
        return np.zeros((256, 256), dtype=bool)
    image_path = Path(sample["path"])
    defect = image_path.parent.name
    candidates = [
        class_root / "ground_truth" / defect / f"{image_path.stem}_mask.png",
        # VisA keeps .JPG queries but stores same-stem masks as .png.
        class_root / "ground_truth" / defect / f"{image_path.stem}.png",
        class_root / "ground_truth" / defect / image_path.name,
        class_root / "test" / "ground_truth" / defect / f"{image_path.stem}_mask.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return preprocess_mask(candidate)
    raise FileNotFoundError(
        f"Pixel mask missing for anomalous query {image_path}; tried {candidates}"
    )


def _heat_overlay(image: np.ndarray, score_map: np.ndarray) -> np.ndarray:
    normalized = np.clip(np.asarray(score_map, dtype=np.float32), 0.0, 1.0)
    heat = cv2.applyColorMap(
        np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_JET
    )
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 0.52, heat, 0.48, 0.0)


def _label_panel(panel: np.ndarray, text: str) -> np.ndarray:
    result = panel.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 27), (0, 0, 0), -1)
    cv2.putText(
        result, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return result


def _save_visualization(
    directory: Path,
    sample: Mapping[str, Any],
    mask: np.ndarray,
    baseline_map: np.ndarray,
    symbolic_map: np.ndarray,
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
    baseline = _heat_overlay(original, baseline_map)
    symbolic = _heat_overlay(original, symbolic_map)
    fused = _heat_overlay(original, fused_map)
    prefix = f"{index:03d}_{image_path.parent.name}_{image_path.stem}"
    outputs = {
        "original": directory / f"{prefix}_original.jpg",
        "gt": directory / f"{prefix}_gt.jpg",
        "baseline": directory / f"{prefix}_baseline_INPFormer.jpg",
        "symbolic": directory / f"{prefix}_symbolic_rules.jpg",
        "fused": directory / f"{prefix}_fused_NeuroSymbolic.jpg",
        "comparison": directory / f"{prefix}_comparison.jpg",
    }
    for key, value in {
        "original": original, "gt": gt, "baseline": baseline,
        "symbolic": symbolic, "fused": fused,
    }.items():
        Image.fromarray(value).save(outputs[key], quality=95)
    comparison = np.concatenate([
        _label_panel(original, "Original"),
        _label_panel(gt, "Pixel GT"),
        _label_panel(baseline, "INP-Former baseline"),
        _label_panel(symbolic, "Executed rule violations"),
        _label_panel(fused, "Grounded AND/OR neuro-symbolic"),
    ], axis=1)
    Image.fromarray(comparison).save(outputs["comparison"], quality=95)
    return {key: str(value) for key, value in outputs.items()}


def _candidate_bank(path: str | None) -> Mapping[str, Any] | None:
    if not path:
        return None
    candidate = Path(path).expanduser().resolve()
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not bool(payload.get("complete", False)):
        raise ValueError(f"Qwen candidate bank is incomplete: {candidate}")
    return payload


def _rule_support(args: argparse.Namespace) -> tuple[list[Path], Dict[str, Any]]:
    """Resolve either a shot-specific split or a prefix of one frozen order."""
    if not args.rule_support_manifest:
        paths = select_inpformer_support(
            args.data_root, args.dataset, args.class_name,
            args.rule_support_count, args.support_seed,
        )
        return [Path(path).expanduser().resolve() for path in paths], {
            "mode": "shot_specific_inpformer_sampler",
            "manifest": None,
            "nested_prefix": False,
        }
    manifest_path = Path(args.rule_support_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, expected in {
        "dataset": args.dataset,
        "class_name": args.class_name,
        "support_seed": args.support_seed,
    }.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Rule support manifest mismatch for {key}: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    entries = sorted(
        manifest.get("support", []), key=lambda row: int(row["sample_index"])
    )
    if len(entries) < args.rule_support_count:
        raise ValueError(
            f"Manifest has {len(entries)} supports, requested "
            f"{args.rule_support_count}"
        )
    entries = entries[:args.rule_support_count]
    paths = [
        Path(str(row["image_path"])).expanduser().resolve() for row in entries
    ]
    if len(set(paths)) != len(paths) or not all(path.is_file() for path in paths):
        raise ValueError("Rule support manifest contains duplicate or missing images")
    return paths, {
        "mode": "nested_prefix_manifest",
        "manifest": str(manifest_path),
        "source_support_count": int(
            manifest.get("source_support_count", len(manifest.get("support", [])))
        ),
        "nested_prefix": True,
        "selected_sample_ids": [str(row["sample_id"]) for row in entries],
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.dataset != "mvtec" or args.class_name != "transistor":
        raise ValueError("v7 controlled implementation currently targets mvtec/transistor")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).expanduser().resolve()
    class_root = data_root / args.class_name

    # Freeze the symbolic library before opening any query cache, directory, or
    # ground-truth path.  This ordering makes accidental test-guided selection
    # impossible inside the experiment implementation.
    rule_support, support_selection = _rule_support(args)
    bank = _candidate_bank(args.candidate_bank)
    library = compile_transistor_logic_library(
        rule_support,
        candidate_bank=bank,
        dataset=args.dataset,
        class_name=args.class_name,
    )
    library["support_selection"] = support_selection

    cache_dir = Path(args.prediction_cache_dir).expanduser().resolve()
    if (cache_dir / "cache").is_dir():
        cache_dir = cache_dir / "cache"
    test_cache = _read_cache(cache_dir / "test_inp_v4.npz")
    metadata = test_cache["metadata"]
    for key, expected in {
        "dataset": args.dataset,
        "class_name": args.class_name,
        "shot": args.baseline_shot,
    }.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Baseline cache metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {expected!r}"
            )

    samples = collect_test_samples(class_root)
    expected_paths = np.asarray([str(Path(row["path"]).resolve()) for row in samples])
    if not np.array_equal(test_cache["paths"], expected_paths):
        raise ValueError("Test cache path/order mismatch")

    library["baseline"] = {
        "name": "INP-Former",
        "checkpoint": metadata.get("checkpoint"),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "shot": args.baseline_shot,
        "neural_truth": "raw frozen anomaly score/map clipped to [0,1]",
        "adapter_config": {
            key: metadata.get(key) for key in (
                "inpformer_root", "encoder", "inp_num", "input_size",
                "crop_size", "map_size",
            )
        },
    }
    library["final_logic"] = {
        "formula": (
            "OR(INPFormerAnomaly, "
            "AND(NeuralGroundingConfidence, SymbolicAnomaly))"
        ),
        "numeric_form": "OR(a,AND(a,s)); AND=x*y; OR=x+y-x*y",
        "neural_grounding_source": (
            "the same frozen INP-Former anomaly truth, used as a distinct "
            "logic predicate to prevent unsupported symbolic-only hot regions"
        ),
        "ast": FINAL_FUSION_AST,
        "learned_or_searched_fusion_weight": False,
    }
    library_path = output / "normality_rules_v7_executable_logic.json"
    _dump(library_path, library)
    _dump(output / "candidate_stability_audit_v7.json", library["candidate_audit"])

    engine = TransistorLogicEngine(library)
    image_labels = np.asarray([int(row["label"]) for row in samples], dtype=np.uint8)
    baseline_scores = np.clip(test_cache["scores"], 0.0, 1.0)
    baseline_maps = np.clip(test_cache["maps"], 0.0, 1.0)
    symbolic_scores = []
    fused_scores = []
    symbolic_maps = []
    fused_maps = []
    masks = []
    evidence_rows = []
    prediction_rows = []
    visualization_rows = []

    for index, (sample, baseline_score, baseline_map) in enumerate(
        zip(samples, baseline_scores, baseline_maps)
    ):
        inference = engine.evaluate(sample["path"])
        symbolic_score = inference.symbolic_anomaly
        fused_score, final_scalar_trace = execute_grounded_neural_symbolic(
            float(baseline_score), symbolic_score
        )
        fused_score = float(fused_score)
        fused_map = np.asarray(
            execute_grounded_neural_symbolic(
                baseline_map, inference.symbolic_map
            )[0],
            dtype=np.float32,
        )
        mask = _mask_for_sample(sample, class_root)
        symbolic_scores.append(symbolic_score)
        fused_scores.append(fused_score)
        symbolic_maps.append(inference.symbolic_map)
        fused_maps.append(fused_map)
        masks.append(mask)
        violated = [
            atom_id for atom_id, row in inference.atoms.items()
            if float(row["violation"]) > 1e-8
        ]
        trace_path = output / "logic_traces" / (
            f"{index:03d}_{Path(sample['path']).parent.name}_{Path(sample['path']).stem}.json"
        )
        trace_payload = {
            "path": str(Path(sample["path"]).resolve()),
            "defect_type": str(sample["defect_type"]),
            "formula": library["logic"]["formula"],
            "facts": inference.facts,
            "atoms": inference.atoms,
            "symbolic_logic_nodes": inference.scalar_trace,
            "final_fusion_logic_nodes": final_scalar_trace,
            "baseline_anomaly": float(baseline_score),
            "symbolic_anomaly": symbolic_score,
            "final_logic_or": fused_score,
            "violated_rules": violated,
            "explanation": inference.explanation,
            "diagnostics": inference.diagnostics,
        }
        _dump(trace_path, trace_payload)
        evidence_rows.append({
            "path": trace_payload["path"],
            "atoms": inference.atoms,
            "baseline_anomaly": float(baseline_score),
            "symbolic_anomaly": symbolic_score,
            "final_logic_or": fused_score,
            "violated_rules": violated,
            "explanation": inference.explanation,
            "trace": str(trace_path),
        })
        prediction_rows.append({
            "path": trace_payload["path"],
            "defect_type": str(sample["defect_type"]),
            "label": int(sample["label"]),
            "baseline_score": float(baseline_score),
            "symbolic_score": symbolic_score,
            "fused_score": fused_score,
            "violated_rule_count": len(violated),
            "violated_rules": "|".join(violated) if violated else "none",
            "explanation": inference.explanation,
            "trace": str(trace_path),
        })
        if args.max_visualizations <= 0 or index < args.max_visualizations:
            visualization_rows.append(_save_visualization(
                output / "visualizations", sample, mask, baseline_map,
                inference.symbolic_map, fused_map, index,
            ))

    symbolic_scores = np.asarray(symbolic_scores, dtype=np.float64)
    fused_scores = np.asarray(fused_scores, dtype=np.float64)
    symbolic_maps = np.asarray(symbolic_maps, dtype=np.float32)
    fused_maps = np.asarray(fused_maps, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    baseline_metrics = _all_metrics(
        image_labels, baseline_scores, masks, baseline_maps
    )
    symbolic_metrics = _all_metrics(
        image_labels, symbolic_scores, masks, symbolic_maps
    )
    fused_metrics = _all_metrics(
        image_labels, fused_scores, masks, fused_maps
    )

    defect_types = np.asarray([str(row["defect_type"]) for row in samples])
    per_defect = {}
    for defect in sorted(set(defect_types) - {"good", "ok"}):
        selected = np.isin(defect_types, ["good", "ok", defect])
        per_defect[defect] = {
            "anomaly_count": int(np.sum(defect_types == defect)),
            "baseline_i_auroc": float(
                roc_auc_score(image_labels[selected], baseline_scores[selected])
            ),
            "symbolic_i_auroc": float(
                roc_auc_score(image_labels[selected], symbolic_scores[selected])
            ),
            "fused_i_auroc": float(
                roc_auc_score(image_labels[selected], fused_scores[selected])
            ),
        }

    result = {
        "schema_version": "7.0-controlled-executable-logic-experiment",
        "experiment_status": "mechanism_development_on_mvtec_transistor",
        "dataset": args.dataset,
        "class_name": args.class_name,
        "baseline_shot": args.baseline_shot,
        "rule_support_count": args.rule_support_count,
        "support_seed": args.support_seed,
        "support_selection": support_selection,
        "test_count": len(samples),
        "normal_count": int(np.sum(image_labels == 0)),
        "anomaly_count": int(np.sum(image_labels == 1)),
        "candidate_rule_count": library["candidate_rule_count"],
        "groundable_rule_count": library["groundable_rule_count"],
        "accepted_rule_count": library["accepted_rule_count"],
        "accepted_rule_ids": library["accepted_rule_ids"],
        "logic_formula": library["logic"]["formula"],
        "fusion": {
            "operator": "product_fuzzy_AND_OR_with_neural_grounding",
            "formula": "OR(a,AND(a,s))",
            "alpha_or_rule_weights": False,
        },
        "baseline": baseline_metrics,
        "symbolic_only": symbolic_metrics,
        "neuro_symbolic": fused_metrics,
        "delta_neuro_symbolic_minus_baseline": _delta(fused_metrics, baseline_metrics),
        "per_defect_type": per_defect,
        "normal_symbolic_trigger_rate": float(
            np.mean(symbolic_scores[image_labels == 0] > 1e-8)
        ),
        "anomaly_symbolic_trigger_rate": float(
            np.mean(symbolic_scores[image_labels == 1] > 1e-8)
        ),
        "mean_normal_symbolic_score": float(
            np.mean(symbolic_scores[image_labels == 0])
        ),
        "mean_anomaly_symbolic_score": float(
            np.mean(symbolic_scores[image_labels == 1])
        ),
        "rule_library": str(library_path),
        "candidate_bank": str(Path(args.candidate_bank).resolve()) if args.candidate_bank else None,
        "prediction_cache": str(cache_dir),
        "visualization_count": len(visualization_rows),
        "visualization_scale": "fixed [0,1] shared by baseline, symbolic, and fused maps",
        "query_text_explanations": True,
        "test_labels_used_for_rule_selection_or_calibration": False,
        "fusion_formula_developed_on_same_category": True,
        "strict_unseen_category_confirmation": False,
        "runtime_requires_llm": False,
    }
    if args.old_weighted_metrics:
        old_path = Path(args.old_weighted_metrics).expanduser().resolve()
        old = json.loads(old_path.read_text(encoding="utf-8"))
        old_image = old.get("baseline_plus_rules", {})
        result["historical_weighted_sum_control"] = {
            "source": str(old_path),
            "method": "baseline + alpha * sum(weight * rule_violation)",
            "i_auroc": old_image.get("i_auroc"),
            "i_aupr": old_image.get("i_aupr"),
            "p_auroc": baseline_metrics["p_auroc"],
            "p_aupr": baseline_metrics["p_aupr"],
            "pixel_note": "historical rules had no spatial map; pixel output equals baseline",
        }

    with (output / "predictions_v7_executable_logic.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    _dump(output / "rule_evidence_v7.json", evidence_rows)
    _dump(output / "visualization_manifest_v7.json", visualization_rows)
    np.savez_compressed(
        output / "pixel_predictions_v7.npz",
        paths=expected_paths,
        labels=image_labels,
        masks=masks,
        baseline_maps=baseline_maps,
        symbolic_maps=symbolic_maps,
        fused_maps=fused_maps,
    )
    _dump(output / "metrics_v7_executable_logic.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mvtec")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--class-name", default="transistor")
    parser.add_argument("--baseline-shot", type=int, default=4)
    parser.add_argument("--rule-support-count", type=int, default=30)
    parser.add_argument("--support-seed", type=int, default=1)
    parser.add_argument(
        "--rule-support-manifest",
        help=(
            "ordered larger-support manifest; use its first N entries for "
            "strictly nested support controls"
        ),
    )
    parser.add_argument("--candidate-bank")
    parser.add_argument("--prediction-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-visualizations", type=int, default=100)
    parser.add_argument("--old-weighted-metrics")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
