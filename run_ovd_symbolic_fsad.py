#!/usr/bin/env python3
"""V12: Qwen-guided GroundingDINO symbolic reasoning fused with INP-Former."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from neurosymbolic_fsad.data import collect_test_samples, select_inpformer_support
from neurosymbolic_fsad.experiment_logic import _heat_overlay, _label_panel, _mask_for_sample
from neurosymbolic_fsad.features import prepare_inp_aligned_rgb
from neurosymbolic_fsad.inpformer_adapter import (
    DEFAULT_INPFORMER_ROOT, INPFormerAdapter,
    load_prediction_cache, save_prediction_cache, file_sha256,
)
from neurosymbolic_fsad.orthogonal_symbolic import RuleConditionedINPFusion
from neurosymbolic_fsad.ovd_grounder import (
    DEFAULT_BERT, DEFAULT_GROUNDING_DINO_CONFIG, DEFAULT_GROUNDING_DINO_ROOT,
    DEFAULT_GROUNDING_DINO_WEIGHTS, GroundingDINOAdapter,
    OpenVocabularySymbolicEngine, compile_ovd_program,
)
from neurosymbolic_fsad.qwen_generic_relations import validate_generic_relational_proposal


def _default(value: Any):
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
        json.dumps(payload, ensure_ascii=False, indent=2, default=_default) + "\n",
        encoding="utf-8",
    )


def _metrics(labels, scores, masks, maps):
    return {
        "i_auroc": float(roc_auc_score(labels, scores)),
        "i_aupr": float(average_precision_score(labels, scores)),
        "p_auroc": float(roc_auc_score(masks.reshape(-1), maps.reshape(-1))),
        "p_aupr": float(average_precision_score(masks.reshape(-1), maps.reshape(-1))),
    }


def _delta(candidate: Mapping[str, float], baseline: Mapping[str, float]):
    return {key: float(candidate[key] - baseline[key]) for key in baseline}


def _box_panel(image: np.ndarray, diagnostics: Mapping[str, Any]) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    for index, row in enumerate(diagnostics.get("detections", [])):
        x0, y0, x1, y1 = row["box"]
        start = (int(x0 * width), int(y0 * height))
        stop = (int(x1 * width), int(y1 * height))
        cv2.rectangle(output, start, stop, (0, 255, 0), 2)
        cv2.putText(
            output, f"{index}:{row['score']:.2f}",
            (start[0], max(12, start[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return output


def _visualize(directory, sample, mask, baseline, symbolic, fused, diagnostics, index):
    directory.mkdir(parents=True, exist_ok=True)
    aligned = prepare_inp_aligned_rgb(sample["path"])
    original = np.asarray(Image.fromarray(aligned).resize(
        (baseline.shape[1], baseline.shape[0]), Image.Resampling.BILINEAR
    ))
    gt = original.copy()
    if np.any(mask):
        contours = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        cv2.drawContours(gt, contours, -1, (255, 0, 0), 2)
    panels = [
        _label_panel(original, "Original"),
        _label_panel(gt, "Pixel GT"),
        _label_panel(_box_panel(original, diagnostics), "GroundingDINO entities"),
        _label_panel(_heat_overlay(original, baseline), "INP baseline"),
        _label_panel(_heat_overlay(original, symbolic), "OVD symbolic logic"),
        _label_panel(_heat_overlay(original, fused), "OVD-rule INP fusion"),
    ]
    path = directory / f"{index:03d}_{Path(sample['path']).parent.name}_{Path(sample['path']).stem}.jpg"
    Image.fromarray(np.concatenate(panels, axis=1)).save(path, quality=95)
    return str(path)


def run(args):
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).expanduser().resolve()
    class_root = data_root / args.class_name
    supports = select_inpformer_support(
        data_root, args.dataset, args.class_name, args.shot, args.support_seed
    )
    raw_proposal = json.loads(Path(args.proposal_json).read_text(encoding="utf-8"))
    proposal = validate_generic_relational_proposal(raw_proposal)
    proposal["model"] = raw_proposal.get("model", args.llm_model)

    detector = GroundingDINOAdapter(
        device=args.ovd_device, root=args.ovd_root, config=args.ovd_config,
        weights=args.ovd_weights, bert=args.bert,
        text_threshold=args.text_threshold, nms_iou=args.nms_iou,
    )
    library, compiler_audit = compile_ovd_program(
        detector, proposal, supports, image_size=args.map_size,
        search_seed=args.search_seed,
    )
    library.update({
        "dataset": args.dataset, "class_name": args.class_name,
        "shot": args.shot, "support": [str(path.resolve()) for path in supports],
        "llm": {
            "model": proposal.get("model", args.llm_model),
            "proposal": proposal,
            "role": "offline semantic seed; not called during query inference",
        },
        "open_vocabulary_detector": {
            "name": "GroundingDINO-SwinT", "weights": str(Path(args.ovd_weights).resolve()),
            "device": args.ovd_device,
        },
    })
    _dump(output / "normality_rules_v12_ovd.json", library)
    _dump(output / "compiler_audit_v12.json", compiler_audit)

    samples = collect_test_samples(class_root, test_subdir=args.test_subdir)
    if args.max_queries > 0:
        samples = samples[:args.max_queries]
    test_paths = [Path(row["path"]) for row in samples]
    metadata = {
        "schema": "inp-layer-residual-v11", "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "dataset": args.dataset, "class_name": args.class_name, "shot": args.shot,
        "support_seed": args.support_seed, "encoder": args.encoder,
        "inp_num": args.inp_num, "input_size": args.input_size,
        "crop_size": args.crop_size, "map_size": args.map_size,
        "support_sha256": [file_sha256(path) for path in supports],
        "support_selector_schema": "inp-support-v2",
    }
    support_ids = [str(path.resolve()) for path in supports]
    test_ids = [str(path.resolve()) for path in test_paths]
    cache_root = Path(args.prediction_cache_dir).resolve() if args.prediction_cache_dir else output
    support_cache = cache_root / "cache" / "support_inp_layers_v11.npz"
    test_cache = cache_root / "cache" / "test_inp_layers_v11.npz"
    support_predictions = load_prediction_cache(
        support_cache, support_ids, expected_metadata=metadata
    )
    test_predictions = load_prediction_cache(
        test_cache, test_ids, expected_metadata=metadata
    )
    if support_predictions is None or "layer_maps" not in support_predictions:
        adapter = INPFormerAdapter(
            args.checkpoint, args.inp_device, args.inpformer_root,
            args.encoder, args.inp_num, args.input_size, args.crop_size, args.map_size,
        )
        support_predictions = adapter.infer_images(
            supports, args.batch_size, support_ids, return_layer_maps=True
        )
        save_prediction_cache(support_cache, support_predictions, metadata)
        adapter.close(); del adapter; gc.collect()
    if test_predictions is None or "layer_maps" not in test_predictions:
        adapter = INPFormerAdapter(
            args.checkpoint, args.inp_device, args.inpformer_root,
            args.encoder, args.inp_num, args.input_size, args.crop_size, args.map_size,
        )
        test_predictions = adapter.infer_images(
            test_paths, args.batch_size, test_ids, return_layer_maps=True
        )
        save_prediction_cache(test_cache, test_predictions, metadata)
        adapter.close(); del adapter; gc.collect()

    engine = OpenVocabularySymbolicEngine(library, detector)
    fusion = RuleConditionedINPFusion.calibrate(
        engine, supports, support_predictions["layer_maps"]
    )
    _dump(output / "feature_fusion_calibration_v12.json", fusion.calibration)
    labels = np.asarray([row["label"] for row in samples], dtype=np.uint8)
    baseline_scores = np.asarray(test_predictions["scores"], dtype=float)
    baseline_maps = np.asarray(test_predictions["maps"], dtype=np.float32)
    masks, fused_scores, fused_maps, symbolic_scores, symbolic_maps = [], [], [], [], []
    rows, evidence, visuals = [], [], []
    explanations = output / "query_explanations"
    explanations.mkdir(parents=True, exist_ok=True)
    for index, (sample, base_score, base_map, layer_maps) in enumerate(zip(
        samples, baseline_scores, baseline_maps, test_predictions["layer_maps"]
    )):
        score, fused_map, symbolic, trace = fusion.infer(
            sample["path"], layer_maps,
            baseline_map=base_map, baseline_score=float(base_score),
        )
        mask = _mask_for_sample(sample, class_root)
        masks.append(mask); fused_scores.append(score); fused_maps.append(fused_map)
        symbolic_scores.append(symbolic.symbolic_anomaly); symbolic_maps.append(symbolic.symbolic_map)
        triggered = [
            name for name, row in trace.get("rules", {}).items()
            if float(row["supported_violation"]) > 1e-8
        ]
        explanation = explanations / f"{index:03d}_{Path(sample['path']).parent.name}_{Path(sample['path']).stem}.txt"
        text = [
            f"Query: {Path(sample['path']).resolve()}",
            f"OVD prompt: {symbolic.diagnostics.get('prompt')}",
            f"Detected entities: {symbolic.diagnostics.get('detected_count', 0)}",
            f"Baseline: {base_score:.6f}", f"Fused: {score:.6f}",
            f"AST: {json.dumps((library.get('logic') or {}).get('anomaly_ast'), ensure_ascii=False)}",
        ]
        for name, atom in symbolic.atoms.items():
            gate = trace.get("rules", {}).get(name, {}).get("aggregate_gate", 0.0)
            text.append(
                f"- {name}: fact={atom['fact_value']:.6f} upper={atom['upper']:.6f} "
                f"violation={atom['violation']:.6f} INP_gate={gate:.6f}"
            )
        explanation.write_text("\n".join(text) + "\n", encoding="utf-8")
        rows.append({
            "path": str(Path(sample["path"]).resolve()),
            "defect_type": sample["defect_type"], "label": int(sample["label"]),
            "ovd_count": symbolic.diagnostics.get("detected_count", 0),
            "baseline_score": float(base_score), "symbolic_score": symbolic.symbolic_anomaly,
            "fused_score": float(score), "triggered_rules": "|".join(triggered) or "none",
        })
        evidence.append({
            "path": rows[-1]["path"], "facts": symbolic.facts,
            "atoms": symbolic.atoms, "diagnostics": symbolic.diagnostics,
            "logic_trace": symbolic.trace, "feature_fusion_trace": trace,
            "explanation": str(explanation),
        })
        # Always retain rule-triggered cases even when routine visualization is
        # capped, so every textual symbolic decision has an inspectable image.
        if (
            args.max_visualizations <= 0
            or index < args.max_visualizations
            or bool(triggered)
        ):
            visuals.append(_visualize(
                output / "visualizations", sample, mask, base_map,
                symbolic.symbolic_map, fused_map, symbolic.diagnostics, index,
            ))
    masks = np.asarray(masks, dtype=bool)
    fused_scores = np.asarray(fused_scores, dtype=float)
    fused_maps = np.asarray(fused_maps, dtype=np.float32)
    symbolic_scores = np.asarray(symbolic_scores, dtype=float)
    symbolic_maps = np.asarray(symbolic_maps, dtype=np.float32)
    baseline_metrics = _metrics(labels, baseline_scores, masks, baseline_maps)
    fused_metrics = _metrics(labels, fused_scores, masks, fused_maps)
    symbolic_metrics = _metrics(labels, symbolic_scores, masks, symbolic_maps)
    result = {
        "schema_version": "12.0-qwen-groundingdino-evolution-inp-fusion",
        "dataset": args.dataset, "class_name": args.class_name, "shot": args.shot,
        "strict_few_shot": True, "query_or_test_used_for_rule_search": False,
        "category_specific_python_branching": False,
        "grounding": compiler_audit.get("grounding"),
        "rules": [{"id": row["id"], "predicate": row["predicate"]} for row in library.get("rules", [])],
        "evolved_anomaly_ast": (library.get("logic") or {}).get("anomaly_ast"),
        "baseline": baseline_metrics, "symbolic_only": symbolic_metrics,
        "ovd_rule_conditioned_inp_fusion": fused_metrics,
        "delta_fused_minus_baseline": _delta(fused_metrics, baseline_metrics),
        "rule_feature_trigger_rate_normal": float(np.mean([
            row["triggered_rules"] != "none" for row, label in zip(rows, labels) if label == 0
        ])),
        "rule_feature_trigger_rate_anomaly": float(np.mean([
            row["triggered_rules"] != "none" for row, label in zip(rows, labels) if label == 1
        ])),
    }
    with (output / "predictions_v12.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    _dump(output / "rule_evidence_v12.json", evidence)
    _dump(output / "visualization_manifest_v12.json", visuals)
    np.savez_compressed(
        output / "pixel_predictions_v12.npz", labels=labels, masks=masks,
        baseline_maps=baseline_maps, symbolic_maps=symbolic_maps, fused_maps=fused_maps,
    )
    _dump(output / "metrics_v12.json", result)
    detector.close()
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mvtec"); p.add_argument("--data-root", required=True)
    p.add_argument("--class-name", required=True); p.add_argument("--test-subdir", default="test")
    p.add_argument("--shot", type=int, default=4); p.add_argument("--support-seed", type=int, default=1)
    p.add_argument("--proposal-json", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--prediction-cache-dir")
    p.add_argument("--inp-device", default="cuda:0"); p.add_argument("--ovd-device", default="cuda:1")
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--inpformer-root", default=DEFAULT_INPFORMER_ROOT)
    p.add_argument("--encoder", default="dinov2reg_vit_base_14"); p.add_argument("--inp-num", type=int, default=6)
    p.add_argument("--input-size", type=int, default=448); p.add_argument("--crop-size", type=int, default=392)
    p.add_argument("--map-size", type=int, default=256); p.add_argument("--search-seed", type=int, default=668)
    p.add_argument("--ovd-root", default=DEFAULT_GROUNDING_DINO_ROOT)
    p.add_argument("--ovd-config", default=DEFAULT_GROUNDING_DINO_CONFIG)
    p.add_argument("--ovd-weights", default=DEFAULT_GROUNDING_DINO_WEIGHTS)
    p.add_argument("--bert", default=DEFAULT_BERT); p.add_argument("--text-threshold", type=float, default=0.15)
    p.add_argument("--nms-iou", type=float, default=0.45); p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--max-visualizations", type=int, default=20)
    p.add_argument("--llm-model", default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking")
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2, default=_default))
