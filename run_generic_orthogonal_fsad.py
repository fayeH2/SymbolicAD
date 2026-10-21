#!/usr/bin/env python3
"""Strict few-shot, category-agnostic V11 neuro-symbolic experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from neurosymbolic_fsad.data import collect_test_samples, select_inpformer_support
from neurosymbolic_fsad.experiment_logic import (
    _heat_overlay, _label_panel, _mask_for_sample,
)
from neurosymbolic_fsad.inpformer_adapter import (
    DEFAULT_INPFORMER_ROOT, INPFormerAdapter,
    load_prediction_cache, save_prediction_cache,
)
from neurosymbolic_fsad.features import prepare_inp_aligned_rgb
from neurosymbolic_fsad.orthogonal_symbolic import (
    OrthogonalSymbolicEngine, RuleConditionedINPFusion,
    compile_orthogonal_program,
)
from neurosymbolic_fsad.qwen_generic_relations import (
    validate_generic_relational_proposal,
)


def _default(value: Any):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def _dump(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_default) + "\n",
        encoding="utf-8",
    )


def _metrics(labels, scores, masks, maps) -> Dict[str, float]:
    return {
        "i_auroc": float(roc_auc_score(labels, scores)),
        "i_aupr": float(average_precision_score(labels, scores)),
        "p_auroc": float(roc_auc_score(masks.reshape(-1), maps.reshape(-1))),
        "p_aupr": float(average_precision_score(masks.reshape(-1), maps.reshape(-1))),
    }


def _delta(candidate: Mapping[str, float], baseline: Mapping[str, float]):
    return {key: float(candidate[key] - baseline[key]) for key in baseline}


def _visualize(directory, sample, mask, baseline, symbolic, fused, index):
    directory.mkdir(parents=True, exist_ok=True)
    image_path = Path(sample["path"])
    aligned = prepare_inp_aligned_rgb(image_path)
    original = np.asarray(
        Image.fromarray(aligned).resize(
            (int(baseline.shape[1]), int(baseline.shape[0])),
            Image.Resampling.BILINEAR,
        )
    )
    gt = original.copy()
    if np.any(mask):
        import cv2
        contours = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        cv2.drawContours(gt, contours, -1, (255, 0, 0), 2)
    base_panel = _heat_overlay(original, baseline)
    symbolic_panel = _heat_overlay(original, symbolic)
    fused_panel = _heat_overlay(original, fused)
    comparison = np.concatenate([
        _label_panel(original, "Original"), _label_panel(gt, "Pixel GT"),
        _label_panel(base_panel, "INP baseline residual mean"),
        _label_panel(symbolic_panel, "Evolved orthogonal logic"),
        _label_panel(fused_panel, "Rule-conditioned INP layer fusion"),
    ], axis=1)
    prefix = f"{index:03d}_{image_path.parent.name}_{image_path.stem}"
    path = directory / f"{prefix}_comparison.jpg"
    Image.fromarray(comparison).save(path, quality=95)
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
    for key in (
        "model", "consensus_protocol", "required_support_agreement",
        "count_votes", "predicate_support_votes", "per_support_proposals",
    ):
        if key in raw_proposal:
            proposal[key] = raw_proposal[key]
    library, compiler_audit = compile_orthogonal_program(
        proposal, supports, image_size=args.map_size,
        max_rules=args.max_rules,
        maximum_response_correlation=args.maximum_response_correlation,
        search_seed=args.search_seed,
    )
    library.update({
        "dataset": args.dataset, "class_name": args.class_name,
        "shot": args.shot, "support": [str(path.resolve()) for path in supports],
        "llm": {
            "model": proposal.get("model", args.llm_model),
            "proposal": proposal,
            "role": "closed-schema candidate proposal; no category-specific runtime code",
        },
    })
    _dump(output / "normality_rules_v11_orthogonal.json", library)
    _dump(output / "compiler_audit_v11.json", compiler_audit)

    samples = collect_test_samples(class_root, test_subdir=args.test_subdir)
    test_paths = [Path(row["path"]) for row in samples]
    metadata = {
        "schema": "inp-layer-residual-v11", "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": args.dataset, "class_name": args.class_name, "shot": args.shot,
        "support_seed": args.support_seed, "encoder": args.encoder,
        "inp_num": args.inp_num, "input_size": args.input_size,
        "crop_size": args.crop_size, "map_size": args.map_size,
    }
    support_ids = [str(path.resolve()) for path in supports]
    test_ids = [str(path.resolve()) for path in test_paths]
    cache_root = (
        Path(args.prediction_cache_dir).expanduser().resolve()
        if args.prediction_cache_dir else output
    )
    support_cache_path = cache_root / "cache" / "support_inp_layers_v11.npz"
    test_cache_path = cache_root / "cache" / "test_inp_layers_v11.npz"
    support_predictions = load_prediction_cache(
        support_cache_path, support_ids, expected_metadata=metadata
    )
    test_predictions = load_prediction_cache(
        test_cache_path, test_ids, expected_metadata=metadata
    )
    if (
        support_predictions is None or "layer_maps" not in support_predictions
        or test_predictions is None or "layer_maps" not in test_predictions
    ):
        adapter = INPFormerAdapter(
            checkpoint=args.checkpoint, device=args.device,
            inpformer_root=args.inpformer_root, encoder_name=args.encoder,
            inp_num=args.inp_num, input_size=args.input_size,
            crop_size=args.crop_size, map_size=args.map_size,
        )
        if support_predictions is None or "layer_maps" not in support_predictions:
            support_predictions = adapter.infer_images(
                supports, batch_size=args.batch_size, identifiers=support_ids,
                return_layer_maps=True,
            )
            save_prediction_cache(support_cache_path, support_predictions, metadata)
        if test_predictions is None or "layer_maps" not in test_predictions:
            test_predictions = adapter.infer_images(
                test_paths, batch_size=args.batch_size, identifiers=test_ids,
                return_layer_maps=True,
            )
            save_prediction_cache(test_cache_path, test_predictions, metadata)
        adapter.close(); del adapter; gc.collect()

    engine = OrthogonalSymbolicEngine(library)
    fusion = RuleConditionedINPFusion.calibrate(
        engine, supports, support_predictions["layer_maps"]
    )
    _dump(output / "feature_fusion_calibration_v11.json", fusion.calibration)

    labels = np.asarray([int(row["label"]) for row in samples], dtype=np.uint8)
    baseline_scores = np.asarray(test_predictions["scores"], dtype=np.float64)
    baseline_maps = np.asarray(test_predictions["maps"], dtype=np.float32)
    masks, fused_scores, fused_maps, symbolic_scores, symbolic_maps = [], [], [], [], []
    evidence_rows, csv_rows, visuals = [], [], []
    explanations = output / "query_explanations"
    explanations.mkdir(parents=True, exist_ok=True)
    for index, (sample, baseline_score, baseline_map, layer_maps) in enumerate(zip(
        samples, baseline_scores, baseline_maps, test_predictions["layer_maps"]
    )):
        score, fused_map, symbolic, trace = fusion.infer(sample["path"], layer_maps)
        mask = _mask_for_sample(sample, class_root)
        masks.append(mask); fused_scores.append(score); fused_maps.append(fused_map)
        symbolic_scores.append(symbolic.symbolic_anomaly)
        symbolic_maps.append(symbolic.symbolic_map)
        triggered = [
            name for name, row in trace.get("rules", {}).items()
            if float(row["supported_violation"]) > 1e-8
        ]
        text = [
            f"Query: {Path(sample['path']).resolve()}",
            f"Baseline: {baseline_score:.6f}", f"Fused: {score:.6f}",
            f"Evolved symbolic anomaly: {symbolic.symbolic_anomaly:.6f}",
            f"AST: {json.dumps(library.get('logic', {}).get('anomaly_ast'), ensure_ascii=False)}",
        ]
        for name, atom in symbolic.atoms.items():
            fusion_row = trace.get("rules", {}).get(name, {})
            text.append(
                f"- {name} family={atom['family']} predicate={atom['predicate']} "
                f"fact={atom['fact_value']:.6f} upper={atom['upper']:.6f} "
                f"violation={atom['violation']:.6f} "
                f"feature_gate={fusion_row.get('aggregate_gate', 0.0):.6f}"
            )
        explanation_path = explanations / f"{index:03d}_{Path(sample['path']).parent.name}_{Path(sample['path']).stem}.txt"
        explanation_path.write_text("\n".join(text) + "\n", encoding="utf-8")
        evidence_rows.append({
            "path": str(Path(sample["path"]).resolve()), "facts": symbolic.facts,
            "atoms": symbolic.atoms, "diagnostics": symbolic.diagnostics,
            "symbolic_trace": symbolic.trace, "feature_fusion_trace": trace,
            "explanation": str(explanation_path),
        })
        csv_rows.append({
            "path": str(Path(sample["path"]).resolve()),
            "defect_type": sample["defect_type"], "label": int(sample["label"]),
            "baseline_score": float(baseline_score), "symbolic_score": symbolic.symbolic_anomaly,
            "fused_score": float(score), "triggered_rules": "|".join(triggered) or "none",
        })
        if args.max_visualizations <= 0 or index < args.max_visualizations:
            visuals.append(_visualize(
                output / "visualizations", sample, mask, baseline_map,
                symbolic.symbolic_map, fused_map, index,
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
        "schema_version": "11.0-strict-fsad-orthogonal-feature-fusion",
        "dataset": args.dataset, "class_name": args.class_name, "shot": args.shot,
        "strict_few_shot": True, "query_or_test_used_for_rule_search": False,
        "category_specific_python_branching": False,
        "rules": [{"id": row["id"], "predicate": row["predicate"]} for row in library.get("rules", [])],
        "evolved_anomaly_ast": library.get("logic", {}).get("anomaly_ast"),
        "baseline": baseline_metrics, "symbolic_only": symbolic_metrics,
        "rule_conditioned_feature_fusion": fused_metrics,
        "delta_fused_minus_baseline": _delta(fused_metrics, baseline_metrics),
        "normal_rule_feature_trigger_rate": float(np.mean([
            row["triggered_rules"] != "none" for row, label in zip(csv_rows, labels) if label == 0
        ])),
        "anomaly_rule_feature_trigger_rate": float(np.mean([
            row["triggered_rules"] != "none" for row, label in zip(csv_rows, labels) if label == 1
        ])),
        "rule_library": str(output / "normality_rules_v11_orthogonal.json"),
    }
    with (output / "predictions_v11.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    _dump(output / "rule_evidence_v11.json", evidence_rows)
    _dump(output / "visualization_manifest_v11.json", visuals)
    np.savez_compressed(
        output / "pixel_predictions_v11.npz", labels=labels, masks=masks,
        baseline_maps=baseline_maps, symbolic_maps=symbolic_maps, fused_maps=fused_maps,
    )
    _dump(output / "metrics_v11.json", result)
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mvtec")
    p.add_argument("--data-root", required=True); p.add_argument("--class-name", required=True)
    p.add_argument("--test-subdir", default="test"); p.add_argument("--shot", type=int, default=4)
    p.add_argument("--support-seed", type=int, default=1); p.add_argument("--proposal-json", required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--prediction-cache-dir")
    p.add_argument("--device", default="cuda:0"); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--inpformer-root", default=DEFAULT_INPFORMER_ROOT)
    p.add_argument("--encoder", default="dinov2reg_vit_base_14"); p.add_argument("--inp-num", type=int, default=6)
    p.add_argument("--input-size", type=int, default=448); p.add_argument("--crop-size", type=int, default=392)
    p.add_argument("--map-size", type=int, default=256); p.add_argument("--max-rules", type=int, default=4)
    p.add_argument("--maximum-response-correlation", type=float, default=0.85)
    p.add_argument("--search-seed", type=int, default=668); p.add_argument("--max-visualizations", type=int, default=20)
    p.add_argument("--llm-model", default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking")
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2, default=_default))
