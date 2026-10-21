#!/usr/bin/env python3
"""V13: shared INP entity tokens and bidirectional symbolic correction."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np

from neurosymbolic_fsad.data import collect_test_samples, select_inpformer_support
from neurosymbolic_fsad.entity_bridge import EntityTokenBridge
from neurosymbolic_fsad.experiment_logic import _mask_for_sample
from neurosymbolic_fsad.inpformer_adapter import (
    DEFAULT_INPFORMER_ROOT, INPFormerAdapter, load_prediction_cache,
    save_prediction_cache, file_sha256,
)
from neurosymbolic_fsad.ovd_grounder import (
    DEFAULT_BERT, DEFAULT_GROUNDING_DINO_CONFIG, DEFAULT_GROUNDING_DINO_ROOT,
    DEFAULT_GROUNDING_DINO_WEIGHTS, GroundingDINOAdapter,
    OpenVocabularySymbolicEngine, compile_ovd_program,
)
from neurosymbolic_fsad.qwen_generic_relations import validate_generic_relational_proposal
from run_ovd_symbolic_fsad import _default, _delta, _dump, _metrics, _visualize


def run(args):
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).expanduser().resolve()
    class_root = data_root / args.class_name
    supports = select_inpformer_support(
        data_root, args.dataset, args.class_name, args.shot, args.support_seed
    )
    raw = json.loads(Path(args.proposal_json).read_text(encoding="utf-8"))
    proposal = validate_generic_relational_proposal(raw)
    proposal["model"] = raw.get("model", args.llm_model)

    detector = None
    if proposal.get("primary_entity") is not None:
        detector = GroundingDINOAdapter(
            device=args.ovd_device, root=args.ovd_root, config=args.ovd_config,
            weights=args.ovd_weights, bert=args.bert,
        )
    library, compiler_audit = compile_ovd_program(
        detector, proposal, supports, image_size=args.map_size,
        search_seed=args.search_seed,
    )
    library.update({
        "dataset": args.dataset, "class_name": args.class_name,
        "shot": args.shot, "support": [str(Path(path).resolve()) for path in supports],
        "llm": {
            "model": proposal.get("model", args.llm_model),
            "proposal": proposal,
            "role": "offline semantic entity/rule proposal only",
        },
    })
    _dump(output / "normality_rules_v13.json", library)
    _dump(output / "compiler_audit_v13.json", compiler_audit)

    samples = collect_test_samples(class_root, test_subdir=args.test_subdir)
    if args.max_queries > 0:
        samples = samples[:args.max_queries]
    test_paths = [Path(row["path"]) for row in samples]
    metadata = {
        "schema": "inp-entity-feature-grid-v13", "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "dataset": args.dataset, "class_name": args.class_name, "shot": args.shot,
        "support_seed": args.support_seed, "encoder": args.encoder,
        "inp_num": args.inp_num, "input_size": args.input_size,
        "crop_size": args.crop_size, "map_size": args.map_size,
        "projection_dim": args.feature_projection_dim,
        "support_sha256": [file_sha256(path) for path in supports],
        "support_selector_schema": "inp-support-v2",
    }
    cache_root = Path(args.prediction_cache_dir or output).expanduser().resolve()
    support_cache = cache_root / "cache" / "support_inp_entity_v13.npz"
    test_cache = cache_root / "cache" / "test_inp_entity_v13.npz"
    support_ids = [str(Path(path).resolve()) for path in supports]
    test_ids = [str(path.resolve()) for path in test_paths]
    support_predictions = load_prediction_cache(
        support_cache, support_ids, expected_metadata=metadata
    )
    test_predictions = load_prediction_cache(
        test_cache, test_ids, expected_metadata=metadata
    )
    needs_entity_features = isinstance(library.get("grounder"), dict)
    required = lambda row: row is not None and all(
        key in row for key in (
            ("maps", "scores", "feature_grids")
            if needs_entity_features else ("maps", "scores")
        )
    )
    if not required(support_predictions) or not required(test_predictions):
        adapter = INPFormerAdapter(
            args.checkpoint, args.inp_device, args.inpformer_root,
            args.encoder, args.inp_num, args.input_size, args.crop_size, args.map_size,
        )
        if not required(support_predictions):
            support_predictions = adapter.infer_images(
                supports, args.batch_size, support_ids,
                return_feature_grids=needs_entity_features,
                feature_projection_dim=args.feature_projection_dim,
            )
            save_prediction_cache(support_cache, support_predictions, metadata)
        if not required(test_predictions):
            test_predictions = adapter.infer_images(
                test_paths, args.batch_size, test_ids,
                return_feature_grids=needs_entity_features,
                feature_projection_dim=args.feature_projection_dim,
            )
            save_prediction_cache(test_cache, test_predictions, metadata)
        adapter.close(); del adapter; gc.collect()

    engine = OpenVocabularySymbolicEngine(library, detector)
    bridge, bridge_error = None, None
    if engine.grounder is not None and engine.program is not None:
        try:
            bridge = EntityTokenBridge.fit(
                engine, supports, support_predictions["feature_grids"],
                support_predictions["maps"], support_predictions["scores"],
            )
            _dump(output / "entity_bridge_v13.json", bridge.state_dict())
        except Exception as error:
            bridge_error = f"{type(error).__name__}: {error}"
    if bridge is None:
        _dump(output / "entity_bridge_v13.json", {
            "abstained": True,
            "reason": bridge_error or library.get("abstention_reason", "no_executable_program"),
        })

    labels = np.asarray([row["label"] for row in samples], dtype=np.uint8)
    baseline_scores = np.asarray(test_predictions["scores"], dtype=float)
    baseline_maps = np.asarray(test_predictions["maps"], dtype=np.float32)
    masks, fused_scores, fused_maps, rows, evidence, visuals = [], [], [], [], [], []
    explanations = output / "query_explanations"
    explanations.mkdir(parents=True, exist_ok=True)
    query_features = (
        test_predictions["feature_grids"]
        if "feature_grids" in test_predictions else [None] * len(samples)
    )
    for index, (sample, baseline_score, baseline_map, feature_grids) in enumerate(zip(
        samples, baseline_scores, baseline_maps, query_features
    )):
        if bridge is None:
            score, fused_map = float(baseline_score), baseline_map.copy()
            symbolic = engine.evaluate(sample["path"])
            diagnostics = {"abstained": True, "reason": bridge_error}
        else:
            result = bridge.infer(
                sample["path"], feature_grids, baseline_map, baseline_score
            )
            score, fused_map = result.score, result.anomaly_map
            symbolic, diagnostics = result.raw_symbolic, result.diagnostics
        mask = _mask_for_sample(sample, class_root)
        masks.append(mask); fused_scores.append(score); fused_maps.append(fused_map)
        correction = np.abs(fused_map - baseline_map)
        triggered = bool(diagnostics.get("corrected_symbolic_anomaly", 0.0) > 1e-8)
        explanation = explanations / f"{index:03d}_{Path(sample['path']).parent.name}_{Path(sample['path']).stem}.txt"
        explanation.write_text("\n".join([
            f"Query: {Path(sample['path']).resolve()}",
            f"Baseline: {float(baseline_score):.6f}",
            f"Fused: {float(score):.6f}",
            f"Detected entities: {symbolic.diagnostics.get('detected_count', 0)}",
            f"Token distances: {diagnostics.get('slot_token_distances', [])}",
            f"Token violations: {diagnostics.get('slot_token_violations', [])}",
            f"Corrected rules: {json.dumps(diagnostics.get('corrected_rules', {}), ensure_ascii=False)}",
        ]) + "\n", encoding="utf-8")
        rows.append({
            "path": str(Path(sample["path"]).resolve()),
            "defect_type": sample["defect_type"], "label": int(sample["label"]),
            "baseline_score": float(baseline_score), "fused_score": float(score),
            "bridge_triggered": int(triggered),
            "rescue_max": float(diagnostics.get("rescue_max", 0.0)),
            "suppression_max": float(diagnostics.get("suppression_max", 0.0)),
        })
        evidence.append({
            "path": rows[-1]["path"], "raw_symbolic_facts": symbolic.facts,
            "raw_symbolic_atoms": symbolic.atoms, "bridge": diagnostics,
            "explanation": str(explanation),
        })
        if args.max_visualizations <= 0 or index < args.max_visualizations or triggered:
            visuals.append(_visualize(
                output / "visualizations", sample, mask, baseline_map,
                correction, fused_map, symbolic.diagnostics, index,
            ))
    masks = np.asarray(masks, dtype=bool)
    fused_scores = np.asarray(fused_scores, dtype=float)
    fused_maps = np.asarray(fused_maps, dtype=np.float32)
    baseline_metrics = _metrics(labels, baseline_scores, masks, baseline_maps)
    fused_metrics = _metrics(labels, fused_scores, masks, fused_maps)
    result = {
        "schema_version": "13.0-shared-inp-entity-token-bidirectional",
        "dataset": args.dataset, "class_name": args.class_name, "shot": args.shot,
        "strict_few_shot": True, "query_labels_used_for_calibration": False,
        "bridge_abstained": bridge is None, "bridge_error": bridge_error,
        "grounding": compiler_audit.get("grounding"),
        "baseline": baseline_metrics, "entity_bridge_fusion": fused_metrics,
        "delta_fused_minus_baseline": _delta(fused_metrics, baseline_metrics),
        "trigger_rate_normal": float(np.mean([
            row["bridge_triggered"] for row in rows if not row["label"]
        ])),
        "trigger_rate_anomaly": float(np.mean([
            row["bridge_triggered"] for row in rows if row["label"]
        ])),
    }
    with (output / "predictions_v13.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    _dump(output / "rule_evidence_v13.json", evidence)
    _dump(output / "visualization_manifest_v13.json", visuals)
    np.savez_compressed(
        output / "pixel_predictions_v13.npz", labels=labels, masks=masks,
        baseline_maps=baseline_maps, fused_maps=fused_maps,
    )
    _dump(output / "metrics_v13.json", result)
    if detector is not None:
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
    p.add_argument("--batch-size", type=int, default=12); p.add_argument("--inpformer-root", default=DEFAULT_INPFORMER_ROOT)
    p.add_argument("--encoder", default="dinov2reg_vit_base_14"); p.add_argument("--inp-num", type=int, default=6)
    p.add_argument("--input-size", type=int, default=448); p.add_argument("--crop-size", type=int, default=392)
    p.add_argument("--map-size", type=int, default=256); p.add_argument("--feature-projection-dim", type=int, default=16)
    p.add_argument("--search-seed", type=int, default=668); p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--max-visualizations", type=int, default=20)
    p.add_argument("--ovd-root", default=DEFAULT_GROUNDING_DINO_ROOT)
    p.add_argument("--ovd-config", default=DEFAULT_GROUNDING_DINO_CONFIG)
    p.add_argument("--ovd-weights", default=DEFAULT_GROUNDING_DINO_WEIGHTS)
    p.add_argument("--bert", default=DEFAULT_BERT)
    p.add_argument("--llm-model", default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking")
    return p


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2, default=_default))
