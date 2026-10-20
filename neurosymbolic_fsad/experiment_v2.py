"""Reliable, support-validated neuro-symbolic FSAD experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .augmentations import build_calibration_views
from .data import collect_test_samples, select_inpformer_support, select_support, stratified_limit
from .features import prepare_inp_aligned_rgb
from .inpformer_adapter import (
    DEFAULT_INPFORMER_ROOT,
    INPFormerAdapter,
    load_prediction_cache,
    save_prediction_cache,
)
from .qwen_rules import DEFAULT_MODEL_PATH, QwenRuleProposer, make_support_montage
from .reliable_rules import (
    ReliableRuleEngine,
    build_reliable_library,
    build_shape_context,
    extract_reliable_features,
    mine_reliable_rules,
    train_reliable_fusion,
)
from .rule_engine import default_rule_proposals


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(np.unique(scores)) < 2:
        auroc = 0.5
    else:
        auroc = float(roc_auc_score(labels, scores))
    return {
        "i_auroc": auroc,
        "i_aupr": float(average_precision_score(labels, scores)),
    }


def _cached_infer(
    adapter: INPFormerAdapter,
    cache_path: Path,
    images: Sequence,
    identifiers: Sequence[str],
    metadata: Mapping,
    batch_size: int,
):
    cached = load_prediction_cache(cache_path, identifiers, expected_metadata=dict(metadata))
    if cached is not None:
        return cached
    predictions = adapter.infer_images(images, batch_size=batch_size, identifiers=identifiers)
    save_prediction_cache(cache_path, predictions, metadata=dict(metadata))
    return predictions


def _write_predictions(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "path", "defect_type", "label", "baseline_score", "baseline_z",
        "rule_score", "decision_score", "top_rule", "top_rule_violation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch = __import__("torch")
    requested_cuda = [args.device, args.head_device]
    if any(item.startswith("cuda") for item in requested_cuda) and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the v2 experiment but is unavailable")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    class_root = Path(args.data_root).expanduser().resolve() / args.class_name
    if args.support_strategy == "inp_official":
        support_paths = select_inpformer_support(
            args.data_root, args.dataset, args.class_name, args.shot, args.support_seed
        )
    else:
        support_paths = select_support(class_root, args.shot, args.support_seed)
    test_samples = stratified_limit(
        collect_test_samples(class_root), args.max_normal, args.max_anomaly, args.seed
    )
    test_paths = [Path(item["path"]) for item in test_samples]
    views = build_calibration_views(support_paths, seed=args.calibration_seed)
    view_images = [item.image for item in views]
    view_ids = [
        f"{'pseudo' if item.is_pseudo_anomaly else 'normal'}://{item.source_index}/{item.transform}"
        for item in views
    ]

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
    cache_metadata = adapter.cache_metadata()
    cache_root = Path(args.prediction_cache_dir).expanduser().resolve() \
        if args.prediction_cache_dir else output
    support_predictions = _cached_infer(
        adapter, cache_root / "cache" / "support_inp.npz", support_paths,
        [str(path.resolve()) for path in support_paths], cache_metadata, args.batch_size,
    )
    test_predictions = _cached_infer(
        adapter, cache_root / "cache" / "test_inp.npz", test_paths,
        [str(path.resolve()) for path in test_paths], cache_metadata, args.batch_size,
    )
    calibration_predictions = _cached_infer(
        adapter, cache_root / "cache" / "calibration_inp.npz", view_images,
        view_ids, cache_metadata, args.batch_size,
    )
    adapter.close()
    del adapter
    gc.collect()

    normal_indices = [index for index, item in enumerate(views) if not item.is_pseudo_anomaly]
    pseudo_indices = [index for index, item in enumerate(views) if item.is_pseudo_anomaly]
    normal_maps = calibration_predictions["maps"][normal_indices]
    shape_context = build_shape_context(normal_maps, grid_size=args.shape_grid)

    calibration_rows = [
        extract_reliable_features(
            item.image, score_map, shape_context, args.input_size, args.crop_size
        )
        for item, score_map in zip(views, calibration_predictions["maps"])
    ]
    normal_rows = [calibration_rows[index] for index in normal_indices]
    pseudo_rows = [calibration_rows[index] for index in pseudo_indices]
    normal_scores = calibration_predictions["scores"][normal_indices]
    pseudo_scores = calibration_predictions["scores"][pseudo_indices]
    normal_groups = [views[index].source_index for index in normal_indices]
    pseudo_groups = [views[index].source_index for index in pseudo_indices]

    support_rows = [
        extract_reliable_features(
            path, score_map, shape_context, args.input_size, args.crop_size
        )
        for path, score_map in zip(support_paths, support_predictions["maps"])
    ]
    montage = make_support_montage(
        support_paths, support_predictions["maps"], output / "normal_support_montage.jpg"
    )
    if args.proposal_json:
        proposal = json.loads(Path(args.proposal_json).read_text(encoding="utf-8"))
        proposal_source = "provided_json"
    elif args.no_llm:
        proposal = {
            "normality_summary": "fixed candidate ablation; empirical verifier remains enabled",
            "rules": default_rule_proposals(8),
        }
        proposal_source = "no_llm_ablation"
    else:
        proposer = QwenRuleProposer(
            model_path=args.llm_model,
            device=args.llm_device,
            max_new_tokens=args.max_new_tokens,
        )
        proposal = proposer.propose(args.class_name, support_rows, montage)
        (output / "qwen_raw_output.txt").write_text(
            proposal.get("raw_output", "") + "\n", encoding="utf-8"
        )
        proposal_source = "qwen3_vl"
        del proposer
        gc.collect()
        if __import__("torch").cuda.is_available():
            __import__("torch").cuda.empty_cache()
    _dump(output / "rule_candidates.json", {
        key: value for key, value in proposal.items() if key != "raw_output"
    })

    rules, candidate_audit = mine_reliable_rules(
        normal_rows, pseudo_rows, normal_scores, pseudo_scores,
        normal_groups, pseudo_groups, qwen_proposal=proposal,
        max_rules=args.max_rules, minimum_cv_auc=args.minimum_rule_auc,
        maximum_normal_fpr=args.maximum_rule_fpr,
    )
    fusion = train_reliable_fusion(
        rules, normal_rows, pseudo_rows, normal_scores, pseudo_scores,
        normal_groups, pseudo_groups, device=args.head_device,
        epochs=args.head_epochs, seed=args.calibration_seed,
        minimum_cv_gain=args.minimum_fusion_gain,
        require_cuda=not args.allow_cpu_head,
    )
    library = build_reliable_library(
        dataset=args.dataset,
        class_name=args.class_name,
        support_paths=[str(path) for path in support_paths],
        shape_context=shape_context,
        rules=rules,
        audit=candidate_audit,
        fusion=fusion,
        qwen_proposal=proposal,
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=str(cache_metadata["checkpoint_sha256"]),
        input_size=args.input_size,
        crop_size=args.crop_size,
    )
    library["qwen"]["source"] = proposal_source
    engine = ReliableRuleEngine(library)
    rules_path = engine.save(output / "normality_rules_v2.json")

    labels = np.asarray([int(item["label"]) for item in test_samples])
    baseline_scores = np.asarray(test_predictions["scores"], dtype=float)
    rule_scores, decisions, rows, evidence_rows = [], [], [], []
    for sample, score_map, baseline in zip(
        test_samples, test_predictions["maps"], baseline_scores
    ):
        facts = engine.facts(sample["path"], score_map)
        decision, rule_score, evidence = engine.decision_score(float(baseline), facts)
        top = evidence[0] if evidence else None
        rule_scores.append(rule_score)
        decisions.append(decision)
        rows.append({
            "path": str(sample["path"]), "defect_type": sample["defect_type"],
            "label": int(sample["label"]), "baseline_score": float(baseline),
            "baseline_z": engine.baseline_z(float(baseline)), "rule_score": rule_score,
            "decision_score": decision,
            "top_rule": top["rule_id"] if top else "abstain",
            "top_rule_violation": top["violation"] if top else 0.0,
        })
        evidence_rows.append({
            "path": str(sample["path"]), "facts": facts, "evidence": evidence,
        })
    rule_scores, decisions = np.asarray(rule_scores), np.asarray(decisions)
    result = {
        "schema_version": "2.0",
        "dataset": args.dataset,
        "class_name": args.class_name,
        "shot": args.shot,
        "test_count": len(labels),
        "normal_count": int((labels == 0).sum()),
        "anomaly_count": int((labels == 1).sum()),
        "rules": str(rules_path),
        "selected_rule_count": len(rules),
        "selected_features": [rule["feature"] for rule in rules],
        "fusion_enabled": bool(fusion["enabled"]),
        "fusion_cv_mean_gain": float(fusion["cv_mean_gain"]),
        "baseline": _metrics(labels, baseline_scores),
        "rules_only": _metrics(labels, rule_scores),
        "baseline_plus_rules": _metrics(labels, decisions),
    }
    result["delta_i_auroc"] = (
        result["baseline_plus_rules"]["i_auroc"] - result["baseline"]["i_auroc"]
    )
    result["delta_i_aupr"] = (
        result["baseline_plus_rules"]["i_aupr"] - result["baseline"]["i_aupr"]
    )
    _write_predictions(output / "predictions_v2.csv", rows)
    _dump(output / "rule_evidence_v2.json", evidence_rows)
    _dump(output / "candidate_audit.json", candidate_audit)
    _dump(output / "calibration_manifest.json", [
        {
            "identifier": identifier,
            "source_index": item.source_index,
            "transform": item.transform,
            "label": int(item.is_pseudo_anomaly),
        }
        for identifier, item in zip(view_ids, views)
    ])
    _dump(output / "split.json", {
        "support": [str(path) for path in support_paths],
        "test": [{**item, "path": str(item["path"])} for item in test_samples],
        "test_labels_used_for_training_or_selection": False,
    })
    _dump(output / "metrics_v2.json", result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Support-validated Qwen-guided static rules for INP-Former"
    )
    parser.add_argument("--dataset", default="mvtec")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--support-seed", type=int, default=1)
    parser.add_argument("--support-strategy", choices=["inp_official", "hash"], default="inp_official")
    parser.add_argument("--seed", type=int, default=668)
    parser.add_argument("--calibration-seed", type=int, default=20261013)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inpformer-root", default=DEFAULT_INPFORMER_ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-cache-dir",
                        help="Read matching fingerprinted INP caches from another run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--head-device", default="cuda:0")
    parser.add_argument("--llm-device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--map-size", type=int, default=256)
    parser.add_argument("--shape-grid", type=int, default=8)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--proposal-json")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--max-rules", type=int, default=8)
    parser.add_argument("--minimum-rule-auc", type=float, default=0.58)
    parser.add_argument("--maximum-rule-fpr", type=float, default=0.25)
    parser.add_argument("--minimum-fusion-gain", type=float, default=0.005)
    parser.add_argument("--head-epochs", type=int, default=350)
    parser.add_argument("--allow-cpu-head", action="store_true",
                        help="Allow fallback only for debugging; formal runs require CUDA")
    parser.add_argument("--max-normal", type=int, default=0)
    parser.add_argument("--max-anomaly", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
