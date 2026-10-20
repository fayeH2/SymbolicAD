"""End-to-end baseline versus static-rule experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import (
    collect_test_samples, select_inpformer_support, select_support, stratified_limit,
)
from .features import extract_features
from .inpformer_adapter import (
    DEFAULT_INPFORMER_ROOT,
    INPFormerAdapter,
    load_prediction_cache,
    save_prediction_cache,
)
from .qwen_rules import DEFAULT_MODEL_PATH, QwenRuleProposer, make_support_montage
from .rule_engine import StaticRuleEngine, calibrate_rule_library, default_rule_proposals


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _predictions(
    cache_path: Path,
    paths: Sequence[Path],
    adapter: INPFormerAdapter,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    cached = load_prediction_cache(cache_path, paths)
    if cached is not None:
        return cached
    result = adapter.infer(paths, batch_size=batch_size)
    save_prediction_cache(cache_path, result)
    return result


def _metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    if len(np.unique(labels)) != 2:
        raise ValueError("Evaluation subset must contain normal and anomalous samples")
    return {
        "i_auroc": float(roc_auc_score(labels, scores)),
        "i_aupr": float(average_precision_score(labels, scores)),
    }


def _write_predictions(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path", "defect_type", "label", "baseline_score", "baseline_normalized",
        "rule_score", "fused_score", "top_rule", "top_rule_violation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    _json_dump(output_dir / "split.json", {
        "support": [str(path) for path in support_paths],
        "test": [{**item, "path": str(item["path"])} for item in test_samples],
        "support_only_rule_calibration": True,
    })

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
    support_predictions = _predictions(
        output_dir / "cache" / "support_inp.npz", support_paths, adapter, args.batch_size
    )
    test_predictions = _predictions(
        output_dir / "cache" / "test_inp.npz", test_paths, adapter, args.batch_size
    )
    adapter.close()
    del adapter
    gc.collect()

    support_rows = [
        extract_features(path, score_map)
        for path, score_map in zip(support_paths, support_predictions["maps"])
    ]
    test_rows = [
        extract_features(path, score_map)
        for path, score_map in zip(test_paths, test_predictions["maps"])
    ]
    _json_dump(output_dir / "support_facts.json", [
        {"path": str(path), "facts": facts}
        for path, facts in zip(support_paths, support_rows)
    ])

    montage = make_support_montage(
        support_paths, support_predictions["maps"], output_dir / "normal_support_montage.jpg"
    )
    if args.no_llm:
        proposal = {
            "normality_summary": "fixed feature set for no-LLM ablation",
            "rules": default_rule_proposals(args.max_rules),
            "raw_output": "",
        }
        llm_metadata = {"enabled": False, "ablation": "fixed_features"}
    else:
        proposer = QwenRuleProposer(
            model_path=args.llm_model,
            device=args.llm_device,
            max_new_tokens=args.max_new_tokens,
        )
        try:
            proposal = proposer.propose(args.class_name, support_rows, montage)
        except Exception:
            (output_dir / "qwen_raw_output.failed.txt").write_text(
                proposer.last_raw_output + "\n", encoding="utf-8"
            )
            raise
        llm_metadata = {
            "enabled": True,
            "model_path": str(Path(args.llm_model).resolve()),
            "raw_output_file": "qwen_raw_output.txt",
        }
        (output_dir / "qwen_raw_output.txt").write_text(
            proposal.get("raw_output", "") + "\n", encoding="utf-8"
        )
        del proposer
        gc.collect()

    _json_dump(output_dir / "rule_proposal.json", {
        key: value for key, value in proposal.items() if key != "raw_output"
    })
    library = calibrate_rule_library(
        proposal["rules"],
        support_rows,
        class_name=args.class_name,
        dataset=args.dataset,
        support_paths=[str(path) for path in support_paths],
        normality_summary=proposal["normality_summary"],
        llm_metadata=llm_metadata,
    )
    library["baseline"] = {
        "name": "INP-Former", "checkpoint": str(Path(args.checkpoint).resolve()),
        "encoder": args.encoder,
    }
    engine = StaticRuleEngine(library)
    rules_path = engine.save(output_dir / "normality_rules.json")

    labels = np.asarray([int(item["label"]) for item in test_samples])
    baseline_scores = np.asarray(test_predictions["scores"], dtype=float)
    rule_scores, fused_scores, prediction_rows, all_evidence = [], [], [], []
    for sample, facts, baseline in zip(test_samples, test_rows, baseline_scores):
        rule_score, evidence = engine.score(facts)
        fused = engine.fused_score(baseline, rule_score, args.alpha)
        top = evidence[0]
        rule_scores.append(rule_score)
        fused_scores.append(fused)
        prediction_rows.append({
            "path": str(sample["path"]),
            "defect_type": sample["defect_type"],
            "label": int(sample["label"]),
            "baseline_score": float(baseline),
            "baseline_normalized": engine.normalize_baseline(baseline),
            "rule_score": rule_score,
            "fused_score": fused,
            "top_rule": top["rule_id"],
            "top_rule_violation": top["violation"],
        })
        all_evidence.append({
            "path": str(sample["path"]), "facts": facts, "evidence": evidence,
        })
    rule_scores = np.asarray(rule_scores)
    fused_scores = np.asarray(fused_scores)
    baseline_metrics = _metrics(labels, baseline_scores)
    rule_metrics = _metrics(labels, rule_scores)
    fused_metrics = _metrics(labels, fused_scores)
    result = {
        "dataset": args.dataset,
        "class_name": args.class_name,
        "shot": args.shot,
        "test_count": len(test_samples),
        "normal_count": int((labels == 0).sum()),
        "anomaly_count": int((labels == 1).sum()),
        "alpha": args.alpha,
        "rules": str(rules_path),
        "baseline": baseline_metrics,
        "rules_only": rule_metrics,
        "baseline_plus_rules": fused_metrics,
        "delta_i_auroc": fused_metrics["i_auroc"] - baseline_metrics["i_auroc"],
        "delta_i_aupr": fused_metrics["i_aupr"] - baseline_metrics["i_aupr"],
        "note": "Pixel metrics are intentionally unchanged; v1 rules re-rank image anomaly scores.",
    }
    _write_predictions(output_dir / "predictions.csv", prediction_rows)
    _json_dump(output_dir / "rule_evidence.json", all_evidence)
    _json_dump(output_dir / "metrics.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="INP-Former + Qwen3-VL-guided static normality rules"
    )
    parser.add_argument("--dataset", default="mvtec")
    parser.add_argument("--data-root", required=True,
                        help="Root that directly contains class directories")
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--seed", type=int, default=668)
    parser.add_argument("--support-seed", type=int, default=1,
                        help="Seed used by the checkpoint's support sampler")
    parser.add_argument("--support-strategy", choices=["inp_official", "hash"],
                        default="inp_official",
                        help="Use exactly the support convention that trained INP-Former")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inpformer-root", default=DEFAULT_INPFORMER_ROOT,
                        help="External read-only INP-Former repository")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--map-size", type=int, default=256)
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--llm-device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Thinking model needs room to reach its final JSON")
    parser.add_argument("--max-rules", type=int, default=8)
    parser.add_argument("--no-llm", action="store_true",
                        help="Fixed-feature ablation; does not satisfy the LLM experiment")
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="Fixed support-only fusion weight")
    parser.add_argument("--max-normal", type=int, default=0,
                        help="0 uses all normal test images")
    parser.add_argument("--max-anomaly", type=int, default=0,
                        help="0 uses all anomalous test images")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
