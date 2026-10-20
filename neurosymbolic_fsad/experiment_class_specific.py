"""Frozen category-specific static-rule experiment on cached INP-Former scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .class_specific_rules import ClassSpecificRuleEngine, build_class_specific_library
from .data import collect_test_samples


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    return {
        "i_auroc": float(roc_auc_score(labels, scores)) if np.ptp(scores) else 0.5,
        "i_aupr": float(average_precision_score(labels, scores)),
    }


def _read_cache(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "paths": payload["paths"].astype(str),
            "scores": payload["scores"].astype(np.float64),
            "metadata": json.loads(str(payload["metadata_json"].item())),
        }


def _assert_paths(actual: Sequence[str], expected: Sequence[Path], name: str) -> None:
    normalized = np.asarray([str(Path(path).resolve()) for path in expected])
    if not np.array_equal(np.asarray(actual), normalized):
        raise ValueError(f"{name} cache path/order mismatch")


def _support_only_fusion(support_scores: np.ndarray) -> Dict[str, Any]:
    """One frozen calibration shared by all category programs.

    A full logical violation (score 1) receives half of the normal INP median.
    Since each category has 3--4 equally weighted rules, one isolated rule adds
    only 12.5--16.7% of that median.  This is intentionally fixed before test
    evaluation and never searches a test-set coefficient.
    """
    center = float(np.median(support_scores))
    q10, q90 = (float(value) for value in np.quantile(support_scores, [0.10, 0.90]))
    alpha = max(0.5 * abs(center), 4.0 * max(q90 - q10, 0.0), 1e-4)
    return {
        "enabled": True,
        "alpha": alpha,
        "baseline_center": center,
        "normal_q10": q10,
        "normal_q90": q90,
        "calibration_rule": "alpha=max(0.5*abs(normal_median),4*(normal_q90-normal_q10),1e-4)",
        "calibration_uses": "four normal support INP scores only",
        "test_labels_used": False,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.source_run_dir).expanduser().resolve()
    split = json.loads((source / "split_v4_generic.json").read_text(encoding="utf-8"))
    support_paths = [Path(path).resolve() for path in split["support"]]
    class_root = Path(args.data_root).expanduser().resolve() / args.class_name
    samples = collect_test_samples(class_root)
    test_paths = [Path(item["path"]).resolve() for item in samples]

    cache_root = Path(args.prediction_cache_dir).expanduser().resolve() / "cache"
    support_cache = _read_cache(cache_root / "support_inp_v4.npz")
    test_cache = _read_cache(cache_root / "test_inp_v4.npz")
    _assert_paths(support_cache["paths"], support_paths, "support")
    _assert_paths(test_cache["paths"], test_paths, "test")
    for key, value in {"dataset": args.dataset, "class_name": args.class_name, "shot": 4}.items():
        if support_cache["metadata"].get(key) != value or test_cache["metadata"].get(key) != value:
            raise ValueError(f"cache metadata mismatch for {key}")

    qwen_path = source / "qwen_generic_relational_proposal.json"
    qwen: Mapping[str, Any] = (
        json.loads(qwen_path.read_text(encoding="utf-8")) if qwen_path.is_file() else {}
    )
    library = build_class_specific_library(
        args.dataset, args.class_name, support_paths, qwen_proposal=qwen
    )
    qwen_predicates = {
        str(row.get("predicate")) for row in qwen.get("rules", [])
        if isinstance(row, Mapping)
    }
    qwen_consensus_available = isinstance(qwen.get("primary_entity"), Mapping)
    for rule in library["rules"]:
        rule["llm_proposal_supported"] = bool(
            qwen_consensus_available and rule["predicate"] in qwen_predicates
        )
    library["shot"] = len(support_paths)
    library["experiment_status"] = "exploratory_category_specific_development"
    library["qwen_model"] = args.llm_model
    library["compiler"] = {
        "type": "category_specific_executable_scene_program",
        "qwen_consensus_available": qwen_consensus_available,
        "selection": (
            "LLM-consensus entity/relations grounded and calibrated on normal supports"
            if qwen_consensus_available else
            "category-specific exploratory program after the generic Qwen proposal abstained"
        ),
        "test_time": "static JSON plus deterministic OpenCV grounder; no LLM",
    }
    fusion = _support_only_fusion(np.asarray(support_cache["scores"], dtype=float))
    library["fusion"] = fusion
    library["inpformer"] = support_cache["metadata"]
    library_path = output / "normality_rules_v5_class_specific.json"
    _dump(library_path, library)

    engine = ClassSpecificRuleEngine(library)
    support_evidence = []
    for path, baseline_score in zip(support_paths, support_cache["scores"]):
        relation, evidence, facts, diagnostics = engine.evaluate(path)
        support_evidence.append({
            "path": str(path), "baseline_score": float(baseline_score),
            "relation_score": relation, "facts": facts,
            "diagnostics": diagnostics, "evidence": evidence,
        })

    labels = np.asarray([int(item["label"]) for item in samples])
    baseline = np.asarray(test_cache["scores"], dtype=float)
    relation_scores, decisions, evidence_rows, csv_rows = [], [], [], []
    for sample, baseline_score in zip(samples, baseline):
        relation, evidence, facts, diagnostics = engine.evaluate(sample["path"])
        decision = float(baseline_score + fusion["alpha"] * relation)
        relation_scores.append(relation)
        decisions.append(decision)
        top = evidence[0] if evidence else None
        csv_rows.append({
            "path": str(Path(sample["path"]).resolve()),
            "defect_type": str(sample["defect_type"]),
            "label": int(sample["label"]),
            "baseline_score": float(baseline_score),
            "relation_score": float(relation),
            "decision_score": decision,
            "top_rule": top["rule_id"] if top else "none",
            "top_violation": float(top["violation"]) if top else 0.0,
        })
        evidence_rows.append({
            "path": str(Path(sample["path"]).resolve()),
            "facts": facts, "diagnostics": diagnostics, "evidence": evidence,
        })

    relation_scores = np.asarray(relation_scores, dtype=float)
    decisions = np.asarray(decisions, dtype=float)
    defect_types = np.asarray([str(item["defect_type"]) for item in samples])
    per_defect = {}
    for defect in sorted(set(defect_types) - {"good", "ok"}):
        selected = np.isin(defect_types, ["good", "ok", defect])
        anomaly = defect_types == defect
        per_defect[defect] = {
            "count": int(anomaly.sum()),
            "baseline_i_auroc": float(roc_auc_score(labels[selected], baseline[selected])),
            "fused_i_auroc": float(roc_auc_score(labels[selected], decisions[selected])),
            "relation_trigger_rate": float(np.mean(relation_scores[anomaly] > 0)),
            "mean_relation_score": float(np.mean(relation_scores[anomaly])),
        }
    baseline_metrics = _metrics(labels, baseline)
    fused_metrics = _metrics(labels, decisions)
    result = {
        "schema_version": "5.0-class-specific-static",
        "dataset": args.dataset,
        "class_name": args.class_name,
        "experiment_status": "exploratory_category_specific_development",
        "shot": len(support_paths),
        "test_count": len(samples),
        "normal_count": int(np.sum(labels == 0)),
        "anomaly_count": int(np.sum(labels == 1)),
        "selected_rule_count": len(library["rules"]),
        "selected_predicates": [row["predicate"] for row in library["rules"]],
        "fusion": fusion,
        "baseline": baseline_metrics,
        "baseline_plus_rules": fused_metrics,
        "delta_i_auroc": fused_metrics["i_auroc"] - baseline_metrics["i_auroc"],
        "delta_i_aupr": fused_metrics["i_aupr"] - baseline_metrics["i_aupr"],
        "normal_trigger_rate": float(np.mean(relation_scores[labels == 0] > 0)),
        "anomaly_trigger_rate": float(np.mean(relation_scores[labels == 1] > 0)),
        "mean_normal_relation_score": float(np.mean(relation_scores[labels == 0])),
        "mean_anomaly_relation_score": float(np.mean(relation_scores[labels == 1])),
        "per_defect_type": per_defect,
        "rule_library": str(library_path),
        "test_labels_used_for_training_or_selection": False,
        "runtime_requires_llm": False,
    }
    with (output / "predictions_v5_class_specific.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    _dump(output / "support_evidence_v5_class_specific.json", support_evidence)
    _dump(output / "rule_evidence_v5_class_specific.json", evidence_rows)
    _dump(output / "metrics_v5_class_specific.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mvtec", "visa", "mpdd"), required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--prediction-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-model", default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
