#!/usr/bin/env python3
"""Independent metric recomputation, rule ablation and paired bootstrap for v5."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def metric(labels, scores):
    return {
        "i_auroc": float(roc_auc_score(labels, scores)),
        "i_aupr": float(average_precision_score(labels, scores)),
    }


def bootstrap(labels, baseline, fused, repeats, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    normal = np.flatnonzero(labels == 0)
    anomaly = np.flatnonzero(labels == 1)
    deltas = []
    for _ in range(repeats):
        selected = np.concatenate([
            rng.choice(normal, len(normal), replace=True),
            rng.choice(anomaly, len(anomaly), replace=True),
        ])
        deltas.append(
            roc_auc_score(labels[selected], fused[selected])
            - roc_auc_score(labels[selected], baseline[selected])
        )
    values = np.asarray(deltas)
    return {
        "mean_delta_i_auroc": float(np.mean(values)),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "probability_positive": float(np.mean(values > 0)),
    }


def validate(run_dir: Path, repeats: int, seed: int):
    metrics_path = run_dir / "metrics_v5_class_specific.json"
    predictions_path = run_dir / "predictions_v5_class_specific.csv"
    evidence_path = run_dir / "rule_evidence_v5_class_specific.json"
    reported = json.loads(metrics_path.read_text(encoding="utf-8"))
    library = json.loads(Path(reported["rule_library"]).read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows])
    baseline = np.asarray([float(row["baseline_score"]) for row in rows])
    relation = np.asarray([float(row["relation_score"]) for row in rows])
    decision = np.asarray([float(row["decision_score"]) for row in rows])
    alpha = float(library["fusion"]["alpha"])
    recomputed_baseline = metric(labels, baseline)
    recomputed_fused = metric(labels, decision)
    checks = {
        "row_count": len(rows) == reported["test_count"] == len(evidence),
        "decision_formula": bool(np.allclose(decision, baseline + alpha * relation, atol=1e-12)),
        "baseline_metric": bool(all(
            np.isclose(recomputed_baseline[key], reported["baseline"][key], atol=1e-12)
            for key in recomputed_baseline
        )),
        "fused_metric": bool(all(
            np.isclose(recomputed_fused[key], reported["baseline_plus_rules"][key], atol=1e-12)
            for key in recomputed_fused
        )),
        "path_alignment": all(row["path"] == item["path"] for row, item in zip(rows, evidence)),
        "runtime_llm_free": library.get("runtime_requires_llm") is False,
        "no_test_label_selection": (
            library.get("test_labels_used_for_training_or_selection") is False
            and reported.get("test_labels_used_for_training_or_selection") is False
            and library["fusion"].get("test_labels_used") is False
        ),
        "four_normal_supports": len(library.get("support", [])) == 4,
        "crossfit_complete": len(library.get("crossfit", [])) == 4,
        "qwen_model_recorded": "Qwen3-VL-8B-Thinking" in reported["rule_library"] or (
            "Qwen3-VL-8B-Thinking" in str(library.get("qwen_model", ""))
        ),
    }
    contributions = {}
    for rule in library["rules"]:
        rule_id = rule["id"]
        component = np.asarray([
            next(item["weight"] * item["violation"] for item in row["evidence"]
                 if item["rule_id"] == rule_id)
            for row in evidence
        ])
        ablated = baseline + alpha * (relation - component)
        score = metric(labels, ablated)
        contributions[rule_id] = {
            "ablated_i_auroc": score["i_auroc"],
            "full_minus_ablated_i_auroc": recomputed_fused["i_auroc"] - score["i_auroc"],
        }
    return {
        "dataset": reported["dataset"],
        "class_name": reported["class_name"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "recomputed": {"baseline": recomputed_baseline, "baseline_plus_rules": recomputed_fused},
        "delta_i_auroc": recomputed_fused["i_auroc"] - recomputed_baseline["i_auroc"],
        "paired_bootstrap": bootstrap(labels, baseline, decision, repeats, seed),
        "leave_one_rule_out": contributions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    reports = [
        validate(Path(path).resolve(), args.bootstrap, args.seed + index)
        for index, path in enumerate(args.run_dir)
    ]
    output = {
        "schema_version": "5.0-class-specific-independent-validation",
        "all_checks_passed": all(row["all_checks_passed"] for row in reports),
        "passed_checks": sum(sum(row["checks"].values()) for row in reports),
        "total_checks": sum(len(row["checks"]) for row in reports),
        "classes": reports,
        "macro_i_auroc": {
            "baseline": float(np.mean([row["recomputed"]["baseline"]["i_auroc"] for row in reports])),
            "fused": float(np.mean([row["recomputed"]["baseline_plus_rules"]["i_auroc"] for row in reports])),
        },
    }
    output["macro_i_auroc"]["delta"] = (
        output["macro_i_auroc"]["fused"] - output["macro_i_auroc"]["baseline"]
    )
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
