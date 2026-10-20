#!/usr/bin/env python3
"""Independent recomputation and paired bootstrap for v4 suite outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _metric(labels, scores):
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def _bootstrap(labels, baseline, fused, repeats, seed):
    # Most classes deliberately fall back to the untouched baseline when no
    # rule survives support-only validation.  In that case every paired
    # bootstrap replicate has an exactly-zero delta, so avoid thousands of
    # redundant sklearn metric calls without changing the statistic.
    if np.array_equal(np.asarray(baseline), np.asarray(fused)):
        zero = {"mean": 0.0, "ci95": [0.0, 0.0], "probability_positive": 0.0}
        return {"delta_i_auroc": dict(zero), "delta_i_aupr": dict(zero)}
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    normal, anomaly = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    auroc, aupr = [], []
    for _ in range(repeats):
        indices = np.concatenate([
            rng.choice(normal, len(normal), replace=True),
            rng.choice(anomaly, len(anomaly), replace=True),
        ])
        b_auc, b_ap = _metric(labels[indices], baseline[indices])
        f_auc, f_ap = _metric(labels[indices], fused[indices])
        auroc.append(f_auc - b_auc)
        aupr.append(f_ap - b_ap)
    def describe(values):
        values = np.asarray(values)
        return {
            "mean": float(np.mean(values)),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "probability_positive": float(np.mean(values > 0)),
        }
    return {"delta_i_auroc": describe(auroc), "delta_i_aupr": describe(aupr)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-summary", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    reports = []
    all_checks = []
    for summary_name in args.suite_summary:
        summary_path = Path(summary_name).resolve()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        class_reports = []
        for index, item in enumerate(summary["classes"]):
            run_dir = Path(item["rule_library"]).resolve().parent
            library = json.loads(Path(item["rule_library"]).read_text(encoding="utf-8"))
            audit = json.loads((run_dir / "candidate_audit_v4_generic.json").read_text(encoding="utf-8"))
            split = json.loads((run_dir / "split_v4_generic.json").read_text(encoding="utf-8"))
            with (run_dir / "predictions_v4_generic.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            labels = np.asarray([int(row["label"]) for row in rows])
            baseline = np.asarray([float(row["baseline_score"]) for row in rows])
            fused = np.asarray([float(row["decision_score"]) for row in rows])
            baseline_metric = _metric(labels, baseline)
            fused_metric = _metric(labels, fused)
            qwen = library.get("qwen", {})
            agreement = str(qwen.get("consensus_support_agreement", "0/0")).split("/")
            agreement_count = int(agreement[0]) if agreement[0].isdigit() else 0
            checks = {
                "row_count_matches": len(rows) == item["test_count"] == len(split["test"]),
                "labels_match_split": labels.tolist() == [int(row["label"]) for row in split["test"]],
                "baseline_metrics_recompute": np.allclose(
                    baseline_metric, [item["baseline"]["i_auroc"], item["baseline"]["i_aupr"]], atol=1e-12
                ),
                "fused_metrics_recompute": np.allclose(
                    fused_metric, [item["baseline_plus_rules"]["i_auroc"], item["baseline_plus_rules"]["i_aupr"]], atol=1e-12
                ),
                "test_labels_not_used_for_selection": (
                    item["test_labels_used_for_training_or_selection"] is False
                    and split["test_labels_used_for_training_or_selection"] is False
                ),
                "runtime_is_llm_free": library.get("runtime_requires_llm") is False,
                "selected_rules_have_accepted_audit": all(
                    any(row.get("accepted") and row.get("predicate") == rule.get("predicate") for row in audit)
                    for rule in library.get("rules", [])
                ),
                "consensus_provenance_present": (
                    qwen.get("primary_entity") is None
                    or (
                        qwen.get("consensus_protocol")
                        == "four_independent_support_queries_3of4_count_consensus"
                        and len(qwen.get("per_support_proposals", [])) == 4
                        and agreement_count >= 3
                    )
                ),
            }
            all_checks.extend(checks.values())
            class_reports.append({
                "class_name": item["class_name"],
                "checks": checks,
                "all_checks_passed": all(checks.values()),
                "bootstrap": _bootstrap(
                    labels, baseline, fused, args.bootstrap, args.seed + index
                ),
            })
        macro_baseline = float(np.mean([item["baseline"]["i_auroc"] for item in summary["classes"]]))
        macro_fused = float(np.mean([item["baseline_plus_rules"]["i_auroc"] for item in summary["classes"]]))
        reports.append({
            "dataset": summary["dataset"],
            "class_count": len(class_reports),
            "macro_i_auroc_recomputed": {
                "baseline": macro_baseline, "fused": macro_fused,
                "delta": macro_fused - macro_baseline,
            },
            "classes": class_reports,
        })
    output = {
        "schema_version": "4.0-generic-relational-validation",
        "all_checks_passed": bool(all(all_checks)),
        "passed_checks": int(sum(all_checks)),
        "total_checks": len(all_checks),
        "datasets": reports,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.quiet:
        print(json.dumps({
            "all_checks_passed": output["all_checks_passed"],
            "passed_checks": output["passed_checks"],
            "total_checks": output["total_checks"],
            "datasets": [
                {"dataset": item["dataset"], **item["macro_i_auroc_recomputed"]}
                for item in output["datasets"]
            ],
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
