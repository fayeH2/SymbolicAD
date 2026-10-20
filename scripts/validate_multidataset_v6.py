#!/usr/bin/env python3
"""Independently recompute v6 metrics and audit rule-library lineage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(labels, scores):
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def _bootstrap(labels, baseline, fused, repeats, seed):
    labels = np.asarray(labels, dtype=int)
    baseline = np.asarray(baseline, dtype=float)
    fused = np.asarray(fused, dtype=float)
    if np.array_equal(baseline, fused):
        return {"mean": 0.0, "ci95": [0.0, 0.0], "probability_positive": 0.0}
    normal = np.flatnonzero(labels == 0)
    anomaly = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = np.concatenate([
            rng.choice(normal, len(normal), replace=True),
            rng.choice(anomaly, len(anomaly), replace=True),
        ])
        values.append(
            roc_auc_score(labels[indices], fused[indices])
            - roc_auc_score(labels[indices], baseline[indices])
        )
    values = np.asarray(values)
    return {
        "mean": float(np.mean(values)),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "probability_positive": float(np.mean(values > 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()

    summary = _load(Path(args.summary).expanduser().resolve())
    reports, all_checks = [], []
    for index, row in enumerate(summary["classes"]):
        metric_path = Path(row["generic_metrics"])
        run_dir = metric_path.parent
        metric = _load(metric_path)
        library = _load(Path(metric["rule_library"]))
        split = _load(run_dir / "split_v4_generic.json")
        audit = _load(run_dir / "candidate_audit_v4_generic.json")
        with (run_dir / "predictions_v4_generic.csv").open(newline="", encoding="utf-8") as handle:
            predictions = list(csv.DictReader(handle))
        labels = np.asarray([int(item["label"]) for item in predictions])
        baseline = np.asarray([float(item["baseline_score"]) for item in predictions])
        fused = np.asarray([float(item["decision_score"]) for item in predictions])
        baseline_metric = _metrics(labels, baseline)
        fused_metric = _metrics(labels, fused)
        support = {str(Path(path).resolve()) for path in split["support"]}
        test = {str(Path(item["path"]).resolve()) for item in split["test"]}
        accepted = {item.get("predicate") for item in audit if item.get("accepted")}
        qwen = library.get("qwen", {})
        proposal_ok = (
            qwen.get("primary_entity") is None
            or (
                len(qwen.get("per_support_proposals", [])) == metric["shot"]
                and "independent_support_queries" in qwen.get("consensus_protocol", "")
            )
        )
        checks = {
            "both_test_labels_present": set(labels.tolist()) == {0, 1},
            "row_count_matches": len(predictions) == metric["test_count"] == len(split["test"]),
            "labels_match_split": labels.tolist() == [int(item["label"]) for item in split["test"]],
            "baseline_metrics_recomputed": bool(np.allclose(
                baseline_metric,
                [metric["baseline"]["i_auroc"], metric["baseline"]["i_aupr"]],
                atol=1e-12,
            )),
            "fused_metrics_recomputed": bool(np.allclose(
                fused_metric,
                [metric["baseline_plus_rules"]["i_auroc"], metric["baseline_plus_rules"]["i_aupr"]],
                atol=1e-12,
            )),
            "support_test_disjoint": not bool(support & test),
            "test_labels_not_used_for_selection": (
                metric["test_labels_used_for_training_or_selection"] is False
                and split["test_labels_used_for_training_or_selection"] is False
            ),
            "runtime_is_llm_free": library.get("runtime_requires_llm") is False,
            "selected_rules_have_accepted_audit": all(
                item.get("predicate") in accepted for item in library.get("rules", [])
            ),
            "qwen_consensus_provenance": proposal_ok,
        }
        all_checks.extend(checks.values())
        reports.append({
            "domain": row["domain"],
            "dataset": row["dataset"],
            "class_name": row["class_name"],
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "delta_i_auroc": metric["delta_i_auroc"],
            "bootstrap_delta_i_auroc": _bootstrap(
                labels, baseline, fused, args.bootstrap, args.seed + index
            ),
        })
    output = {
        "schema_version": "6.0-multidataset-independent-validation",
        "all_checks_passed": bool(all(all_checks)),
        "passed_checks": int(sum(all_checks)),
        "total_checks": len(all_checks),
        "bootstrap_repeats": args.bootstrap,
        "classes": reports,
    }
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "all_checks_passed": output["all_checks_passed"],
        "passed_checks": output["passed_checks"],
        "total_checks": output["total_checks"],
    }, ensure_ascii=False, indent=2))
    return 0 if output["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
