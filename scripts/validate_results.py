#!/usr/bin/env python3
"""Independently validate saved experiment metrics and bootstrap paired deltas."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics(labels, scores):
    return {
        "i_auroc": float(roc_auc_score(labels, scores)),
        "i_aupr": float(average_precision_score(labels, scores)),
    }


def paired_bootstrap(labels, baseline, fused, iterations=5000, seed=20261012):
    labels = np.asarray(labels, dtype=int)
    baseline = np.asarray(baseline, dtype=float)
    fused = np.asarray(fused, dtype=float)
    normal = np.flatnonzero(labels == 0)
    anomaly = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    auroc_delta = np.empty(iterations, dtype=float)
    aupr_delta = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        chosen = np.concatenate([
            rng.choice(normal, len(normal), replace=True),
            rng.choice(anomaly, len(anomaly), replace=True),
        ])
        y = labels[chosen]
        auroc_delta[iteration] = (
            roc_auc_score(y, fused[chosen]) - roc_auc_score(y, baseline[chosen])
        )
        aupr_delta[iteration] = (
            average_precision_score(y, fused[chosen])
            - average_precision_score(y, baseline[chosen])
        )
    return {
        "delta_i_auroc_ci95": [float(x) for x in np.quantile(auroc_delta, [0.025, 0.975])],
        "delta_i_aupr_ci95": [float(x) for x in np.quantile(aupr_delta, [0.025, 0.975])],
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
    }


def validate_run(run_dir, iterations):
    run_dir = Path(run_dir).resolve()
    saved = read_json(run_dir / "metrics.json")
    split = read_json(run_dir / "split.json")
    library = read_json(run_dir / "normality_rules.json")
    proposal = read_json(run_dir / "rule_proposal.json")
    with (run_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    baseline = np.asarray([float(row["baseline_score"]) for row in rows])
    rule = np.asarray([float(row["rule_score"]) for row in rows])
    fused = np.asarray([float(row["fused_score"]) for row in rows])
    recomputed = {
        "baseline": metrics(labels, baseline),
        "rules_only": metrics(labels, rule),
        "baseline_plus_rules": metrics(labels, fused),
    }
    maximum_error = max(
        abs(recomputed[group][metric] - saved[group][metric])
        for group in recomputed for metric in recomputed[group]
    )
    test_paths = {str(Path(row["path"]).resolve()) for row in rows}
    support_paths = {str(Path(path).resolve()) for path in library["support_paths"]}
    support_is_normal_train = all(
        ("/train/good/" in path or "/train/ok/" in path) for path in support_paths
    )
    leakage_free = support_paths.isdisjoint(test_paths)

    defect_breakdown = {}
    normal_indices = np.flatnonzero(labels == 0)
    defect_types = sorted({row["defect_type"] for row in rows if int(row["label"]) == 1})
    for defect_type in defect_types:
        defect_indices = np.asarray([
            index for index, row in enumerate(rows)
            if int(row["label"]) == 1 and row["defect_type"] == defect_type
        ])
        chosen = np.concatenate([normal_indices, defect_indices])
        defect_breakdown[defect_type] = {
            "anomaly_count": int(len(defect_indices)),
            "baseline_i_auroc": float(roc_auc_score(labels[chosen], baseline[chosen])),
            "fused_i_auroc": float(roc_auc_score(labels[chosen], fused[chosen])),
        }
        defect_breakdown[defect_type]["delta_i_auroc"] = (
            defect_breakdown[defect_type]["fused_i_auroc"]
            - defect_breakdown[defect_type]["baseline_i_auroc"]
        )

    bootstrap = paired_bootstrap(labels, baseline, fused, iterations=iterations)
    return {
        "run_dir": str(run_dir),
        "dataset": saved["dataset"],
        "class_name": saved["class_name"],
        "shot": saved["shot"],
        "test_count": int(len(rows)),
        "normal_count": int((labels == 0).sum()),
        "anomaly_count": int((labels == 1).sum()),
        "support_paths": sorted(support_paths),
        "support_matches_shot": len(support_paths) == int(saved["shot"]),
        "support_is_normal_train": support_is_normal_train,
        "support_test_disjoint": leakage_free,
        "metrics_recomputed": recomputed,
        "saved_metric_max_abs_error": maximum_error,
        "delta_i_auroc": recomputed["baseline_plus_rules"]["i_auroc"] - recomputed["baseline"]["i_auroc"],
        "delta_i_aupr": recomputed["baseline_plus_rules"]["i_aupr"] - recomputed["baseline"]["i_aupr"],
        **bootstrap,
        "selected_features": [rule_item["feature"] for rule_item in proposal["rules"]],
        "defect_breakdown": defect_breakdown,
        "valid": bool(
            maximum_error < 1e-12 and leakage_free and support_is_normal_train
            and len(support_paths) == int(saved["shot"])
            and len(split["test"]) == len(rows)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--output-root", default="results")
    args = parser.parse_args()
    results = [validate_run(path, args.iterations) for path in args.run_dirs]
    baseline_mean = float(np.mean([
        row["metrics_recomputed"]["baseline"]["i_auroc"] for row in results
    ]))
    fused_mean = float(np.mean([
        row["metrics_recomputed"]["baseline_plus_rules"]["i_auroc"] for row in results
    ]))
    payload = {
        "validation_status": "share_with_caveats" if all(row["valid"] for row in results) else "needs_revision",
        "scope": "four selected 4-shot classes; not a full benchmark or significance claim",
        "macro_i_auroc_baseline": baseline_mean,
        "macro_i_auroc_fused": fused_mean,
        "macro_delta_i_auroc": fused_mean - baseline_mean,
        "runs": results,
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "validated_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "validated_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "dataset", "class_name", "shot", "test_count", "normal_count", "anomaly_count",
            "baseline_i_auroc", "rules_i_auroc", "fused_i_auroc", "delta_i_auroc",
            "delta_i_auroc_ci95_low", "delta_i_auroc_ci95_high",
            "baseline_i_aupr", "fused_i_aupr", "delta_i_aupr", "valid",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "dataset": row["dataset"], "class_name": row["class_name"],
                "shot": row["shot"], "test_count": row["test_count"],
                "normal_count": row["normal_count"], "anomaly_count": row["anomaly_count"],
                "baseline_i_auroc": row["metrics_recomputed"]["baseline"]["i_auroc"],
                "rules_i_auroc": row["metrics_recomputed"]["rules_only"]["i_auroc"],
                "fused_i_auroc": row["metrics_recomputed"]["baseline_plus_rules"]["i_auroc"],
                "delta_i_auroc": row["delta_i_auroc"],
                "delta_i_auroc_ci95_low": row["delta_i_auroc_ci95"][0],
                "delta_i_auroc_ci95_high": row["delta_i_auroc_ci95"][1],
                "baseline_i_aupr": row["metrics_recomputed"]["baseline"]["i_aupr"],
                "fused_i_aupr": row["metrics_recomputed"]["baseline_plus_rules"]["i_aupr"],
                "delta_i_aupr": row["delta_i_aupr"], "valid": row["valid"],
            })
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
