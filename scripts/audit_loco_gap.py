#!/usr/bin/env python3
"""Recompute the historical LOCO logical/structural baseline gap from CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _family(rows, family):
    keep = [
        row for row in rows
        if row["defect_type"].lower() in {"good", "ok"}
        or family == "all"
        or family in row["defect_type"].lower()
    ]
    labels = np.asarray([int(row["label"]) for row in keep], dtype=np.uint8)
    scores = np.asarray([float(row["baseline_score"]) for row in keep])
    return labels, scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    files = sorted(Path(args.prediction_root).glob("*/k4/predictions_v4_generic.csv"))
    if not files:
        raise FileNotFoundError(args.prediction_root)
    loaded = {
        path.parts[-3]: list(csv.DictReader(path.open(encoding="utf-8")))
        for path in files
    }
    per_class = {}
    for class_name, rows in loaded.items():
        per_class[class_name] = {}
        for family in ("all", "logical", "structural"):
            labels, scores = _family(rows, family)
            per_class[class_name][family] = {
                "image_auroc": float(roc_auc_score(labels, scores)),
                "image_ap": float(average_precision_score(labels, scores)),
                "count": int(len(labels)),
            }
    macro = {
        family: {
            metric: float(np.mean([
                row[family][metric] for row in per_class.values()
            ]))
            for metric in ("image_auroc", "image_ap")
        }
        for family in ("all", "logical", "structural")
    }
    rng = np.random.default_rng(args.seed)
    gaps = []
    for _ in range(args.bootstrap):
        class_gaps = []
        for rows in loaded.values():
            defects = np.asarray([row["defect_type"].lower() for row in rows])
            scores = np.asarray([float(row["baseline_score"]) for row in rows])
            good = np.flatnonzero(np.isin(defects, ["good", "ok"]))
            logical = np.flatnonzero(np.char.find(defects.astype(str), "logical") >= 0)
            structural = np.flatnonzero(
                np.char.find(defects.astype(str), "structural") >= 0
            )
            sampled_good = rng.choice(good, len(good))
            sampled_logical = rng.choice(logical, len(logical))
            sampled_structural = rng.choice(structural, len(structural))
            logical_labels = np.r_[np.zeros(len(good)), np.ones(len(logical))]
            structural_labels = np.r_[np.zeros(len(good)), np.ones(len(structural))]
            logical_auc = roc_auc_score(
                logical_labels, np.r_[scores[sampled_good], scores[sampled_logical]]
            )
            structural_auc = roc_auc_score(
                structural_labels,
                np.r_[scores[sampled_good], scores[sampled_structural]],
            )
            class_gaps.append(logical_auc - structural_auc)
        gaps.append(float(np.mean(class_gaps)))
    result = {
        "per_class": per_class,
        "macro": macro,
        "logical_minus_structural_auroc_bootstrap": {
            "replicates": args.bootstrap,
            "seed": args.seed,
            "median": float(np.median(gaps)),
            "ci95": np.quantile(gaps, [0.025, 0.975]).tolist(),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
