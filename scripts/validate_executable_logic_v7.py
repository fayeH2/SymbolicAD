#!/usr/bin/env python3
"""Independent metric and paired image-block bootstrap audit for v7."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


METRICS = ("i_auroc", "i_aupr", "p_auroc", "p_aupr")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def _hist_metric(positive: np.ndarray, negative: np.ndarray) -> tuple[float, float]:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    p_total = float(positive.sum())
    n_total = float(negative.sum())
    if p_total <= 0 or n_total <= 0:
        return float("nan"), float("nan")
    negative_below = np.cumsum(negative) - negative
    auc = float(np.sum(positive * (negative_below + 0.5 * negative)) / (p_total * n_total))
    tp = np.cumsum(positive[::-1])
    fp = np.cumsum(negative[::-1])
    recall = tp / p_total
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    delta_recall = np.diff(np.concatenate(([0.0], recall)))
    ap = float(np.sum(delta_recall * precision))
    return auc, ap


def _image_histograms(
    maps: np.ndarray, masks: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.zeros((len(maps), bins), dtype=np.int64)
    negative = np.zeros((len(maps), bins), dtype=np.int64)
    for index, (score_map, mask) in enumerate(zip(maps, masks)):
        indices = np.minimum(
            np.floor(np.clip(score_map, 0.0, 1.0) * bins).astype(np.int64),
            bins - 1,
        )
        positive[index] = np.bincount(indices[mask], minlength=bins)
        negative[index] = np.bincount(indices[~mask], minlength=bins)
    return positive, negative


def _bootstrap(
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    fused_scores: np.ndarray,
    baseline_hist: tuple[np.ndarray, np.ndarray],
    fused_hist: tuple[np.ndarray, np.ndarray],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    normal = np.flatnonzero(labels == 0)
    anomaly = np.flatnonzero(labels == 1)
    deltas = {key: np.empty(repeats, dtype=np.float64) for key in METRICS}
    for repeat in range(repeats):
        sampled = np.concatenate([
            rng.choice(normal, size=len(normal), replace=True),
            rng.choice(anomaly, size=len(anomaly), replace=True),
        ])
        weights = np.bincount(sampled, minlength=len(labels)).astype(np.float64)
        baseline_i = (
            float(roc_auc_score(labels, baseline_scores, sample_weight=weights)),
            float(average_precision_score(labels, baseline_scores, sample_weight=weights)),
        )
        fused_i = (
            float(roc_auc_score(labels, fused_scores, sample_weight=weights)),
            float(average_precision_score(labels, fused_scores, sample_weight=weights)),
        )
        baseline_p = _hist_metric(
            weights @ baseline_hist[0], weights @ baseline_hist[1]
        )
        fused_p = _hist_metric(
            weights @ fused_hist[0], weights @ fused_hist[1]
        )
        values = {
            "i_auroc": fused_i[0] - baseline_i[0],
            "i_aupr": fused_i[1] - baseline_i[1],
            "p_auroc": fused_p[0] - baseline_p[0],
            "p_aupr": fused_p[1] - baseline_p[1],
        }
        for key, value in values.items():
            deltas[key][repeat] = value
    return {
        key: {
            "mean_delta": float(np.mean(values)),
            "ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
            "probability_delta_gt_zero": float(np.mean(values > 0.0)),
        }
        for key, values in deltas.items()
    }


def validate(run_dir: Path, repeats: int, bins: int, seed: int) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    metrics = json.loads(
        (run_dir / "metrics_v7_executable_logic.json").read_text(encoding="utf-8")
    )
    rows = _rows(run_dir / "predictions_v7_executable_logic.csv")
    with np.load(run_dir / "pixel_predictions_v7.npz", allow_pickle=False) as payload:
        paths = payload["paths"].astype(str)
        labels = payload["labels"].astype(np.uint8)
        masks = payload["masks"].astype(bool)
        baseline_maps = payload["baseline_maps"].astype(np.float32)
        fused_maps = payload["fused_maps"].astype(np.float32)
    row_paths = np.asarray([str(Path(row["path"]).resolve()) for row in rows])
    if not np.array_equal(paths, row_paths):
        raise ValueError("CSV and pixel NPZ query paths differ")
    csv_labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
    if not np.array_equal(labels, csv_labels):
        raise ValueError("CSV and pixel NPZ labels differ")
    baseline_scores = np.asarray([float(row["baseline_score"]) for row in rows])
    fused_scores = np.asarray([float(row["fused_score"]) for row in rows])
    baseline_i = _metric(labels, baseline_scores)
    fused_i = _metric(labels, fused_scores)
    baseline_p = _metric(masks.reshape(-1), baseline_maps.reshape(-1))
    fused_p = _metric(masks.reshape(-1), fused_maps.reshape(-1))
    recomputed = {
        "baseline": dict(zip(METRICS, (*baseline_i, *baseline_p))),
        "neuro_symbolic": dict(zip(METRICS, (*fused_i, *fused_p))),
    }
    maximum_error = max(
        abs(recomputed[method][key] - float(metrics[method][key]))
        for method in recomputed for key in METRICS
    )
    baseline_hist = _image_histograms(baseline_maps, masks, bins)
    fused_hist = _image_histograms(fused_maps, masks, bins)
    full_hist = {
        "baseline": dict(zip(("p_auroc", "p_aupr"), _hist_metric(
            baseline_hist[0].sum(axis=0), baseline_hist[1].sum(axis=0)
        ))),
        "neuro_symbolic": dict(zip(("p_auroc", "p_aupr"), _hist_metric(
            fused_hist[0].sum(axis=0), fused_hist[1].sum(axis=0)
        ))),
    }
    histogram_max_error = max(
        abs(full_hist[method][key] - recomputed[method][key])
        for method in full_hist for key in ("p_auroc", "p_aupr")
    )
    library = json.loads(
        (run_dir / "normality_rules_v7_executable_logic.json").read_text(encoding="utf-8")
    )
    support = {str(Path(path).resolve()) for path in library["support"]}
    leakage = sorted(support.intersection(set(paths)))
    result = {
        "schema_version": "7.1-independent-metric-bootstrap-validation",
        "run_dir": str(run_dir),
        "query_count": len(rows),
        "exact_metric_recomputation": recomputed,
        "maximum_exact_metric_error": maximum_error,
        "exact_metrics_match": maximum_error <= 1e-12,
        "support_query_path_overlap": leakage,
        "support_query_disjoint": not leakage,
        "bootstrap": {
            "unit": "stratified query-image block",
            "repeats": repeats,
            "seed": seed,
            "pixel_histogram_bins": bins,
            "full_sample_histogram_max_error": histogram_max_error,
            "delta_neuro_symbolic_minus_baseline": _bootstrap(
                labels, baseline_scores, fused_scores,
                baseline_hist, fused_hist, repeats, seed,
            ),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--pixel-histogram-bins", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=668)
    args = parser.parse_args()
    result = validate(
        args.run_dir, args.repeats, args.pixel_histogram_bins, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
