"""Per-region overlap (PRO / AUPRO) metric.

Implements the standard region-wise overlap curve and integrates it up to a
bounded false-positive rate.  The implementation pools regions across images,
which matches the dataset-level protocol used by MVTec LOCO reports.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import cv2
import numpy as np


def _build_regions(masks: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    region_ids: List[np.ndarray] = []
    sizes: List[int] = []
    offset = 0
    for mask in masks:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        count, labels = cv2.connectedComponents(binary, connectivity=8)
        if count > 1:
            raw = labels.reshape(-1).astype(np.int64)
            flat = np.where(raw > 0, raw - 1 + offset, -1)
            region_ids.append(flat)
            for index in range(1, count):
                sizes.append(int((labels == index).sum()))
            offset += count - 1
        else:
            region_ids.append(np.zeros(binary.size, dtype=np.int64) - 1)
    return np.concatenate(region_ids), np.asarray(sizes, dtype=np.float64)


def compute_pro(
    predictions: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    fpr_limit: float = 0.30,
    thresholds: int = 64,
) -> Dict[str, float]:
    if not predictions:
        return {"aupro": float("nan"), "regions": 0}
    flat_predictions = np.concatenate(
        [np.asarray(item, dtype=np.float32).reshape(-1) for item in predictions]
    )
    flat_masks = np.concatenate(
        [np.asarray(item, dtype=np.uint8).reshape(-1) > 0 for item in masks]
    )
    region_ids, region_sizes = _build_regions(masks)
    if region_sizes.size == 0:
        return {"aupro": float("nan"), "regions": 0}

    negatives = int((~flat_masks).sum())
    if negatives == 0:
        return {"aupro": float("nan"), "regions": int(region_sizes.size)}

    positive_values = flat_predictions[flat_masks]
    if positive_values.size == 0:
        return {"aupro": float("nan"), "regions": int(region_sizes.size)}
    quantiles = np.linspace(0.5, 0.9999, max(2, int(thresholds)))
    threshold_values = np.unique(np.quantile(positive_values, quantiles))
    threshold_values = np.concatenate([[0.0], threshold_values])

    fprs, pros = [], []
    for threshold in threshold_values:
        binary = flat_predictions >= threshold
        region_sums = np.bincount(
            region_ids[binary & (region_ids >= 0)], minlength=region_sizes.size
        )
        overlaps = region_sums / np.maximum(region_sizes, 1.0)
        false_positives = int((binary & ~flat_masks).sum())
        fpr = false_positives / float(negatives)
        fprs.append(fpr)
        pros.append(float(overlaps.mean()))

    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)
    order = np.argsort(fprs)
    fprs, pros = fprs[order], pros[order]

    limit = float(min(max(fpr_limit, 1e-6), max(float(fprs[-1]), 1e-6)))
    grid = np.linspace(0.0, limit, 100)
    interpolated = np.interp(grid, fprs, pros, left=0.0, right=pros[-1])
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area = float(trapezoid(interpolated, grid) / limit)
    return {
        "aupro": area,
        "regions": int(region_sizes.size),
        "fpr_limit": limit,
    }
