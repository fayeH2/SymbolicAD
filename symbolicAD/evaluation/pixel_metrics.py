"""Pixel-level AUROC / AP / F1-max with pooled, calibrated predictions."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

try:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
    )
except Exception:  # pragma: no cover
    average_precision_score = roc_auc_score = precision_recall_curve = None


def _subsample(
    predictions: np.ndarray,
    labels: np.ndarray,
    max_points: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or predictions.size <= max_points:
        return predictions, labels
    rng = np.random.default_rng(seed)
    indices = rng.choice(predictions.size, size=max_points, replace=False)
    return predictions[indices], labels[indices]


def _f1_max(labels: np.ndarray, predictions: np.ndarray) -> float:
    if precision_recall_curve is None or len(np.unique(labels)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, predictions)
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(precision),
        where=denominator > 0,
    )
    return float(np.nanmax(f1)) if f1.size else float("nan")


def pixel_metrics(
    predictions: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    max_points: int = 0,
    seed: int = 0,
) -> Dict[str, float]:
    """Pool all images before computing metrics.

    Predictions and masks must already share the same resolution.  Per-image
    min-max normalisation is intentionally never applied here.
    """
    if not predictions:
        return {"auroc": float("nan"), "ap": float("nan"), "f1max": float("nan"), "pixels": 0}
    flat_predictions = np.concatenate(
        [np.asarray(item, dtype=np.float32).reshape(-1) for item in predictions]
    )
    flat_labels = np.concatenate(
        [np.asarray(item, dtype=np.uint8).reshape(-1) for item in masks]
    )
    flat_predictions, flat_labels = _subsample(
        flat_predictions, flat_labels, max_points, seed=seed
    )
    if flat_labels.size == 0 or len(np.unique(flat_labels)) < 2:
        return {
            "auroc": float("nan"),
            "ap": float("nan"),
            "f1max": float("nan"),
            "pixels": int(flat_labels.size),
        }
    if roc_auc_score is None:  # pragma: no cover
        return {
            "auroc": float("nan"),
            "ap": float("nan"),
            "f1max": float("nan"),
            "pixels": int(flat_labels.size),
        }
    return {
        "auroc": float(roc_auc_score(flat_labels, flat_predictions)),
        "ap": float(average_precision_score(flat_labels, flat_predictions)),
        "f1max": _f1_max(flat_labels, flat_predictions),
        "pixels": int(flat_labels.size),
    }
