"""Image-level AUROC / AP / F1-max."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

try:  # sklearn is a hard requirement in the project environment
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
    )
except Exception:  # pragma: no cover - fallback for minimal environments
    average_precision_score = roc_auc_score = precision_recall_curve = None


def _f1_max(labels: np.ndarray, scores: np.ndarray) -> float:
    if precision_recall_curve is None or len(np.unique(labels)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, scores)
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(precision),
        where=denominator > 0,
    )
    return float(np.nanmax(f1)) if f1.size else float("nan")


def image_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0:
        return {"auroc": float("nan"), "ap": float("nan"), "f1max": float("nan"), "n": 0}
    if len(np.unique(labels)) < 2:
        return {
            "auroc": float("nan"),
            "ap": float("nan"),
            "f1max": float("nan"),
            "n": int(labels.size),
        }
    if roc_auc_score is None:  # pragma: no cover
        return {"auroc": float("nan"), "ap": float("nan"), "f1max": float("nan"), "n": int(labels.size)}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "f1max": _f1_max(labels, scores),
        "n": int(labels.size),
    }
