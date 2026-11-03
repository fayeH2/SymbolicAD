"""Geometry helpers shared by support memory and relation grounding."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .scene_graph import ComponentInstance


def bbox_iou(a: ComponentInstance, b: ComponentInstance) -> float:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    intersection = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 1e-12 else 0.0


def containment(inner: ComponentInstance, outer: ComponentInstance) -> float:
    """Fraction of ``inner``'s mask that lies inside ``outer``'s bbox."""
    x0, y0, x1, y1 = outer.bbox
    height, width = inner.mask.shape[:2]
    xs = np.arange(width)
    ys = np.arange(height)
    inside_x = (xs / max(width - 1, 1) >= x0) & (xs / max(width - 1, 1) <= x1)
    inside_y = (ys / max(height - 1, 1) >= y0) & (ys / max(height - 1, 1) <= y1)
    region = inner.mask * inside_y[:, None] * inside_x[None, :]
    total = float(inner.mask.sum())
    return float(region.sum() / total) if total > 1e-12 else 0.0


def bbox_size(instance: ComponentInstance) -> Tuple[float, float]:
    x0, y0, x1, y1 = instance.bbox
    return max(x1 - x0, 1e-4), max(y1 - y0, 1e-4)


def pair_geometry(a: ComponentInstance, b: ComponentInstance) -> np.ndarray:
    """Relative geometry used by spatial relations.

    ``[dx, dy, log(w_a/w_b), log(h_a/h_b), IoU]`` with normalised coordinates.
    """
    dx = float(b.center[0] - a.center[0])
    dy = float(b.center[1] - a.center[1])
    width_a, height_a = bbox_size(a)
    width_b, height_b = bbox_size(b)
    return np.asarray(
        [
            dx,
            dy,
            math.log(width_a / width_b),
            math.log(height_a / height_b),
            bbox_iou(a, b),
        ],
        dtype=np.float64,
    )


def pair_feature(a: ComponentInstance, b: ComponentInstance) -> np.ndarray:
    """Appearance + geometry vector used by one-to-one matching."""
    fa = np.asarray(a.feature, dtype=np.float64)
    fb = np.asarray(b.feature, dtype=np.float64)
    geometry = pair_geometry(a, b)
    return np.concatenate([fa, fb, fa - fb, fa * fb, geometry])


def mahalanobis(vector: np.ndarray, mean: np.ndarray, inverse_cov: np.ndarray) -> float:
    diff = np.asarray(vector, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    value = float(diff @ inverse_cov @ diff)
    return max(0.0, value)


def fit_gaussian(samples: np.ndarray, ridge: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    mean = samples.mean(axis=0)
    if samples.shape[0] < 2:
        covariance = np.eye(samples.shape[1], dtype=np.float64)
    else:
        covariance = np.cov(samples, rowvar=False)
        covariance = np.atleast_2d(covariance)
    covariance = covariance + ridge * np.eye(covariance.shape[0], dtype=np.float64)
    inverse = np.linalg.pinv(covariance)
    return mean, inverse
