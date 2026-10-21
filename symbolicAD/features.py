"""Deterministic visual facts exposed to the symbolic rule engine.

The LLM is deliberately not allowed to invent executable Python expressions.  It
may only select facts from ``FEATURE_CATALOG`` and specify whether a normal value
should be stable, low, or high.  This keeps the final rules inspectable and makes
online inference independent of the LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np
from PIL import Image


# ``tolerance`` is a conservative absolute margin used when only one or two
# normal supports are available.  All features are dimensionless or normalized.
FEATURE_CATALOG: Dict[str, Dict[str, Any]] = {
    "luminance_mean": {
        "description": "mean image luminance in [0,1]",
        "default_direction": "stable", "tolerance": 0.06,
    },
    "luminance_std": {
        "description": "global luminance contrast",
        "default_direction": "stable", "tolerance": 0.04,
    },
    "saturation_mean": {
        "description": "mean HSV color saturation",
        "default_direction": "stable", "tolerance": 0.06,
    },
    "colorfulness": {
        "description": "global colorfulness based on opponent RGB channels",
        "default_direction": "stable", "tolerance": 0.06,
    },
    "image_entropy": {
        "description": "normalized grayscale histogram entropy",
        "default_direction": "stable", "tolerance": 0.04,
    },
    "edge_strength": {
        "description": "mean Sobel edge magnitude",
        "default_direction": "stable", "tolerance": 0.03,
    },
    "border_contrast": {
        "description": "luminance difference between border and image center",
        "default_direction": "stable", "tolerance": 0.04,
    },
    "lr_symmetry_error": {
        "description": "left-right grayscale reflection error; lower is more symmetric",
        "default_direction": "stable", "tolerance": 0.05,
    },
    "tb_symmetry_error": {
        "description": "top-bottom grayscale reflection error; lower is more symmetric",
        "default_direction": "stable", "tolerance": 0.05,
    },
    "foreground_area": {
        "description": "foreground silhouette area divided by image area",
        "default_direction": "stable", "tolerance": 0.06,
    },
    "foreground_cx": {
        "description": "normalized horizontal centroid of the foreground",
        "default_direction": "stable", "tolerance": 0.06,
    },
    "foreground_cy": {
        "description": "normalized vertical centroid of the foreground",
        "default_direction": "stable", "tolerance": 0.06,
    },
    "bbox_aspect_log": {
        "description": "log width/height of the foreground bounding box",
        "default_direction": "stable", "tolerance": 0.15,
    },
    "component_count": {
        "description": "number of non-trivial foreground connected components",
        "default_direction": "stable", "tolerance": 0.75,
    },
    "mask_lr_symmetry_error": {
        "description": "left-right reflection error of the foreground silhouette",
        "default_direction": "stable", "tolerance": 0.08,
    },
    "mask_tb_symmetry_error": {
        "description": "top-bottom reflection error of the foreground silhouette",
        "default_direction": "stable", "tolerance": 0.08,
    },
    "map_mean": {
        "description": "mean INP-Former anomaly-map response",
        "default_direction": "low", "tolerance": 0.015,
    },
    "map_std": {
        "description": "standard deviation of INP-Former anomaly-map response",
        "default_direction": "low", "tolerance": 0.015,
    },
    "map_p95": {
        "description": "95th percentile INP-Former anomaly response",
        "default_direction": "low", "tolerance": 0.025,
    },
    "map_p99": {
        "description": "99th percentile INP-Former anomaly response",
        "default_direction": "low", "tolerance": 0.03,
    },
    "map_top1_mean": {
        "description": "mean of the hottest 1 percent anomaly-map pixels (baseline image score)",
        "default_direction": "low", "tolerance": 0.03,
    },
    "map_peak_ratio": {
        "description": "map p99 divided by map mean; high means unusually concentrated response",
        "default_direction": "low", "tolerance": 0.75,
    },
    "map_center_x": {
        "description": "horizontal center of mass of anomaly response",
        "default_direction": "stable", "tolerance": 0.08,
    },
    "map_center_y": {
        "description": "vertical center of mass of anomaly response",
        "default_direction": "stable", "tolerance": 0.08,
    },
    "map_border_ratio": {
        "description": "mean border anomaly response divided by full-map mean",
        "default_direction": "stable", "tolerance": 0.25,
    },
    "map_lr_imbalance": {
        "description": "relative anomaly-energy imbalance between left and right halves",
        "default_direction": "low", "tolerance": 0.10,
    },
    "map_tb_imbalance": {
        "description": "relative anomaly-energy imbalance between top and bottom halves",
        "default_direction": "low", "tolerance": 0.10,
    },
    "map_spatial_entropy": {
        "description": "normalized entropy of anomaly energy across the image",
        "default_direction": "stable", "tolerance": 0.05,
    },
    "map_hot_component": {
        "description": "largest connected hot region as a fraction of image area",
        "default_direction": "low", "tolerance": 0.02,
    },
}


def _as_rgb(image: Any, max_side: int = 256) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
            array = np.clip(array * scale, 0, 255).astype(np.uint8)
    height, width = array.shape[:2]
    if max(height, width) > max_side:
        ratio = max_side / float(max(height, width))
        array = cv2.resize(
            array, (max(1, round(width * ratio)), max(1, round(height * ratio))),
            interpolation=cv2.INTER_AREA,
        )
    return array


def prepare_inp_aligned_rgb(
    image: Any, input_size: int = 448, crop_size: int = 392,
) -> np.ndarray:
    """Return the RGB view seen by INP-Former before tensor normalization.

    The v1 facts were measured on the uncropped source image while the baseline
    was measured after ``Resize((448, 448)) + CenterCrop(392)``.  Geometry and
    luminance facts therefore referred to a different field of view.  Keeping
    this small, dependency-free implementation here also avoids importing or
    modifying the external INP-Former repository.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            pil = handle.convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
            array = np.clip(array * scale, 0, 255).astype(np.uint8)
        pil = Image.fromarray(array, mode="RGB")
    pil = pil.resize((input_size, input_size), Image.Resampling.BILINEAR)
    offset = max(0, (input_size - crop_size) // 2)
    pil = pil.crop((offset, offset, offset + crop_size, offset + crop_size))
    return np.asarray(pil, dtype=np.uint8)


def _border_values(array: np.ndarray, fraction: float = 0.08) -> np.ndarray:
    height, width = array.shape[:2]
    border = max(1, int(round(min(height, width) * fraction)))
    parts = [
        array[:border].reshape(-1, *array.shape[2:]),
        array[-border:].reshape(-1, *array.shape[2:]),
        array[border:-border, :border].reshape(-1, *array.shape[2:]),
        array[border:-border, -border:].reshape(-1, *array.shape[2:]),
    ]
    return np.concatenate([part for part in parts if part.size], axis=0)


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    data = rgb.astype(np.float32) / 255.0
    border = _border_values(data)
    background = np.median(border, axis=0)
    distance = np.sqrt(np.sum((data - background) ** 2, axis=2))
    border_distance = np.sqrt(np.sum((border - background) ** 2, axis=1))
    median = float(np.median(border_distance))
    mad = float(np.median(np.abs(border_distance - median)))
    threshold = max(0.08, median + 4.0 * 1.4826 * mad)
    mask = (distance > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _hist_entropy(gray: np.ndarray) -> float:
    hist = np.histogram(gray, bins=64, range=(0.0, 1.0))[0].astype(np.float64)
    probability = hist / max(float(hist.sum()), 1.0)
    probability = probability[probability > 0]
    entropy = -float(np.sum(probability * np.log(probability)))
    return entropy / np.log(64.0)


def _appearance_features(rgb: np.ndarray) -> Dict[str, float]:
    data = rgb.astype(np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    red, green, blue = data[..., 0], data[..., 1], data[..., 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    colorfulness = np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(
        np.mean(rg) ** 2 + np.mean(yb) ** 2
    )

    height, width = gray.shape
    pad_y, pad_x = max(1, height // 5), max(1, width // 5)
    center = gray[pad_y: height - pad_y, pad_x: width - pad_x]
    border_gray = _border_values(gray[..., None]).reshape(-1)

    mask = _foreground_mask(rgb)
    foreground_area = float(mask.mean())
    ys, xs = np.nonzero(mask)
    if len(xs):
        foreground_cx = float(np.mean(xs) / max(width - 1, 1))
        foreground_cy = float(np.mean(ys) / max(height - 1, 1))
        box_width = int(xs.max() - xs.min() + 1)
        box_height = int(ys.max() - ys.min() + 1)
        bbox_aspect_log = float(np.log(max(box_width, 1) / max(box_height, 1)))
    else:
        foreground_cx = foreground_cy = 0.5
        bbox_aspect_log = 0.0

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(4, int(round(mask.size * 0.001)))
    component_count = float(sum(stats[index, cv2.CC_STAT_AREA] >= min_area
                                for index in range(1, count)))

    return {
        "luminance_mean": float(gray.mean()),
        "luminance_std": float(gray.std()),
        "saturation_mean": float(hsv[..., 1].mean() / 255.0),
        "colorfulness": float(colorfulness),
        "image_entropy": _hist_entropy(gray),
        "edge_strength": float(np.mean(np.clip(edge, 0.0, 1.0))),
        "border_contrast": float(abs(float(border_gray.mean()) - float(center.mean()))),
        "lr_symmetry_error": float(np.mean(np.abs(gray - np.fliplr(gray)))),
        "tb_symmetry_error": float(np.mean(np.abs(gray - np.flipud(gray)))),
        "foreground_area": foreground_area,
        "foreground_cx": foreground_cx,
        "foreground_cy": foreground_cy,
        "bbox_aspect_log": bbox_aspect_log,
        "component_count": component_count,
        "mask_lr_symmetry_error": float(np.mean(np.abs(mask.astype(float) - np.fliplr(mask)))),
        "mask_tb_symmetry_error": float(np.mean(np.abs(mask.astype(float) - np.flipud(mask)))),
    }


def _map_features(anomaly_map: np.ndarray) -> Dict[str, float]:
    score = np.asarray(anomaly_map, dtype=np.float64).squeeze()
    if score.ndim != 2:
        raise ValueError(f"anomaly_map must be 2D after squeeze, got {score.shape}")
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    score = np.maximum(score, 0.0)
    height, width = score.shape
    flat = score.reshape(-1)
    count = max(1, int(round(flat.size * 0.01)))
    top1 = np.partition(flat, flat.size - count)[-count:]
    mean = float(flat.mean())
    p95, p99 = (float(value) for value in np.quantile(flat, [0.95, 0.99]))

    mass = float(score.sum())
    if mass > 1e-12:
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        center_x = float(np.sum(score * x_grid) / mass / max(width - 1, 1))
        center_y = float(np.sum(score * y_grid) / mass / max(height - 1, 1))
        probability = flat / mass
        probability = probability[probability > 0]
        spatial_entropy = -float(np.sum(probability * np.log(probability))) / np.log(flat.size)
    else:
        center_x = center_y = 0.5
        spatial_entropy = 1.0

    border = max(1, int(round(min(height, width) * 0.15)))
    border_mask = np.zeros_like(score, dtype=bool)
    border_mask[:border] = True
    border_mask[-border:] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    border_ratio = float(score[border_mask].mean() / (mean + 1e-12))
    left, right = float(score[:, : width // 2].sum()), float(score[:, width // 2:].sum())
    top, bottom = float(score[: height // 2].sum()), float(score[height // 2:].sum())

    hot = (score >= p95).astype(np.uint8)
    components, _, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
    largest = max((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, components)), default=0)

    return {
        "map_mean": mean,
        "map_std": float(flat.std()),
        "map_p95": p95,
        "map_p99": p99,
        "map_top1_mean": float(top1.mean()),
        "map_peak_ratio": float(p99 / (mean + 1e-12)),
        "map_center_x": center_x,
        "map_center_y": center_y,
        "map_border_ratio": border_ratio,
        "map_lr_imbalance": float(abs(left - right) / (left + right + 1e-12)),
        "map_tb_imbalance": float(abs(top - bottom) / (top + bottom + 1e-12)),
        "map_spatial_entropy": spatial_entropy,
        "map_hot_component": float(largest / flat.size),
    }


def extract_features(
    image: Any,
    anomaly_map: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Extract the complete, finite fact vector for one sample."""
    features = _appearance_features(_as_rgb(image))
    if anomaly_map is not None:
        features.update(_map_features(anomaly_map))
    for name, value in features.items():
        if not np.isfinite(value):
            raise ValueError(f"Non-finite feature {name}={value}")
    return features


def compact_feature_catalog() -> Mapping[str, Mapping[str, Any]]:
    """Return prompt-safe public feature metadata."""
    return {
        name: {
            "description": metadata["description"],
        }
        for name, metadata in FEATURE_CATALOG.items()
    }
