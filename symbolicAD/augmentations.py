"""Support-only calibration views and synthetic local defects.

These transforms are deliberately deterministic and label-free.  Mild global
changes approximate nuisance variation that a normality rule must tolerate;
local corruptions provide counterexamples for ranking candidate rules.  They do
not replace real anomaly validation and are never mixed into reported test data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Union

import cv2
import numpy as np
from PIL import Image


@dataclass
class CalibrationView:
    image: np.ndarray
    source_index: int
    transform: str
    is_pseudo_anomaly: bool


def _rgb(image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            return np.asarray(handle.convert("RGB"), dtype=np.uint8)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    return array


def _affine(array: np.ndarray, angle: float = 0.0, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    height, width = array.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    matrix[:, 2] += (tx * width, ty * height)
    return cv2.warpAffine(
        array, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _brightness(array: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(array.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _contrast(array: np.ndarray, factor: float) -> np.ndarray:
    data = array.astype(np.float32)
    mean = data.mean(axis=(0, 1), keepdims=True)
    return np.clip((data - mean) * factor + mean, 0, 255).astype(np.uint8)


def normal_views(image, source_index: int) -> List[CalibrationView]:
    """Nuisance views that valid normality rules should accept."""
    array = _rgb(image)
    variants = [
        ("identity", array),
        ("rotate_m16", _affine(array, angle=-16.0)),
        ("rotate_m8", _affine(array, angle=-8.0)),
        ("rotate_p8", _affine(array, angle=8.0)),
        ("rotate_p16", _affine(array, angle=16.0)),
        ("translate_left", _affine(array, tx=-0.06)),
        ("translate_right", _affine(array, tx=0.06)),
        ("translate_up", _affine(array, ty=-0.06)),
        ("translate_down", _affine(array, ty=0.06)),
        ("brightness_078", _brightness(array, 0.78)),
        ("brightness_090", _brightness(array, 0.90)),
        ("brightness_110", _brightness(array, 1.10)),
        ("brightness_122", _brightness(array, 1.22)),
        ("contrast_080", _contrast(array, 0.80)),
        ("contrast_120", _contrast(array, 1.20)),
        ("blur", cv2.GaussianBlur(array, (5, 5), 1.0)),
    ]
    return [CalibrationView(item, source_index, name, False) for name, item in variants]


def _box(height: int, width: int, fraction: float, cx: float, cy: float):
    box_h = max(4, int(round(height * fraction)))
    box_w = max(4, int(round(width * fraction)))
    x0 = int(np.clip(round(cx * width - box_w / 2), 0, width - box_w))
    y0 = int(np.clip(round(cy * height - box_h / 2), 0, height - box_h))
    return x0, y0, x0 + box_w, y0 + box_h


def pseudo_anomaly_views(image, source_index: int, seed: int = 20261013) -> List[CalibrationView]:
    """Create local missing, misplaced, color, and scratch counterexamples."""
    array = _rgb(image)
    height, width = array.shape[:2]
    rng = np.random.default_rng(seed + source_index * 1009)
    result: List[CalibrationView] = []

    for index, (cx, cy) in enumerate(((0.35, 0.35), (0.65, 0.45), (0.50, 0.68))):
        x0, y0, x1, y1 = _box(height, width, 0.16 + 0.03 * index, cx, cy)
        changed = array.copy()
        local = changed[y0:y1, x0:x1]
        fill = np.median(array.reshape(-1, 3), axis=0)
        local[:] = np.clip(0.35 * local + 0.65 * fill, 0, 255).astype(np.uint8)
        result.append(CalibrationView(changed, source_index, f"local_missing_{index}", True))

    for index, (source_xy, target_xy) in enumerate((
        ((0.30, 0.40), (0.68, 0.62)),
        ((0.65, 0.35), (0.35, 0.68)),
        ((0.50, 0.62), (0.52, 0.30)),
    )):
        sx0, sy0, sx1, sy1 = _box(height, width, 0.15, *source_xy)
        tx0, ty0, tx1, ty1 = _box(height, width, 0.15, *target_xy)
        patch = array[sy0:sy1, sx0:sx1].copy()
        changed = array.copy()
        target = changed[ty0:ty1, tx0:tx1]
        if target.shape == patch.shape:
            target[:] = patch
        result.append(CalibrationView(changed, source_index, f"local_misplace_{index}", True))

    for index, (cx, cy) in enumerate(((0.38, 0.55), (0.62, 0.52))):
        x0, y0, x1, y1 = _box(height, width, 0.18, cx, cy)
        changed = array.copy().astype(np.float32)
        patch = changed[y0:y1, x0:x1]
        channel_scale = np.asarray((1.55, 0.55, 1.25) if index == 0 else (0.55, 1.45, 0.70))
        patch *= channel_scale[None, None, :]
        changed = np.clip(changed, 0, 255).astype(np.uint8)
        result.append(CalibrationView(changed, source_index, f"local_color_{index}", True))

    for index in range(3):
        changed = array.copy()
        start = (int(width * (0.18 + 0.08 * index)), int(height * (0.25 + 0.18 * index)))
        end = (int(width * (0.78 - 0.05 * index)), int(height * (0.62 + 0.08 * index)))
        color = tuple(int(value) for value in rng.integers(15, 240, size=3))
        thickness = max(2, int(round(min(height, width) * (0.008 + 0.003 * index))))
        cv2.line(changed, start, end, color, thickness=thickness, lineType=cv2.LINE_AA)
        result.append(CalibrationView(changed, source_index, f"scratch_{index}", True))
    return result


def build_calibration_views(
    support_images: Sequence, seed: int = 20261013,
) -> List[CalibrationView]:
    views: List[CalibrationView] = []
    for source_index, image in enumerate(support_images):
        views.extend(normal_views(image, source_index))
        views.extend(pseudo_anomaly_views(image, source_index, seed=seed))
    return views
