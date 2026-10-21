"""Mask-based part measurements with explicit length and orientation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class ShapeMeasurement:
    length: float
    width: float
    aspect_ratio: float
    orientation_radians: float
    area: float
    perimeter: float
    solidity: float
    endpoint_a: tuple[float, float]
    endpoint_b: tuple[float, float]

    def to_dict(self) -> Dict[str, float | list[float]]:
        return {
            "length": self.length,
            "width": self.width,
            "aspect_ratio": self.aspect_ratio,
            "orientation_radians": self.orientation_radians,
            "area": self.area,
            "perimeter": self.perimeter,
            "solidity": self.solidity,
            "endpoint_a": list(self.endpoint_a),
            "endpoint_b": list(self.endpoint_b),
        }


def measure_mask_shape(mask: np.ndarray) -> Optional[ShapeMeasurement]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or np.count_nonzero(binary) < 2:
        return None
    height, width = binary.shape
    ys, xs = np.nonzero(binary)
    points = np.column_stack((xs / max(width - 1, 1), ys / max(height - 1, 1)))
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(len(points), 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, -1]
    if major[int(np.argmax(np.abs(major)))] < 0:
        major = -major
    minor = np.asarray([-major[1], major[0]], dtype=float)
    major_projection = centered @ major
    minor_projection = centered @ minor
    length = float(np.ptp(major_projection))
    thickness = float(np.ptp(minor_projection))
    end_a = center + major * float(np.min(major_projection))
    end_b = center + major * float(np.max(major_projection))

    contours = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    contour = max(contours, key=cv2.contourArea) if contours else None
    perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
    hull = cv2.convexHull(contour) if contour is not None else None
    hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
    pixel_area = float(np.count_nonzero(binary) / binary.size)
    solidity = pixel_area / max(hull_area / binary.size, 1e-8)
    return ShapeMeasurement(
        length=length,
        width=thickness,
        aspect_ratio=length / max(thickness, 1e-8),
        orientation_radians=float(np.arctan2(major[1], major[0])),
        area=pixel_area,
        perimeter=perimeter / max(height + width, 1),
        solidity=float(np.clip(solidity, 0.0, 1.0)),
        endpoint_a=(float(end_a[0]), float(end_a[1])),
        endpoint_b=(float(end_b[0]), float(end_b[1])),
    )


def measure_color(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    rgb = np.asarray(image)
    binary = np.asarray(mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[:2] != binary.shape or not np.any(binary):
        return {}
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    values = hsv[binary]
    return {
        "color_h": float(np.median(values[:, 0]) / 179.0),
        "color_s": float(np.median(values[:, 1]) / 255.0),
        "color_v": float(np.median(values[:, 2]) / 255.0),
    }


def part_attributes(
    mask: np.ndarray, image: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    measurement = measure_mask_shape(mask)
    attributes: Dict[str, Any] = {}
    if measurement is not None:
        attributes.update(measurement.to_dict())
    if image is not None:
        attributes.update(measure_color(image, mask))
    return attributes


__all__ = ["ShapeMeasurement", "measure_mask_shape", "measure_color", "part_attributes"]
