"""Deterministic connected-component instance extraction.

This is intentionally simple and debuggable.  It can later be replaced by
watershed, local peaks, SAM or learned object queries.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .scene_graph import ComponentInstance


_NAMED_RGB = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (210, 45, 45),
    "brown": (125, 70, 45),
    "orange": (235, 135, 35),
    "yellow": (235, 220, 70),
    "green": (60, 155, 75),
    "blue": (55, 105, 210),
    "purple": (135, 70, 170),
    "pink": (230, 145, 170),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "clear": (220, 220, 220),
    "transparent": (220, 220, 220),
}


def image_to_lab_grid(image, grid_size: tuple[int, int]) -> np.ndarray:
    """Resize the complete RGB image to the patch grid and convert to Lab."""
    if hasattr(image, "convert"):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        rgb = np.clip(rgb[..., :3], 0, 255).astype(np.uint8)
    height, width = int(grid_size[0]), int(grid_size[1])
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return (
        cv2.cvtColor(resized, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    )


def component_spatial_prior(
    image,
    grid_size: tuple[int, int],
    position: str = "",
    role: str = "",
) -> np.ndarray:
    """Return a foreground-aware coarse prior on the CLIP patch grid.

    These industrial images use an almost black background.  Without this
    prior, patch/text similarities frequently form confident components in the
    background, making downstream position and colour rules meaningless.  The
    mask stays coarse; support images still determine exact expected geometry.
    """
    if hasattr(image, "convert"):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image, dtype=np.uint8)[..., :3]
    height, width = int(grid_size[0]), int(grid_size[1])
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    lightness = (
        cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
        / 255.0
    )
    foreground = (lightness > 0.03).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if count > 1:
        yy, xx = np.mgrid[:height, :width]
        scores = []
        for index in range(1, count):
            mask = labels == index
            area = float(stats[index, cv2.CC_STAT_AREA])
            cx = float(xx[mask].mean()) if mask.any() else 0.0
            centre_penalty = abs(cx / max(width - 1, 1) - 0.5)
            scores.append((area * (1.0 - 0.5 * centre_penalty), index))
        foreground = (labels == max(scores)[1]).astype(np.uint8)
    points = np.argwhere(foreground > 0)
    if len(points) >= 3:
        hull_xy = cv2.convexHull(points[:, ::-1].astype(np.int32))
        silhouette = np.zeros_like(foreground)
        cv2.fillConvexPoly(silhouette, hull_xy, 1)
    else:
        silhouette = foreground

    role_lower = str(role).lower()
    if any(token in role_lower for token in ("container", "bottle", "housing")):
        return silhouette.astype(np.float32)

    y, x = np.mgrid[:height, :width]
    yn = (y + 0.5) / max(height, 1)
    xn = (x + 0.5) / max(width, 1)
    region = np.ones((height, width), dtype=bool)
    key = str(position).strip().lower()
    if key == "top":
        region &= yn <= 0.38
    elif key == "upper_center":
        region &= (yn >= 0.28) & (yn <= 0.68) & (xn >= 0.18) & (xn <= 0.82)
    elif key == "center":
        region &= (yn >= 0.12) & (yn <= 0.90) & (xn >= 0.18) & (xn <= 0.82)
    elif key == "lower_center":
        region &= (yn >= 0.58) & (yn <= 0.94) & (xn >= 0.18) & (xn <= 0.82)
    elif key == "bottom":
        region &= yn >= 0.65
    elif key == "left":
        region &= xn <= 0.55
    elif key == "right":
        region &= xn >= 0.45
    return (silhouette.astype(bool) & region).astype(np.float32)


def named_color_lab(value: str) -> np.ndarray | None:
    """Return a canonical Lab target for common Qwen colour words."""
    lowered = str(value).strip().lower()
    rgb = next(
        (triplet for name, triplet in _NAMED_RGB.items() if name in lowered),
        None,
    )
    if rgb is None:
        return None
    pixel = np.asarray([[rgb]], dtype=np.uint8)
    return cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32) / 255.0


def lab_color_similarity(
    feature: np.ndarray,
    target: np.ndarray,
    sigma: float = 0.18,
) -> float:
    distance = float(np.linalg.norm(np.asarray(feature) - np.asarray(target)))
    sigma = max(float(sigma), 1e-6)
    return float(np.exp(-(distance**2) / (2.0 * sigma**2)))


def extract_instances(
    component_map: np.ndarray,
    patch_features: np.ndarray,
    concept: str,
    threshold: float = 0.55,
    min_area: int = 8,
    morphology_kernel: int = 3,
    color_grid: np.ndarray | None = None,
) -> List[ComponentInstance]:
    score = np.asarray(component_map, dtype=np.float32)
    if score.ndim != 2:
        raise ValueError(f"component_map must be 2D, got {score.shape}")
    feature = np.asarray(patch_features, dtype=np.float32)
    if feature.ndim != 3 or feature.shape[1:] != score.shape:
        raise ValueError(
            f"patch_features {feature.shape} do not match map {score.shape}"
        )
    if color_grid is not None:
        color_grid = np.asarray(color_grid, dtype=np.float32)
        if color_grid.shape != (*score.shape, 3):
            raise ValueError(
                f"color_grid {color_grid.shape} does not match map {score.shape}"
            )

    binary = (score >= float(threshold)).astype(np.uint8)
    if morphology_kernel > 1:
        kernel = np.ones(
            (morphology_kernel, morphology_kernel), dtype=np.uint8
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    height, width = score.shape
    total = float(height * width)
    instances: List[ComponentInstance] = []

    for label_index in range(1, count):
        area_pixels = int(stats[label_index, cv2.CC_STAT_AREA])
        if area_pixels < int(min_area):
            continue
        mask = (labels == label_index).astype(np.float32)
        weights = mask * np.clip(score, 0.0, 1.0)
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-8:
            weights = mask
            weight_sum = float(mask.sum())

        pooled = np.einsum("dhw,hw->d", feature, weights) / weight_sum
        norm = float(np.linalg.norm(pooled))
        if norm > 1e-8:
            pooled = pooled / norm

        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        w = int(stats[label_index, cv2.CC_STAT_WIDTH])
        h = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        bbox = (
            x / max(width - 1, 1),
            y / max(height - 1, 1),
            (x + w - 1) / max(width - 1, 1),
            (y + h - 1) / max(height - 1, 1),
        )
        cx, cy = centroids[label_index]
        center = (float(cx / max(width - 1, 1)), float(cy / max(height - 1, 1)))
        confidence = float(score[mask > 0].mean())
        color_feature = (
            np.einsum("hwc,hw->c", color_grid, weights) / weight_sum
            if color_grid is not None
            else np.zeros(3, dtype=np.float32)
        )

        instances.append(
            ComponentInstance(
                instance_id=f"{concept}_{len(instances):02d}",
                concept=concept,
                confidence=confidence,
                feature=pooled.astype(np.float32),
                mask=mask,
                bbox=bbox,
                center=center,
                area=float(area_pixels / total),
                color_feature=np.asarray(color_feature, dtype=np.float32),
            )
        )

    instances.sort(key=lambda item: (item.center[1], item.center[0]))
    for index, instance in enumerate(instances):
        instance.instance_id = f"{concept}_{index:02d}"
    return instances


def select_instances(
    instances: List[ComponentInstance],
    max_instances: Optional[int],
) -> List[ComponentInstance]:
    """Keep the strongest coherent instances when a rule declares a count."""
    if max_instances is None or max_instances <= 0 or len(instances) <= max_instances:
        return instances
    ranked = sorted(
        instances,
        key=lambda item: item.confidence * np.sqrt(max(item.area, 1e-8)),
        reverse=True,
    )
    return ranked[: int(max_instances)]
