"""Deterministic connected-component instance extraction.

This is intentionally simple and debuggable.  It can later be replaced by
watershed, local peaks, SAM or learned object queries.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .scene_graph import ComponentInstance


def extract_instances(
    component_map: np.ndarray,
    patch_features: np.ndarray,
    concept: str,
    threshold: float = 0.55,
    min_area: int = 8,
    morphology_kernel: int = 3,
) -> List[ComponentInstance]:
    score = np.asarray(component_map, dtype=np.float32)
    if score.ndim != 2:
        raise ValueError(f"component_map must be 2D, got {score.shape}")
    feature = np.asarray(patch_features, dtype=np.float32)
    if feature.ndim != 3 or feature.shape[1:] != score.shape:
        raise ValueError(
            f"patch_features {feature.shape} do not match map {score.shape}"
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
            )
        )

    instances.sort(key=lambda item: (item.center[1], item.center[0]))
    for index, instance in enumerate(instances):
        instance.instance_id = f"{concept}_{index:02d}"
    return instances
