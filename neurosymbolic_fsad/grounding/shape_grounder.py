"""Shape-specific grounding that preserves mask and endpoint geometry."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .counterfactual_repair import RepairMap, length_endpoint_repair


def resized_instance_mask(instance, map_size: int = 256) -> np.ndarray:
    mask = np.asarray(instance.mask, dtype=np.uint8)
    return cv2.resize(mask, (map_size, map_size), interpolation=cv2.INTER_NEAREST).astype(
        np.float32
    )


def union_instance_masks(instances: Iterable, map_size: int = 256) -> np.ndarray:
    result = np.zeros((map_size, map_size), dtype=np.float32)
    for instance in instances:
        result = np.maximum(result, resized_instance_mask(instance, map_size))
    return result


def length_violation_map(
    instance, target_length: float, map_size: int = 256,
    reference_endpoints=None,
) -> RepairMap:
    return length_endpoint_repair(
        instance, target_length, map_size=map_size,
        reference_endpoints=reference_endpoints,
    )


__all__ = ["resized_instance_mask", "union_instance_masks", "length_violation_map"]
