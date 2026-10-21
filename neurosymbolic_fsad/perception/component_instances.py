"""Canonical component instance representation for all perception backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

from .part_attributes import part_attributes


@dataclass
class ComponentInstance:
    instance_id: str
    entity_type: str
    mask: np.ndarray
    bbox: tuple[float, float, float, float]
    centroid: tuple[float, float]
    confidence: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    source: str = "unknown"

    @classmethod
    def from_mask(
        cls,
        instance_id: str,
        entity_type: str,
        mask: np.ndarray,
        *,
        image: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        source: str = "instance_mask",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "ComponentInstance":
        binary = np.asarray(mask, dtype=bool)
        if binary.ndim != 2 or not np.any(binary):
            raise ValueError("component mask must be a non-empty 2D array")
        height, width = binary.shape
        ys, xs = np.nonzero(binary)
        bbox = (
            float(xs.min() / width), float(ys.min() / height),
            float((xs.max() + 1) / width), float((ys.max() + 1) / height),
        )
        measured = part_attributes(binary, image)
        measured.update(attributes or {})
        return cls(
            instance_id=str(instance_id),
            entity_type=str(entity_type),
            mask=binary,
            bbox=bbox,
            centroid=(float((xs.mean() + 0.5) / width), float((ys.mean() + 0.5) / height)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            attributes=measured,
            source=source,
        )

    @property
    def features(self) -> Dict[str, Any]:
        """Compatibility alias for the historical scene-graph code."""
        return self.attributes

    @property
    def area(self) -> float:
        return float(self.attributes.get("area", np.mean(self.mask)))


def instances_from_label_mask(
    label_mask: np.ndarray,
    *,
    entity_type: str,
    image: Optional[np.ndarray] = None,
    source: str = "manual_instance_mask",
    confidence: float = 1.0,
) -> List[ComponentInstance]:
    labels = np.asarray(label_mask)
    if labels.ndim != 2:
        raise ValueError("label_mask must be two-dimensional")
    instances = []
    for label in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
        instances.append(ComponentInstance.from_mask(
            f"{entity_type}_{label}", entity_type, labels == label,
            image=image, confidence=confidence, source=source,
        ))
    return instances


def instances_from_binary_mask(
    binary_mask: np.ndarray,
    *,
    entity_type: str,
    image: Optional[np.ndarray] = None,
    min_area: int = 4,
    source: str = "connected_components",
    confidence: float = 0.75,
) -> List[ComponentInstance]:
    binary = np.asarray(binary_mask, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    instances = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < int(min_area):
            continue
        instances.append(ComponentInstance.from_mask(
            f"{entity_type}_{len(instances)}", entity_type, labels == label,
            image=image, confidence=confidence, source=source,
        ))
    return instances


def instance_ids(instances: Iterable[ComponentInstance]) -> List[str]:
    return [instance.instance_id for instance in instances]


__all__ = [
    "ComponentInstance", "instances_from_label_mask",
    "instances_from_binary_mask", "instance_ids",
]
