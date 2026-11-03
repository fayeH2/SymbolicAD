"""Query scene graph built from CLIP component grounding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ComponentInstance:
    instance_id: str
    concept: str
    confidence: float
    feature: np.ndarray          # [D]
    mask: np.ndarray             # [h, w] float in {0, 1}
    bbox: tuple                  # (x0, y0, x1, y1) normalised
    center: tuple                # (cx, cy) normalised
    area: float                  # fraction of the patch grid

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "concept": self.concept,
            "confidence": float(self.confidence),
            "bbox": [float(value) for value in self.bbox],
            "center": [float(value) for value in self.center],
            "area": float(self.area),
        }


@dataclass
class SceneGraph:
    component_maps: Dict[str, np.ndarray] = field(default_factory=dict)
    instances: Dict[str, List[ComponentInstance]] = field(default_factory=dict)
    patch_features: Optional[np.ndarray] = None
    pair_features: Dict[tuple, np.ndarray] = field(default_factory=dict)

    def instances_for(self, concept: str) -> List[ComponentInstance]:
        return self.instances.get(concept, [])

    def component_map(self, concept: str) -> Optional[np.ndarray]:
        return self.component_maps.get(concept)

    def grid_size(self) -> tuple[int, int]:
        if self.patch_features is not None:
            return int(self.patch_features.shape[1]), int(self.patch_features.shape[2])
        for value in self.component_maps.values():
            return int(value.shape[0]), int(value.shape[1])
        return 0, 0
