"""Component attribute extraction and representation."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from symbolicAD.perception.segmentation import (
    ComponentInstance as LegacyComponentInstance,
    extract_attributes as _legacy_extract_attributes,
)


@dataclass
class ComponentAttributes:
    """Structured attributes for a component."""
    # Color attributes (HSV)
    color_h: float = 0.0
    color_s: float = 0.0
    color_v: float = 0.0
    color_h_std: float = 0.0
    color_s_std: float = 0.0
    
    # Shape attributes
    shape_area: float = 0.0
    shape_perimeter: float = 0.0
    shape_aspect: float = 1.0
    shape_circularity: float = 0.0
    shape_solidity: float = 0.0
    
    # BBox attributes
    bbox_width: float = 0.0
    bbox_height: float = 0.0
    
    # DINO features
    dino_features: Optional[np.ndarray] = None
    
    # Additional custom attributes
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_instance(cls, instance: Any) -> "ComponentAttributes":
        """Create from ComponentInstance."""
        feat = instance.features
        return cls(
            color_h=feat.get("color_h", 0.0),
            color_s=feat.get("color_s", 0.0),
            color_v=feat.get("color_v", 0.0),
            color_h_std=feat.get("color_h_std", 0.0),
            color_s_std=feat.get("color_s_std", 0.0),
            shape_area=feat.get("shape_area", 0.0),
            shape_perimeter=feat.get("shape_perimeter", 0.0),
            shape_aspect=feat.get("shape_aspect", 1.0),
            shape_circularity=feat.get("shape_circularity", 0.0),
            shape_solidity=feat.get("shape_solidity", 0.0),
            bbox_width=feat.get("bbox_width", 0.0),
            bbox_height=feat.get("bbox_height", 0.0),
            dino_features=getattr(
                instance, "dino_features", getattr(instance, "embedding", None)
            ),
            extra={k: v for k, v in feat.items() if k not in {
                "color_h", "color_s", "color_v", "color_h_std", "color_s_std",
                "shape_area", "shape_perimeter", "shape_aspect", "shape_circularity",
                "shape_solidity", "bbox_width", "bbox_height"
            }},
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "color_h": self.color_h, "color_s": self.color_s, "color_v": self.color_v,
            "color_h_std": self.color_h_std, "color_s_std": self.color_s_std,
            "shape_area": self.shape_area, "shape_perimeter": self.shape_perimeter,
            "shape_aspect": self.shape_aspect, "shape_circularity": self.shape_circularity,
            "shape_solidity": self.shape_solidity,
            "bbox_width": self.bbox_width, "bbox_height": self.bbox_height,
        }
        d.update(self.extra)
        return d
    
    def color_distance(self, other: "ComponentAttributes") -> float:
        """Compute color distance to another component."""
        h_diff = min(abs(self.color_h - other.color_h), 180) / 180.0
        s_diff = abs(self.color_s - other.color_s) / 255.0
        v_diff = abs(self.color_v - other.color_v) / 255.0
        return (h_diff + s_diff + v_diff) / 3.0
    
    def shape_distance(self, other: "ComponentAttributes") -> float:
        """Compute shape distance to another component."""
        aspect_diff = abs(self.shape_aspect - other.shape_aspect)
        circ_diff = abs(self.shape_circularity - other.shape_circularity)
        solid_diff = abs(self.shape_solidity - other.shape_solidity)
        return (aspect_diff + circ_diff + solid_diff) / 3.0


def extract_attributes(image: np.ndarray, instance: Any) -> Dict[str, Any]:
    """Compatibility adapter without the former self-recursion bug."""
    if hasattr(instance, "attributes"):
        from .part_attributes import part_attributes
        return part_attributes(np.asarray(instance.mask), image)
    return _legacy_extract_attributes(image, instance)


def compute_attribute_statistics(
    instances: List[Any],
) -> Dict[str, Dict[str, float]]:
    """Compute statistics of attributes across instances."""
    if not instances:
        return {}
    
    keys = set()
    for inst in instances:
        keys.update(inst.features.keys())
    
    stats = {}
    for key in keys:
        values = [inst.features.get(key, 0.0) for inst in instances]
        values = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
        if values:
            stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }
    return stats
