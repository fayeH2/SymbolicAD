"""Semantic prototype bank for component type matching."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import json


@dataclass
class PartPrototype:
    """Semantic prototype for a part type."""
    part_type: str
    dino_features: List[np.ndarray]  # List of feature vectors [C]
    color_stats: Dict[str, float]  # mean_h, mean_s, mean_v, std_h, std_s
    shape_stats: Dict[str, float]  # aspect, circularity, solidity
    area_range: Tuple[float, float]  # min, max normalized area
    position_stats: Dict[str, float]  # mean_cx, mean_cy, std_cx, std_cy
    count: int
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def match_score(
        self,
        dino_feat: Optional[np.ndarray] = None,
        color: Optional[Dict[str, float]] = None,
        shape: Optional[Dict[str, float]] = None,
        area: Optional[float] = None,
        position: Optional[Tuple[float, float]] = None,
    ) -> float:
        """Compute matching score for a component instance."""
        scores = []
        
        # DINO feature similarity
        if dino_feat is not None and self.dino_features:
            dino_feat = dino_feat / (np.linalg.norm(dino_feat) + 1e-8)
            sims = []
            for proto_feat in self.dino_features:
                proto_feat = proto_feat / (np.linalg.norm(proto_feat) + 1e-8)
                sims.append(float(np.dot(dino_feat, proto_feat)))
            scores.append(max(sims))
        
        # Color similarity
        if color and self.color_stats:
            h_diff = min(abs(color.get("color_h", 0) - self.color_stats.get("mean_h", 0)), 180) / 180.0
            s_diff = abs(color.get("color_s", 0) - self.color_stats.get("mean_s", 0)) / 255.0
            v_diff = abs(color.get("color_v", 0) - self.color_stats.get("mean_v", 0)) / 255.0
            color_score = 1.0 - (h_diff + s_diff + v_diff) / 3.0
            scores.append(max(0.0, color_score))
        
        # Shape similarity
        if shape and self.shape_stats:
            aspect_diff = abs(shape.get("shape_aspect", 1) - self.shape_stats.get("aspect", 1))
            circ_diff = abs(shape.get("shape_circularity", 0) - self.shape_stats.get("circularity", 0))
            solid_diff = abs(shape.get("shape_solidity", 0) - self.shape_stats.get("solidity", 0))
            shape_score = 1.0 - (aspect_diff + circ_diff + solid_diff) / 3.0
            scores.append(max(0.0, shape_score))
        
        # Area similarity
        if area is not None:
            min_a, max_a = self.area_range
            if min_a <= area <= max_a:
                scores.append(1.0)
            else:
                dist = min(abs(area - min_a), abs(area - max_a))
                scores.append(max(0.0, 1.0 - dist / max(max_a, 1e-6)))
        
        # Position similarity
        if position is not None:
            cx, cy = position
            mean_cx = self.position_stats.get("mean_cx", 0.5)
            mean_cy = self.position_stats.get("mean_cy", 0.5)
            std_cx = self.position_stats.get("std_cx", 0.1)
            std_cy = self.position_stats.get("std_cy", 0.1)
            pos_dist = np.sqrt(((cx - mean_cx) / max(std_cx, 0.01))**2 + 
                              ((cy - mean_cy) / max(std_cy, 0.01))**2)
            pos_score = max(0.0, 1.0 - pos_dist / 3.0)  # 3-sigma
            scores.append(pos_score)
        
        return float(np.mean(scores)) if scores else 0.0


class SemanticPrototypeBank:
    """Bank of semantic part prototypes built from support samples."""
    
    def __init__(self):
        self.prototypes: Dict[str, PartPrototype] = {}
        self.part_type_names: List[str] = []
    
    def add_prototype(self, prototype: PartPrototype) -> None:
        """Add or update a prototype."""
        self.prototypes[prototype.part_type] = prototype
        if prototype.part_type not in self.part_type_names:
            self.part_type_names.append(prototype.part_type)
    
    def build_from_support(
        self,
        support_graphs: List["SceneGraph"],
        matches_list: List[List[Tuple[int, int]]],
    ) -> None:
        """Build prototypes from aligned support scene graphs."""
        from collections import defaultdict
        from symbolicAD.perception.scene_graph import compute_normal_slots
        
        slots = compute_normal_slots(support_graphs, matches_list)
        
        # Group slots by part type
        type_slots = defaultdict(list)
        for slot_id, slot_info in slots.items():
            part_type = slot_info.get("reference_type", f"part_{slot_id}")
            type_slots[part_type].append(slot_info)
        
        for part_type, slot_infos in type_slots.items():
            # Aggregate features
            all_dino = []
            all_color = []
            all_shape = []
            all_areas = []
            all_positions = []
            
            for slot in slot_infos:
                if "reference_features" in slot:
                    ref_feat = slot["reference_features"]
                    if "dino_features" in ref_feat:
                        all_dino.append(ref_feat["dino_features"])
                    all_color.append({
                        "color_h": ref_feat.get("color_h", 0),
                        "color_s": ref_feat.get("color_s", 0),
                        "color_v": ref_feat.get("color_v", 0),
                    })
                    all_shape.append({
                        "shape_aspect": ref_feat.get("shape_aspect", 1),
                        "shape_circularity": ref_feat.get("shape_circularity", 0),
                        "shape_solidity": ref_feat.get("shape_solidity", 0),
                    })
                all_areas.append(slot["mean_area"])
                all_positions.append(slot["mean_position"])
            
            # Compute statistics
            color_stats = {}
            if all_color:
                color_stats = {
                    "mean_h": float(np.mean([c["color_h"] for c in all_color])),
                    "mean_s": float(np.mean([c["color_s"] for c in all_color])),
                    "mean_v": float(np.mean([c["color_v"] for c in all_color])),
                    "std_h": float(np.std([c["color_h"] for c in all_color])),
                    "std_s": float(np.std([c["color_s"] for c in all_color])),
                }
            
            shape_stats = {}
            if all_shape:
                shape_stats = {
                    "aspect": float(np.mean([s["shape_aspect"] for s in all_shape])),
                    "circularity": float(np.mean([s["shape_circularity"] for s in all_shape])),
                    "solidity": float(np.mean([s["shape_solidity"] for s in all_shape])),
                }
            
            area_range = (float(np.min(all_areas)), float(np.max(all_areas))) if all_areas else (0.0, 1.0)
            
            position_stats = {}
            if all_positions:
                positions = np.array(all_positions)
                position_stats = {
                    "mean_cx": float(np.mean(positions[:, 0])),
                    "mean_cy": float(np.mean(positions[:, 1])),
                    "std_cx": float(np.std(positions[:, 0])),
                    "std_cy": float(np.std(positions[:, 1])),
                }
            
            prototype = PartPrototype(
                part_type=part_type,
                dino_features=all_dino,
                color_stats=color_stats,
                shape_stats=shape_stats,
                area_range=area_range,
                position_stats=position_stats,
                count=len(slot_infos),
                examples=slot_infos,
            )
            self.add_prototype(prototype)
    
    def match_component(
        self,
        component: Any,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Match a component to the best prototypes."""
        scores = []
        for part_type, proto in self.prototypes.items():
            score = proto.match_score(
                dino_feat=component.dino_features,
                color=component.features,
                shape=component.features,
                area=component.area,
                position=component.centroid,
            )
            scores.append((part_type, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def save(self, path: Union[str, Path]) -> None:
        """Save prototype bank to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "part_types": self.part_type_names,
            "prototypes": {},
        }
        for name, proto in self.prototypes.items():
            data["prototypes"][name] = {
                "part_type": proto.part_type,
                "color_stats": proto.color_stats,
                "shape_stats": proto.shape_stats,
                "area_range": proto.area_range,
                "position_stats": proto.position_stats,
                "count": proto.count,
                "num_dino_features": len(proto.dino_features),
            }
        
        path.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "SemanticPrototypeBank":
        """Load prototype bank from JSON (without DINO features)."""
        path = Path(path)
        data = json.loads(path.read_text())
        
        bank = cls()
        for name, proto_data in data["prototypes"].items():
            prototype = PartPrototype(
                part_type=proto_data["part_type"],
                dino_features=[],  # Features not saved
                color_stats=proto_data["color_stats"],
                shape_stats=proto_data["shape_stats"],
                area_range=tuple(proto_data["area_range"]),
                position_stats=proto_data["position_stats"],
                count=proto_data["count"],
            )
            bank.add_prototype(prototype)
        
        return bank


def match_components_to_prototypes(
    query_components: List[Any],
    prototype_bank: SemanticPrototypeBank,
    threshold: float = 0.5,
) -> Dict[int, str]:
    """Match query components to prototype types."""
    matches = {}
    for i, comp in enumerate(query_components):
        results = prototype_bank.match_component(comp, top_k=1)
        if results and results[0][1] >= threshold:
            matches[i] = results[0][0]
        else:
            matches[i] = "unknown"
    return matches