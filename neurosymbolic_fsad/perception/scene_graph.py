"""Scene graph construction for component spatial and topological relations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SpatialRelation:
    """Spatial relation between two components."""
    source_id: int
    target_id: int
    relation_type: str  # "left_of", "right_of", "above", "below", "near", "far"
    score: float  # confidence [0, 1]
    distance: float  # normalized distance
    angle: float  # angle in radians


@dataclass
class TopologicalRelation:
    """Topological relation between two components."""
    source_id: int
    target_id: int
    relation_type: str  # "connected", "touching", "contains", "inside", "disjoint"
    score: float
    intersection_area: float = 0.0


@dataclass
class SceneGraph:
    """Scene graph representing components and their relations."""
    components: List[Any]  # ComponentInstance objects
    spatial_relations: List[SpatialRelation] = field(default_factory=list)
    topological_relations: List[TopologicalRelation] = field(default_factory=list)
    component_types: Dict[int, str] = field(default_factory=dict)
    global_context: Dict[str, Any] = field(default_factory=dict)
    
    def get_component(self, idx: int) -> Optional[Any]:
        """Get component by index."""
        if 0 <= idx < len(self.components):
            return self.components[idx]
        return None
    
    def get_spatial_relations(self, source_id: int) -> List[SpatialRelation]:
        """Get all spatial relations from a source component."""
        return [r for r in self.spatial_relations if r.source_id == source_id]
    
    def get_topological_relations(self, source_id: int) -> List[TopologicalRelation]:
        """Get all topological relations from a source component."""
        return [r for r in self.topological_relations if r.source_id == source_id]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "num_components": len(self.components),
            "spatial_relations": [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "type": r.relation_type,
                    "score": r.score,
                    "distance": r.distance,
                    "angle": r.angle,
                }
                for r in self.spatial_relations
            ],
            "topological_relations": [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "type": r.relation_type,
                    "score": r.score,
                    "intersection": r.intersection_area,
                }
                for r in self.topological_relations
            ],
            "component_types": self.component_types,
            "global_context": self.global_context,
        }


def build_scene_graph(
    components: List[Any],
    component_types: Optional[Dict[int, str]] = None,
    spatial_threshold: float = 0.3,
    topological_threshold: float = 0.01,
) -> SceneGraph:
    """Build scene graph from component instances."""
    graph = SceneGraph(components=components)
    
    if component_types:
        graph.component_types = component_types
    
    n = len(components)
    if n == 0:
        return graph
    
    # Compute spatial relations
    for i in range(n):
        comp_i = components[i]
        cx_i, cy_i = comp_i.centroid
        
        for j in range(n):
            if i == j:
                continue
            comp_j = components[j]
            cx_j, cy_j = comp_j.centroid
            
            dx = cx_j - cx_i
            dy = cy_j - cy_i
            distance = math.sqrt(dx*dx + dy*dy)
            angle = math.atan2(dy, dx)  # -pi to pi
            
            if distance > spatial_threshold:
                continue
            
            # Determine relation type
            abs_dx, abs_dy = abs(dx), abs(dy)
            if abs_dx > abs_dy:
                rel_type = "right_of" if dx > 0 else "left_of"
            else:
                rel_type = "below" if dy > 0 else "above"
            
            # Near/far classification
            if distance < 0.1:
                rel_type = "near_" + rel_type
            elif distance > 0.2:
                rel_type = "far_" + rel_type
            
            # Confidence based on distance
            score = max(0.0, 1.0 - distance / spatial_threshold)
            
            graph.spatial_relations.append(SpatialRelation(
                source_id=i, target_id=j,
                relation_type=rel_type, score=score,
                distance=distance, angle=angle,
            ))
    
    # Compute topological relations
    for i in range(n):
        mask_i = components[i].mask
        for j in range(n):
            if i == j:
                continue
            mask_j = components[j].mask
            
            intersection = np.logical_and(mask_i, mask_j).sum()
            union = np.logical_or(mask_i, mask_j).sum()
            iou = intersection / union if union > 0 else 0.0
            
            if iou > topological_threshold:
                # Check containment
                area_i = mask_i.sum()
                area_j = mask_j.sum()
                
                if intersection / area_i > 0.9:
                    rel_type = "inside"
                elif intersection / area_j > 0.9:
                    rel_type = "contains"
                elif iou > 0.1:
                    rel_type = "touching"
                else:
                    rel_type = "connected"
                
                graph.topological_relations.append(TopologicalRelation(
                    source_id=i, target_id=j,
                    relation_type=rel_type, score=float(iou),
                    intersection_area=float(intersection),
                ))
    
    # Global context
    graph.global_context = {
        "num_components": n,
        "component_types": graph.component_types,
        "image_center": (0.5, 0.5),
    }
    
    return graph


def align_scene_graphs(
    support_graph: SceneGraph,
    query_graph: SceneGraph,
    method: str = "hungarian",
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Align components between support and query scene graphs.
    
    Returns:
        List of (support_idx, query_idx) matches
        Cost matrix
    """
    from scipy.optimize import linear_sum_assignment
    
    n_support = len(support_graph.components)
    n_query = len(query_graph.components)
    
    if n_support == 0 or n_query == 0:
        return [], np.array([])
    
    # Build cost matrix based on feature similarity
    cost_matrix = np.full((n_support, n_query), 10.0)
    
    for i in range(n_support):
        comp_i = support_graph.components[i]
        feat_i = comp_i.features
        type_i = support_graph.component_types.get(i, "")
        
        for j in range(n_query):
            comp_j = query_graph.components[j]
            feat_j = comp_j.features
            type_j = query_graph.component_types.get(j, "")
            
            # Type matching
            type_cost = 0.0 if type_i == type_j else 1.0
            
            # Feature similarity
            feature_cost = 0.0
            if feat_i and feat_j:
                common_keys = set(feat_i.keys()) & set(feat_j.keys())
                if common_keys:
                    diffs = []
                    for k in common_keys:
                        v1, v2 = feat_i[k], feat_j[k]
                        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                            diffs.append(abs(v1 - v2) / max(abs(v1), abs(v2), 1e-6))
                    if diffs:
                        feature_cost = np.mean(diffs)
            
            # Spatial consistency
            spatial_cost = 0.0
            cx_i, cy_i = comp_i.centroid
            cx_j, cy_j = comp_j.centroid
            spatial_cost = math.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
            
            cost_matrix[i, j] = type_cost + feature_cost + 0.5 * spatial_cost
    
    if method == "hungarian":
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = list(zip(row_ind.tolist(), col_ind.tolist()))
        return matches, cost_matrix
    
    # Greedy matching
    matches = []
    used_q = set()
    for i in range(n_support):
        best_j = -1
        best_cost = float('inf')
        for j in range(n_query):
            if j in used_q:
                continue
            if cost_matrix[i, j] < best_cost:
                best_cost = cost_matrix[i, j]
                best_j = j
        if best_j >= 0:
            matches.append((i, best_j))
            used_q.add(best_j)
    
    return matches, cost_matrix


def compute_normal_slots(
    support_graphs: List[SceneGraph],
    matches_list: List[List[Tuple[int, int]]],
) -> Dict[int, Dict[str, Any]]:
    """Compute normal spatial slots from aligned support graphs.
    
    Args:
        support_graphs: List of scene graphs from normal supports
        matches_list: List of matches where each entry is (ref_idx, other_idx) pairs
                      aligning each graph to the first (reference) graph.
                      First entry should be identity matches for reference graph.
    
    Returns:
        Dict mapping slot_id (reference graph component index) to slot properties.
    """
    from collections import defaultdict
    
    if not support_graphs:
        return {}
    
    ref_graph = support_graphs[0]
    n_ref = len(ref_graph.components)
    
    # slot_components[slot_id] = list of (graph_idx, comp_idx_in_that_graph)
    slot_components = defaultdict(list)
    
    # First graph: identity mapping
    for i in range(n_ref):
        slot_components[i].append((0, i))
    
    # Other graphs: matches are (ref_idx, other_idx)
    for graph_idx in range(1, len(support_graphs)):
        if graph_idx - 1 < len(matches_list):
            matches = matches_list[graph_idx - 1]
            for ref_idx, other_idx in matches:
                if ref_idx < n_ref:
                    slot_components[ref_idx].append((graph_idx, other_idx))
    
    slots = {}
    for slot_id, occurrences in slot_components.items():
        if slot_id >= n_ref:
            continue
            
        # Reference component
        ref_comp = ref_graph.components[slot_id]
        
        positions = []
        bboxes = []
        areas = []
        features_list = []
        
        for g_idx, c_idx in occurrences:
            comp = support_graphs[g_idx].components[c_idx]
            positions.append(comp.centroid)
            bboxes.append(comp.bbox)
            areas.append(comp.area)
            features_list.append(comp.features)
        
        slots[slot_id] = {
            "slot_id": slot_id,
            "mean_position": np.mean(positions, axis=0).tolist(),
            "std_position": np.std(positions, axis=0).tolist(),
            "mean_bbox": np.mean(bboxes, axis=0).tolist(),
            "mean_area": float(np.mean(areas)),
            "std_area": float(np.std(areas)),
            "count": len(occurrences),
            "reference_type": ref_graph.component_types.get(slot_id, ""),
            "reference_features": features_list[0] if features_list else {},
        }
    
    return slots
    
    return slots