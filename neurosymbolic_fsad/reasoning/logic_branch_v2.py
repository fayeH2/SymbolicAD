"""Enhanced Logic Branch v2: Works with fallback segmenter output."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2

from ..rules.generation.dsl import NormalityRule, RuleSet, RuleType, AttributeType
from ..perception.scene_graph import SceneGraph, compute_normal_slots
from ..perception.segmentation import ComponentInstance


@dataclass
class MainObject:
    """Main object detected in scene (largest component)."""
    component: ComponentInstance
    index: int
    relative_size: float  # ratio to image
    position: Tuple[float, float]  # normalized centroid


def extract_main_object(graph: SceneGraph) -> Optional[MainObject]:
    """Extract the main object (largest component) from scene graph."""
    if not graph.components:
        return None
    
    # Find largest component by area
    largest_idx = max(range(len(graph.components)), key=lambda i: graph.components[i].area)
    comp = graph.components[largest_idx]
    
    return MainObject(
        component=comp,
        index=largest_idx,
        relative_size=comp.area,
        position=comp.centroid,
    )


def extract_scene_statistics(graph: SceneGraph) -> Dict[str, float]:
    """Extract global scene statistics that are stable across normal samples."""
    comps = graph.components
    if not comps:
        return {}
    
    areas = [c.area for c in comps]
    positions = [c.centroid for c in comps]
    colors_h = [c.features.get('color_h', 0) for c in comps]
    colors_s = [c.features.get('color_s', 0) for c in comps]
    colors_v = [c.features.get('color_v', 0) for c in comps]
    aspects = [c.features.get('shape_aspect', 1) for c in comps]
    
    stats = {
        'num_components': float(len(comps)),
        'max_area': float(np.max(areas)),
        'mean_area': float(np.mean(areas)),
        'std_area': float(np.std(areas)),
        'total_area': float(np.sum(areas)),
        
        'main_obj_area': float(np.max(areas)),
        'main_obj_rel_size': float(np.max(areas) / np.sum(areas)) if np.sum(areas) > 0 else 0,
        'main_obj_cx': float(positions[np.argmax(areas)][0]),
        'main_obj_cy': float(positions[np.argmax(areas)][1]),
        
        'mean_cx': float(np.mean([p[0] for p in positions])),
        'mean_cy': float(np.mean([p[1] for p in positions])),
        'std_cx': float(np.std([p[0] for p in positions])),
        'std_cy': float(np.std([p[1] for p in positions])),
        
        'mean_color_h': float(np.mean(colors_h)) if colors_h else 0,
        'std_color_h': float(np.std(colors_h)) if colors_h else 0,
        'mean_color_s': float(np.mean(colors_s)) if colors_s else 0,
        'mean_color_v': float(np.mean(colors_v)) if colors_v else 0,
        
        'mean_aspect': float(np.mean(aspects)) if aspects else 1,
        'std_aspect': float(np.std(aspects)) if aspects else 0,
        
        'left_half_count': float(sum(1 for p in positions if p[0] < 0.5)),
        'right_half_count': float(sum(1 for p in positions if p[0] >= 0.5)),
        'upper_half_count': float(sum(1 for p in positions if p[1] < 0.5)),
        'lower_half_count': float(sum(1 for p in positions if p[1] >= 0.5)),
        
        'lr_balance': float(sum(1 for p in positions if p[0] < 0.5) / max(sum(1 for p in positions if p[0] >= 0.5), 1)),
        'vertical_spread': float(np.std([p[1] for p in positions])),
    }
    return stats


class LogicBranchV2:
    """Logic branch that works with global scene statistics."""
    
    def __init__(
        self,
        rule_set: RuleSet,
        normal_slots: Dict[int, Dict[str, Any]],
        normal_stats: Dict[str, Dict[str, float]],  # mean/std per stat
        map_size: int = 256,
    ):
        self.rule_set = rule_set
        self.normal_slots = normal_slots
        self.normal_stats = normal_stats  # {stat_name: {'mean':, 'std':, 'min':, 'max':}}
        self.map_size = map_size
    
    def evaluate(self, graph: SceneGraph) -> Tuple[float, Dict[str, float], Dict[str, np.ndarray]]:
        """Evaluate all rules, return (overall_score, rule_scores, spatial_maps)."""
        # Extract features
        main_obj = extract_main_object(graph)
        scene_stats = extract_scene_statistics(graph)
        
        rule_scores = {}
        spatial_maps = {}
        violations = []
        
        for rule in self.rule_set.rules:
            if rule.rule_type == RuleType.COUNT:
                score, evidence = self._eval_count_rule(rule, graph, scene_stats, main_obj)
            elif rule.rule_type == RuleType.ATTRIBUTE:
                score, evidence = self._eval_attribute_rule(rule, graph, scene_stats, main_obj)
            elif rule.rule_type == RuleType.SPATIAL:
                score, evidence = self._eval_spatial_rule(rule, graph, scene_stats, main_obj)
            else:
                score, evidence = 0.0, None
            
            rule_scores[rule.rule_id] = 1.0 - score
            if score > 1e-6:
                violations.append((rule.rule_id, score))
                if evidence is not None:
                    spatial_maps[rule.rule_id] = evidence
        
        # Overall score
        if violations:
            weighted = sum(s * self._get_rule_confidence(r) for r, s in violations)
            total_w = sum(self._get_rule_confidence(r) for r, _ in violations)
            overall = weighted / max(total_w, 1e-6)
        else:
            overall = 0.0
        
        return overall, rule_scores, spatial_maps
    
    def _get_rule_confidence(self, rule_id: str) -> float:
        for r in self.rule_set.rules:
            if r.rule_id == rule_id:
                return r.confidence
        return 1.0
    
    def _eval_count_rule(
        self, rule: NormalityRule, graph: SceneGraph, 
        scene_stats: Dict, main_obj: Optional[MainObject]
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Evaluate count rule with proper target."""
        target = rule.part_type
        expected = rule.expected_count or 0
        tol = rule.count_tolerance
        
        if target is None:
            # Global count
            actual = int(scene_stats.get('num_components', 0))
        elif target == 'main':
            actual = 1 if main_obj else 0
        elif target == 'large':
            actual = int(scene_stats.get('num_components', 0))  # Would need threshold
        else:
            actual = 0
        
        diff = abs(actual - expected)
        violation = 0.0 if diff <= tol else min(1.0, (diff - tol) / max(expected, 1))
        
        # Witness map: highlight extra/missing components
        witness = self._make_count_witness(graph, actual, expected, target)
        return violation, witness
    
    def _eval_attribute_rule(
        self, rule: NormalityRule, graph: SceneGraph,
        scene_stats: Dict, main_obj: Optional[MainObject]
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Evaluate attribute rule using scene statistics."""
        attr = rule.attribute
        if attr is None:
            return 0.0, None
        
        key = attr.value
        actual = scene_stats.get(key)
        if actual is None:
            return 0.0, None
        
        expected = rule.expected_value or 0
        tol = rule.tolerance or 0.1
        
        if rule.direction == "stable":
            diff = abs(actual - expected)
            violation = max(0.0, (diff - tol) / max(abs(expected), 1e-6))
        elif rule.direction == "low":
            violation = max(0.0, (actual - expected - tol) / max(abs(expected), 1e-6))
        elif rule.direction == "high":
            violation = max(0.0, (expected - actual - tol) / max(abs(expected), 1e-6))
        else:
            violation = 0.0
        
        violation = min(1.0, violation)
        
        # Witness: highlight main object if its attribute violates
        witness = None
        if main_obj and violation > 0.1:
            witness = self._make_component_witness(main_obj.component, violation)
        
        return violation, witness
    
    def _eval_spatial_rule(
        self, rule: NormalityRule, graph: SceneGraph,
        scene_stats: Dict, main_obj: Optional[MainObject]
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Evaluate spatial rule: main object position vs normal."""
        if not rule.spatial_relation or not main_obj:
            return 0.0, None
        
        # Use normal_slots to get expected position
        cx, cy = main_obj.position
        
        # Find closest normal slot for main object type
        expected_pos = None
        for slot_id, slot in self.normal_slots.items():
            if slot.get('reference_type', '') in ['', 'main', 'primary']:
                expected_pos = slot.get('mean_position')
                break
        
        if expected_pos is None:
            # Use scene statistics mean
            expected_pos = (scene_stats.get('mean_cx', 0.5), scene_stats.get('mean_cy', 0.5))
        
        ex, ey = expected_pos
        dx, dy = cx - ex, cy - ey
        dist = math.sqrt(dx*dx + dy*dy)
        
        violation = min(1.0, dist / 0.15)  # 15% threshold
        
        witness = None
        if violation > 0.1:
            witness = self._make_position_witness((cx, cy), (ex, ey), violation)
        
        return violation, witness
    
    def _make_count_witness(self, graph: SceneGraph, actual: int, expected: int, target: Optional[str]) -> np.ndarray:
        """Generate witness map for count violation."""
        witness = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        if actual > expected:
            # Extra components - highlight smallest ones
            sorted_comps = sorted(enumerate(graph.components), key=lambda x: x[1].area)
            for idx, comp in sorted_comps[expected:]:
                mask = cv2.resize(comp.mask.astype(np.float32), (self.map_size, self.map_size))
                witness = np.maximum(witness, mask)
        elif actual < expected and self.normal_slots:
            # Missing - highlight expected slots
            for slot_id, slot in self.normal_slots.items():
                if slot.get('reference_type', '') == target or target is None:
                    cx, cy = slot['mean_position']
                    x, y = int(cx * self.map_size), int(cy * self.map_size)
                    cv2.circle(witness, (x, y), int(0.05 * self.map_size), 1.0, -1)
        
        return cv2.GaussianBlur(witness, (0, 0), 2.0)
    
    def _make_component_witness(self, comp: ComponentInstance, score: float) -> np.ndarray:
        """Generate witness for a specific component."""
        mask = cv2.resize(comp.mask.astype(np.float32), (self.map_size, self.map_size))
        return cv2.GaussianBlur(mask * score, (0, 0), 2.0)
    
    def _make_position_witness(self, actual_pos: Tuple, expected_pos: Tuple, score: float) -> np.ndarray:
        """Generate witness for position deviation."""
        witness = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        ax, ay = int(actual_pos[0] * self.map_size), int(actual_pos[1] * self.map_size)
        ex, ey = int(expected_pos[0] * self.map_size), int(expected_pos[1] * self.map_size)
        cv2.arrowedLine(witness, (ex, ey), (ax, ay), score, 2, tipLength=0.15)
        cv2.circle(witness, (ex, ey), int(0.04 * self.map_size), score * 0.5, -1)
        return cv2.GaussianBlur(witness, (0, 0), 2.0)


def create_enhanced_rule_set(category: str, support_graphs: List[SceneGraph], normal_stats: Dict) -> RuleSet:
    """Create rule set with statistics-aware thresholds."""
    rule_set = RuleSet(category=category)
    
    # Compute statistics across supports
    all_stats = [extract_scene_statistics(g) for g in support_graphs]
    stat_names = all_stats[0].keys() if all_stats else []
    
    # Global count rules
    for key in ['num_components', 'left_half_count', 'right_half_count', 'upper_half_count', 'lower_half_count']:
        if key in stat_names:
            vals = [s[key] for s in all_stats]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            rule_set.add_rule(NormalityRule(
                rule_id=f'{category}_{key}',
                rule_type=RuleType.COUNT,
                part_type=None if 'num_components' in key else key.replace('_count', ''),
                description=f'Normal {key}',
                expected_count=int(round(mean_v)),
                count_tolerance=max(2, int(2 * std_v)),
                confidence=0.85,
            ))
    
    # Attribute rules on main object
    for key in ['main_obj_cx', 'main_obj_cy', 'main_obj_rel_size', 'main_obj_area']:
        if key in stat_names:
            vals = [s[key] for s in all_stats]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            attr = AttributeType.POSITION_X if 'cx' in key else (AttributeType.POSITION_Y if 'cy' in key else AttributeType.AREA)
            rule_set.add_rule(NormalityRule(
                rule_id=f'{category}_{key}',
                rule_type=RuleType.ATTRIBUTE,
                part_type='main',
                description=f'Main object {key}',
                attribute=attr,
                expected_value=mean_v,
                tolerance=max(2*std_v, 0.05),
                direction='stable',
                confidence=0.8,
            ))
    
    # Global spatial stats
    for key in ['mean_cx', 'mean_cy', 'std_cx', 'std_cy', 'vertical_spread', 'lr_balance']:
        if key in stat_names:
            vals = [s[key] for s in all_stats]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            attr = AttributeType.POSITION_X if 'cx' in key else (AttributeType.POSITION_Y if 'cy' in key else AttributeType.AREA)
            direction = 'low' if 'spread' in key or 'std' in key else 'stable'
            rule_set.add_rule(NormalityRule(
                rule_id=f'{category}_{key}',
                rule_type=RuleType.ATTRIBUTE,
                part_type=None,
                description=f'Scene {key}',
                attribute=attr,
                expected_value=mean_v,
                tolerance=max(2*std_v, 0.05),
                direction=direction,
                confidence=0.7,
            ))
    
    return rule_set


# Standalone evaluation function
def evaluate_logic_branch_v2(
    query_graph: SceneGraph,
    support_graphs: List[SceneGraph],
    normal_slots: Dict,
    normal_stats: Dict,
    map_size: int = 256,
) -> Tuple[float, Dict[str, float], Dict[str, np.ndarray]]:
    """Convenience function to evaluate logic branch v2."""
    rule_set = create_enhanced_rule_set("query", support_graphs, normal_stats)
    logic_branch = LogicBranchV2(rule_set, normal_slots, normal_stats, map_size)
    return logic_branch.evaluate(query_graph)