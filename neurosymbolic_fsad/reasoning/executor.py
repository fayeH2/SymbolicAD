"""Rule executor for generic DSL rules."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..rules.generation.dsl import (
    NormalityRule, RuleSet, RuleType, AttributeType,
    SpatialRelationType, TopologicalRelationType, RuleCondition,
)
from ..perception.scene_graph import SceneGraph, SpatialRelation, TopologicalRelation
from ..perception.segmentation import ComponentInstance


@dataclass
class ViolationEvidence:
    """Evidence for a rule violation."""
    rule_id: str
    rule_type: RuleType
    violation_score: float  # [0, 1]
    component_ids: List[int] = field(default_factory=list)
    expected: Any = None
    actual: Any = None
    details: Dict[str, Any] = field(default_factory=dict)
    witness_map: Optional[np.ndarray] = None  # Spatial witness


@dataclass
class RuleExecutionResult:
    """Result of executing a rule set on a scene graph."""
    rule_set_id: str
    image_id: str
    overall_score: float
    violations: List[ViolationEvidence] = field(default_factory=list)
    satisfied_rules: List[str] = field(default_factory=list)
    rule_scores: Dict[str, float] = field(default_factory=dict)
    spatial_maps: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def get_violation_map(self) -> np.ndarray:
        """Combine all violation spatial maps."""
        if not self.spatial_maps:
            return np.zeros((256, 256), dtype=np.float32)
        combined = np.zeros_like(list(self.spatial_maps.values())[0])
        for map_ in self.spatial_maps.values():
            combined = np.maximum(combined, map_)
        return combined


class RuleExecutor:
    """Execute normality rules on scene graphs."""
    
    def __init__(
        self,
        rule_set: RuleSet,
        normal_slots: Optional[Dict[int, Dict[str, Any]]] = None,
        map_size: int = 256,
    ):
        self.rule_set = rule_set
        self.normal_slots = normal_slots or {}
        self.map_size = map_size
    
    def execute(self, graph: SceneGraph, image_id: str = "") -> RuleExecutionResult:
        """Execute all rules on a scene graph."""
        violations = []
        satisfied = []
        rule_scores = {}
        spatial_maps = {}
        
        for rule in self.rule_set.rules:
            evidence = self._execute_rule(rule, graph)
            rule_scores[rule.rule_id] = 1.0 - evidence.violation_score
            
            if evidence.violation_score > 1e-6:
                violations.append(evidence)
                if evidence.witness_map is not None:
                    spatial_maps[rule.rule_id] = evidence.witness_map
            else:
                satisfied.append(rule.rule_id)
        
        # Compute overall anomaly score
        if violations:
            # Weighted by confidence and severity
            weighted_sum = sum(
                v.violation_score * self.rule_set.rules[
                    self.rule_set.rules.index(next(r for r in self.rule_set.rules if r.rule_id == v.rule_id))
                ].confidence
                for v in violations
            )
            weight_total = sum(
                self.rule_set.rules[
                    self.rule_set.rules.index(next(r for r in self.rule_set.rules if r.rule_id == v.rule_id))
                ].confidence
                for v in violations
            )
            overall_score = weighted_sum / max(weight_total, 1e-6)
        else:
            overall_score = 0.0
        
        return RuleExecutionResult(
            rule_set_id=self.rule_set.category,
            image_id=image_id,
            overall_score=overall_score,
            violations=violations,
            satisfied_rules=satisfied,
            rule_scores=rule_scores,
            spatial_maps=spatial_maps,
        )
    
    def _execute_rule(
        self, rule: NormalityRule, graph: SceneGraph
    ) -> ViolationEvidence:
        """Execute a single rule."""
        if rule.rule_type == RuleType.COUNT:
            return self._execute_count_rule(rule, graph)
        elif rule.rule_type == RuleType.ATTRIBUTE:
            return self._execute_attribute_rule(rule, graph)
        elif rule.rule_type == RuleType.SPATIAL:
            return self._execute_spatial_rule(rule, graph)
        elif rule.rule_type == RuleType.TOPOLOGICAL:
            return self._execute_topological_rule(rule, graph)
        elif rule.rule_type == RuleType.COMPOSITION:
            return self._execute_composition_rule(rule, graph)
        elif rule.rule_type == RuleType.CONDITIONAL:
            return self._execute_conditional_rule(rule, graph)
        
        return ViolationEvidence(
            rule_id=rule.rule_id,
            rule_type=rule.rule_type,
            violation_score=0.0,
        )
    
    def _execute_count_rule(
        self, rule: NormalityRule, graph: SceneGraph
    ) -> ViolationEvidence:
        """Execute count rule."""
        target_type = rule.part_type
        if target_type is None:
            # Global count
            count = len(graph.components)
        else:
            count = sum(1 for c in graph.components
                       if graph.component_types.get(id(c), "") == target_type)
        
        expected = rule.expected_count or 0
        tolerance = rule.count_tolerance
        diff = abs(count - expected)
        
        if diff <= tolerance:
            violation = 0.0
        else:
            violation = min(1.0, (diff - tolerance) / max(expected, 1))
        
        # Find component IDs
        comp_ids = []
        if target_type:
            comp_ids = [i for i, c in enumerate(graph.components)
                       if graph.component_types.get(id(c), "") == target_type]
        
        # Generate witness map
        witness_map = self._generate_count_witness(rule, graph, count, expected)
        
        return ViolationEvidence(
            rule_id=rule.rule_id,
            rule_type=RuleType.COUNT,
            violation_score=violation,
            component_ids=comp_ids,
            expected=expected,
            actual=count,
            details={"tolerance": tolerance, "diff": diff},
            witness_map=witness_map,
        )
    
    def _execute_attribute_rule(
        self, rule: NormalityRule, graph: SceneGraph
    ) -> ViolationEvidence:
        """Execute attribute rule."""
        target_type = rule.part_type
        attr_type = rule.attribute
        
        if target_type is None or attr_type is None:
            return ViolationEvidence(
                rule_id=rule.rule_id, rule_type=RuleType.ATTRIBUTE, violation_score=0.0
            )
        
        max_violation = 0.0
        violating_ids = []
        all_values = []
        
        for i, comp in enumerate(graph.components):
            if graph.component_types.get(id(comp), "") == target_type:
                attr_val = comp.features.get(attr_type.value)
                if attr_val is not None:
                    all_values.append(attr_val)
                    expected = rule.expected_value or 0
                    tol = rule.tolerance or 0.1
                    
                    if rule.direction == "stable":
                        diff = abs(attr_val - expected)
                        viol = max(0.0, (diff - tol) / max(abs(expected), 1e-6))
                    elif rule.direction == "low":
                        viol = max(0.0, (attr_val - expected - tol) / max(abs(expected), 1e-6))
                    elif rule.direction == "high":
                        viol = max(0.0, (expected - attr_val - tol) / max(abs(expected), 1e-6))
                    else:
                        viol = 0.0
                    
                    viol = min(1.0, viol)
                    if viol > max_violation:
                        max_violation = viol
                    if viol > 1e-6:
                        violating_ids.append(i)
        
        # Generate witness map
        witness_map = self._generate_attribute_witness(rule, graph, violating_ids, all_values)
        
        return ViolationEvidence(
            rule_id=rule.rule_id,
            rule_type=RuleType.ATTRIBUTE,
            violation_score=max_violation,
            component_ids=violating_ids,
            expected=rule.expected_value,
            actual=np.mean(all_values) if all_values else None,
            details={"values": all_values, "direction": rule.direction},
            witness_map=witness_map,
        )
    
    def _execute_spatial_rule(
        self, rule: NormalityRule, graph: SceneGraph
    ) -> ViolationEvidence:
        """Execute spatial relation rule."""
        if not rule.spatial_relation or not rule.part_type or not rule.target_part_type:
            return ViolationEvidence(
                rule_id=rule.rule_id, rule_type=RuleType.SPATIAL, violation_score=0.0
            )
        
        source_comps = [i for i, c in enumerate(graph.components)
                       if graph.component_types.get(id(c), "") == rule.part_type]
        target_comps = [i for i, c in enumerate(graph.components)
                       if graph.component_types.get(id(c), "") == rule.target_part_type]
        
        if not source_comps or not target_comps:
            return ViolationEvidence(
                rule_id=rule.rule_id, rule_type=RuleType.SPATIAL, violation_score=0.0
            )
        
        min_violation = 1.0
        best_pair = None
        
        for s_idx in source_comps:
            for t_idx in target_comps:
                src = graph.components[s_idx]
                tgt = graph.components[t_idx]
                viol = self._compute_spatial_violation(src, tgt, rule)
                if viol < min_violation:
                    min_violation = viol
                    best_pair = (s_idx, t_idx)
        
        # Violation is 1 - satisfaction
        violation = 1.0 - min_violation
        
        witness_map = None
        if best_pair:
            witness_map = self._generate_spatial_witness(
                rule, graph, best_pair[0], best_pair[1]
            )
        
        return ViolationEvidence(
            rule_id=rule.rule_id,
            rule_type=RuleType.SPATIAL,
            violation_score=violation,
            component_ids=list(best_pair) if best_pair else [],
            expected=rule.spatial_relation.value,
            details={"threshold": rule.spatial_threshold},
            witness_map=witness_map,
        )
    
    def _compute_spatial_violation(
        self, src: ComponentInstance, tgt: ComponentInstance, rule: NormalityRule
    ) -> float:
        """Compute spatial satisfaction score."""
        cx1, cy1 = src.centroid
        cx2, cy2 = tgt.centroid
        dx = cx2 - cx1
        dy = cy2 - cy1
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist > rule.spatial_threshold:
            return 0.0
        
        if rule.spatial_relation == SpatialRelationType.LEFT_OF:
            satisfaction = 1.0 if dx < -0.02 else max(0.0, 1.0 + dx / 0.02)
        elif rule.spatial_relation == SpatialRelationType.RIGHT_OF:
            satisfaction = 1.0 if dx > 0.02 else max(0.0, 1.0 - dx / 0.02)
        elif rule.spatial_relation == SpatialRelationType.ABOVE:
            satisfaction = 1.0 if dy < -0.02 else max(0.0, 1.0 + dy / 0.02)
        elif rule.spatial_relation == SpatialRelationType.BELOW:
            satisfaction = 1.0 if dy > 0.02 else max(0.0, 1.0 - dy / 0.02)
        elif rule.spatial_relation == SpatialRelationType.NEAR:
            satisfaction = max(0.0, 1.0 - dist / rule.spatial_threshold)
        elif rule.spatial_relation == SpatialRelationType.ALIGNED_HORIZONTAL:
            satisfaction = max(0.0, 1.0 - abs(dy) / 0.03)
        elif rule.spatial_relation == SpatialRelationType.ALIGNED_VERTICAL:
            satisfaction = max(0.0, 1.0 - abs(dx) / 0.03)
        elif rule.spatial_relation == SpatialRelationType.SYMMETRIC:
            # Check if pair is symmetric around center
            center_dist = abs((cx1 + cx2) / 2 - 0.5)
            satisfaction = max(0.0, 1.0 - center_dist / 0.1)
        else:
            satisfaction = 1.0
        
        return satisfaction
    
    def _execute_topological_rule(
        self, rule: NormalityRule, graph: SceneGraph
    ) -> ViolationEvidence:
        """Execute topological relation rule."""
        if not rule.topological_relation or not rule.part_type or not rule.target_part_type:
            return ViolationEvidence(
                rule_id=rule.rule_id, rule_type=RuleType.TOPOLOGICAL, violation_score=0.0
            )
        
        found = False
        violating_ids = []
        
        for rel in graph.topological_relations:
            src_type = graph.component_types.get(rel.source_id, "")
            tgt_type = graph.component_types.get(rel.target_id, "")
            
            if src_type == rule.part_type and tgt_type == rule.target_part_type:
                if rel.relation_type == rule.topological_relation.value:
                    found = True
                else:
                    violating_ids.extend([rel.source_id, rel.target_id])
        
        violation = 0.0 if found else 1.0
        
        witness_map = self._generate_topological_witness(rule, graph)
        
        return ViolationEvidence(
            rule_id=rule.rule_id,
            rule_type=RuleType.TOPOLOGICAL,
            violation_score=violation,
            component_ids=violating_ids,
            expected=rule.topological_relation.value,
            details={"found": found},
            witness_map=witness_map,
        )
    
    def _execute_composition_rule(
        self, rule: NormalityRule, graph: SceneGraph
    ) -> ViolationEvidence:
        """Execute composition rule."""
        present_types = set(graph.component_types.values())
        violation = 0.0
        details = {}
        
        # Check required types
        missing = []
        for req_type in rule.required_part_types:
            if req_type not in present_types:
                missing.append(req_type)
                violation = max(violation, 0.8)
        
        # Check forbidden combinations
        forbidden_violated = []
        for forbidden in rule.forbidden_combinations:
            if all(t in present_types for t in forbidden):
                forbidden_violated.append(forbidden)
                violation = max(violation, 1.0)
        
        details = {
            "missing_required": missing,
            "forbidden_violated": forbidden_violated,
            "present_types": list(present_types),
        }
        
        witness_map = self._generate_composition_witness(rule, graph, details)
        
        return ViolationEvidence(
            rule_id=rule.rule_id,
            rule_type=RuleType.COMPOSITION,
            violation_score=violation,
            component_ids=[],
            details=details,
            witness_map=witness_map,
        )
    
    def _execute_conditional_rule(
        self, rule: NormalityRule, graph: SceneGraph
    ) -> ViolationEvidence:
        """Execute conditional rule."""
        # Evaluate conditions
        context = self._build_context(graph)
        
        all_conditions_met = all(
            cond.evaluate(context) for cond in rule.conditions
        )
        
        if all_conditions_met and rule.then_rule:
            return self._execute_rule(rule.then_rule, graph)
        elif not all_conditions_met and rule.else_rule:
            return self._execute_rule(rule.else_rule, graph)
        
        return ViolationEvidence(
            rule_id=rule.rule_id,
            rule_type=RuleType.CONDITIONAL,
            violation_score=0.0,
            details={"conditions_met": all_conditions_met},
        )
    
    def _build_context(self, graph: SceneGraph) -> Dict[str, Any]:
        """Build evaluation context from scene graph."""
        context = {}
        
        # Count by type
        type_counts = {}
        for c in graph.components:
            t = graph.component_types.get(id(c), "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        context["counts"] = type_counts
        
        # Attribute statistics
        for attr in AttributeType:
            values = []
            for c in graph.components:
                v = c.features.get(attr.value)
                if v is not None:
                    values.append(v)
            if values:
                context[f"{attr.value}_mean"] = np.mean(values)
                context[f"{attr.value}_std"] = np.std(values)
                context[f"{attr.value}_min"] = np.min(values)
                context[f"{attr.value}_max"] = np.max(values)
        
        return context
    
    # Witness map generation methods
    
    def _generate_count_witness(
        self, rule: NormalityRule, graph: SceneGraph,
        actual_count: int, expected_count: int
    ) -> np.ndarray:
        """Generate spatial witness map for count violation."""
        witness = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        target_type = rule.part_type
        
        if actual_count > expected_count:
            # Extra objects - highlight them
            for i, comp in enumerate(graph.components):
                if target_type is None or graph.component_types.get(id(comp), "") == target_type:
                    if i >= expected_count:  # Extra components
                        mask = cv2.resize(
                            comp.mask.astype(np.float32),
                            (self.map_size, self.map_size),
                        )
                        witness = np.maximum(witness, mask)
        
        elif actual_count < expected_count and self.normal_slots:
            # Missing objects - highlight expected slots
            for slot_id, slot_info in self.normal_slots.items():
                if slot_info.get("reference_type") == target_type:
                    cx, cy = slot_info["mean_position"]
                    # Draw slot region
                    y, x = int(cy * self.map_size), int(cx * self.map_size)
                    r = int(0.05 * self.map_size)
                    cv2.circle(witness, (x, y), r, 1.0, -1)
        
        return cv2.GaussianBlur(witness, (0, 0), 2.0)
    
    def _generate_attribute_witness(
        self, rule: NormalityRule, graph: SceneGraph,
        violating_ids: List[int], values: List[float]
    ) -> np.ndarray:
        """Generate spatial witness map for attribute violation."""
        witness = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        for idx in violating_ids:
            if idx < len(graph.components):
                comp = graph.components[idx]
                mask = cv2.resize(
                    comp.mask.astype(np.float32),
                    (self.map_size, self.map_size),
                )
                # Weight by violation severity
                attr_val = comp.features.get(rule.attribute.value, 0) if rule.attribute else 0
                expected = rule.expected_value or 0
                weight = min(1.0, abs(attr_val - expected) / max(abs(expected), 1e-6))
                witness = np.maximum(witness, mask * weight)
        
        return cv2.GaussianBlur(witness, (0, 0), 2.0)
    
    def _generate_spatial_witness(
        self, rule: NormalityRule, graph: SceneGraph,
        src_idx: int, tgt_idx: int
    ) -> np.ndarray:
        """Generate spatial witness map for spatial violation."""
        witness = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        src = graph.components[src_idx]
        tgt = graph.components[tgt_idx]
        
        # Highlight both components
        for comp in [src, tgt]:
            mask = cv2.resize(
                comp.mask.astype(np.float32),
                (self.map_size, self.map_size),
            )
            witness = np.maximum(witness, mask)
        
        # Draw arrow between them
        x1, y1 = int(src.centroid[0] * self.map_size), int(src.centroid[1] * self.map_size)
        x2, y2 = int(tgt.centroid[0] * self.map_size), int(tgt.centroid[1] * self.map_size)
        cv2.arrowedLine(witness, (x1, y1), (x2, y2), 0.5, 2, tipLength=0.1)
        
        return cv2.GaussianBlur(witness, (0, 0), 2.0)
    
    def _generate_topological_witness(
        self, rule: NormalityRule, graph: SceneGraph
    ) -> np.ndarray:
        """Generate witness map for topological violation."""
        witness = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        for rel in graph.topological_relations:
            src_type = graph.component_types.get(rel.source_id, "")
            tgt_type = graph.component_types.get(rel.target_id, "")
            
            if src_type == rule.part_type and tgt_type == rule.target_part_type:
                src = graph.components[rel.source_id]
                tgt = graph.components[rel.target_id]
                
                for comp in [src, tgt]:
                    mask = cv2.resize(
                        comp.mask.astype(np.float32),
                        (self.map_size, self.map_size),
                    )
                    witness = np.maximum(witness, mask)
        
        return cv2.GaussianBlur(witness, (0, 0), 2.0)
    
    def _generate_composition_witness(
        self, rule: NormalityRule, graph: SceneGraph, details: Dict
    ) -> np.ndarray:
        """Generate witness map for composition violation."""
        witness = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        
        # Highlight missing required components' expected slots
        for missing_type in details.get("missing_required", []):
            for slot_id, slot_info in self.normal_slots.items():
                if slot_info.get("reference_type") == missing_type:
                    cx, cy = slot_info["mean_position"]
                    y, x = int(cy * self.map_size), int(cx * self.map_size)
                    r = int(0.05 * self.map_size)
                    cv2.circle(witness, (x, y), r, 1.0, -1)
        
        # Highlight forbidden combinations
        for forbidden in details.get("forbidden_violated", []):
            for comp_type in forbidden:
                for i, comp in enumerate(graph.components):
                    if graph.component_types.get(id(comp), "") == comp_type:
                        mask = cv2.resize(
                            comp.mask.astype(np.float32),
                            (self.map_size, self.map_size),
                        )
                        witness = np.maximum(witness, mask * 0.5)
        
        return cv2.GaussianBlur(witness, (0, 0), 2.0)


# Import cv2 at module level for witness generation
try:
    import cv2
except ImportError:
    cv2 = None