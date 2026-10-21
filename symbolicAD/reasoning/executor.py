"""Rule executor with proper interfaces for SymbolicAD.

This replaces the old executor with a clean interface that:
1. Uses predicate registry for extensible rule execution
2. Returns unified ViolationResult with tri-state logic
3. Supports extensible predicates via registry
4. Returns proper spatial attribution (instance_mask, slot_map, etc.)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import cv2

from symbolicAD.rules.generation.dsl import (
    NormalityRule, RuleSet, RuleType, RuleCondition,
    REGISTRY, PredicateSpec,
)
from symbolicAD.perception.scene_graph import SceneGraph, SpatialRelation, TopologicalRelation
from symbolicAD.perception.segmentation import ComponentInstance
from symbolicAD.reasoning.violation_result import ViolationResult, RuleStatus


# ============================================================
# PREDICATE EXECUTORS - Registry-based execution
# ============================================================

class PredicateExecutor(ABC):
    """Abstract base for predicate executors."""
    
    @abstractmethod
    def execute(self, rule: 'NormalityRule', graph: 'SceneGraph', 
                normal_slots: Dict[int, Dict[str, Any]]) -> Tuple[float, Dict[str, Any], Optional[np.ndarray]]:
        """Execute predicate on scene graph.
        
        Returns:
            violation_score: float in [0, 1]
            details: dict with execution details
            witness_map: optional spatial attribution map
        """
        pass
    
    @abstractmethod
    def ground(self, rule: 'NormalityRule', graph: 'SceneGraph',
               normal_slots: Dict[int, Dict[str, Any]],
               violation_score: float, details: Dict[str, Any]) -> np.ndarray:
        """Generate spatial anomaly map for violation."""
        pass


class CountExecutor(PredicateExecutor):
    """Executor for COUNT predicates."""
    
    def execute(self, rule: 'NormalityRule', graph: 'SceneGraph',
                normal_slots: Dict[int, Dict[str, Any]]) -> Tuple[float, Dict[str, Any], Optional[np.ndarray]]:
        target_type = rule.predicate_args.get("entity_type")
        expected = rule.predicate_params.get("expected_count", 0)
        tolerance = rule.predicate_params.get("tolerance", 0)
        
        if target_type is None:
            count = len(graph.components)
        else:
            count = sum(1 for c in graph.components
                       if graph.component_types.get(id(c), "") == target_type)
        
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
        
        details = {
            "tolerance": tolerance, "diff": diff,
            "target_type": target_type, "expected": expected, "actual": count
        }
        
        # Generate witness map
        witness_map = self._make_count_witness(rule, graph, count, expected, target_type, normal_slots)
        
        return violation, details, witness_map
    
    def ground(self, rule: 'NormalityRule', graph: 'SceneGraph',
               normal_slots: Dict[int, Dict[str, Any]],
               violation_score: float, details: Dict[str, Any]) -> np.ndarray:
        return self._make_count_witness(
            None, graph, 
            details.get("actual", 0), 
            details.get("expected", 0),
            details.get("target_type"),
            normal_slots
        )
    
    def _make_count_witness(self, rule, graph, actual_count, expected_count, target_type, normal_slots):
        map_size = 256
        witness = np.zeros((map_size, map_size), dtype=np.float32)
        
        if actual_count > expected_count:
            # Extra objects
            for i, comp in enumerate(graph.components):
                if target_type is None or graph.component_types.get(id(comp), "") == target_type:
                    if i >= expected_count:
                        mask = cv2.resize(
                            comp.mask.astype(np.float32),
                            (map_size, map_size),
                        )
                        witness = np.maximum(witness, mask)
        
        elif actual_count < expected_count and normal_slots:
            for slot_id, slot_info in normal_slots.items():
                if slot_info.get("reference_type") == target_type:
                    cx, cy = slot_info["mean_position"]
                    x, y = int(cx * map_size), int(cy * map_size)
                    r = int(0.05 * map_size)
                    cv2.circle(witness, (x, y), r, 1.0, -1)
        
        return cv2.GaussianBlur(witness, (0, 0), 2.0)


class AttributeExecutor(PredicateExecutor):
    """Executor for ATTRIBUTE predicates (LENGTH, WIDTH, COLOR, etc.)."""
    
    # Mapping from predicate name to feature key
    ATTRIBUTE_FEATURE_MAP = {
        "LENGTH": "shape_length",
        "WIDTH": "shape_width", 
        "HEIGHT": "shape_height",
        "AREA": "shape_area",
        "ASPECT_RATIO": "shape_aspect",
        "CIRCULARITY": "shape_circularity",
        "SOLIDITY": "shape_solidity",
        "COLOR_H": "color_h",
        "COLOR_S": "color_s",
        "COLOR_V": "color_v",
        "ASPECT_RATIO": "shape_aspect",
        "CIRCULARITY": "shape_circularity",
        "SOLIDITY": "shape_solidity",
    }
    
    def execute(self, rule: 'NormalityRule', graph: 'SceneGraph',
                normal_slots: Dict[int, Dict[str, Any]]) -> Tuple[float, Dict[str, Any], Optional[np.ndarray]]:
        target_type = rule.predicate_args.get("entity_type")
        attr_name = rule.predicate_name.upper()
        # Map predicate name to actual feature key
        feature_key = self.ATTRIBUTE_FEATURE_MAP.get(attr_name, attr_name.lower())
        expected = rule.predicate_params.get("expected_value", 0)
        tol = rule.predicate_params.get("tolerance", 0.1)
        direction = rule.predicate_params.get("direction", "stable")
        
        max_violation = 0.0
        violating_ids = []
        all_values = []
        
        for i, comp in enumerate(graph.components):
            if target_type is None or graph.component_types.get(id(comp), "") == target_type:
                attr_val = comp.features.get(feature_key)
                if attr_val is not None:
                    all_values.append(attr_val)
                    if rule.predicate_params.get("direction", "stable") == "stable":
                        diff = abs(attr_val - rule.predicate_params.get("expected_value", 0))
                        viol = max(0.0, (diff - tol) / max(abs(rule.predicate_params.get("expected_value", 0)), 1e-6))
                    elif rule.predicate_params.get("direction") == "low":
                        viol = max(0.0, (attr_val - rule.predicate_params.get("expected_value", 0) - tol) / max(abs(rule.predicate_params.get("expected_value", 0)), 1e-6))
                    else:
                        viol = max(0.0, (rule.predicate_params.get("expected_value", 0) - attr_val - tol) / max(abs(rule.predicate_params.get("expected_value", 0)), 1e-6))
                    
                    viol = min(1.0, viol)
                    if viol > max_violation:
                        max_violation = viol
                    if viol > 1e-6:
                        violating_ids.append(i)
        
        details = {"values": all_values, "direction": direction, "attribute": attr_name}
        witness_map = self._make_attribute_witness(graph, violating_ids, all_values, attr_name) if max_violation > 1e-6 else None
        
        return max_violation, details, witness_map
    
    def ground(self, rule, graph, normal_slots, violation_score, details):
        return self._make_attribute_witness(graph, details.get("violating_ids", []), 
                                           details.get("values", []), details.get("attribute", "LENGTH"))
    
    def _make_attribute_witness(self, graph, violating_ids, values, attr_name):
        map_size = 256
        witness = np.zeros((map_size, map_size), dtype=np.float32)
        
        for idx in violating_ids:
            if idx < len(graph.components):
                comp = graph.components[idx]
                mask = cv2.resize(
                    comp.mask.astype(np.float32),
                    (map_size, map_size),
                )
                attr_val = comp.features.get(attr_name.lower(), 0)
                expected = 0
                weight = min(1.0, abs(attr_val - expected) / max(abs(expected), 1e-6))
                witness = np.maximum(witness, mask * weight)
        
        return cv2.GaussianBlur(witness, (0, 0), 2.0)


class SpatialExecutor(PredicateExecutor):
    """Executor for SPATIAL predicates."""
    
    def execute(self, rule: 'NormalityRule', graph: 'SceneGraph',
                normal_slots: Dict[int, Dict[str, Any]]) -> Tuple[float, Dict[str, Any], Optional[np.ndarray]]:
        if not rule.predicate_args.get("entity_type") or not rule.predicate_args.get("target_entity_type"):
            return 0.0, {}, None
        
        source_comps = [i for i, c in enumerate(graph.components)
                       if graph.component_types.get(id(c), "") == rule.predicate_args.get("entity_type")]
        target_comps = [i for i, c in enumerate(graph.components)
                       if graph.component_types.get(id(c), "") == rule.predicate_args.get("target_entity_type")]
        
        if not source_comps or not target_comps:
            return 0.0, {}, None
        
        threshold = rule.predicate_params.get("threshold", 0.1)
        relation = rule.predicate_name
        
        min_violation = 1.0
        best_pair = None
        
        for s_idx in source_comps:
            for t_idx in target_comps:
                src = graph.components[s_idx]
                tgt = graph.components[t_idx]
                viol = self._compute_spatial_violation(src, tgt, relation, threshold)
                if viol < min_violation:
                    min_violation = viol
                    best_pair = (s_idx, t_idx)
        
        violation = 1.0 - min_violation
        details = {"threshold": threshold, "relation": relation}
        witness_map = self._make_spatial_witness(graph, best_pair, relation) if best_pair and violation > 1e-6 else None
        
        return violation, details, witness_map
    
    def _compute_spatial_violation(self, src, tgt, relation, threshold):
        cx1, cy1 = src.centroid
        cx2, cy2 = tgt.centroid
        dx = cx2 - cx1
        dy = cy2 - cy1
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist > threshold:
            return 0.0
        
        if relation == "LEFT_OF":
            satisfaction = 1.0 if dx < -0.02 else max(0.0, 1.0 + dx / 0.02)
        elif relation == "RIGHT_OF":
            satisfaction = 1.0 if dx > 0.02 else max(0.0, 1.0 - dx / 0.02)
        elif relation == "ABOVE":
            satisfaction = 1.0 if dy < -0.02 else max(0.0, 1.0 + dy / 0.02)
        elif relation == "BELOW":
            satisfaction = 1.0 if dy > 0.02 else max(0.0, 1.0 - dy / 0.02)
        elif relation == "NEAR":
            satisfaction = max(0.0, 1.0 - dist / threshold)
        elif relation == "ALIGNED_HORIZONTAL":
            satisfaction = max(0.0, 1.0 - abs(dy) / 0.03)
        elif relation == "ALIGNED_VERTICAL":
            satisfaction = max(0.0, 1.0 - abs(dx) / 0.03)
        elif relation == "SYMMETRIC_ABOUT":
            center_dist = abs((cx1 + cx2) / 2 - 0.5)
            satisfaction = max(0.0, 1.0 - center_dist / 0.1)
        else:
            satisfaction = 1.0
        
        return satisfaction
    
    def ground(self, rule, graph, normal_slots, violation_score, details):
        return self._make_spatial_witness(graph, details.get("best_pair"), details.get("relation"))
    
    def _make_spatial_witness(self, graph, best_pair, relation):
        if not best_pair:
            return np.zeros((256, 256), dtype=np.float32)
        
        map_size = 256
        witness = np.zeros((map_size, map_size), dtype=np.float32)
        src, tgt = graph.components[best_pair[0]], graph.components[best_pair[1]]
        
        for comp in [src, tgt]:
            mask = cv2.resize(comp.mask.astype(np.float32), (map_size, map_size))
            witness = np.maximum(witness, mask)
        
        x1, y1 = int(src.centroid[0] * map_size), int(src.centroid[1] * map_size)
        x2, y2 = int(tgt.centroid[0] * map_size), int(tgt.centroid[1] * map_size)
        cv2.arrowedLine(witness, (x1, y1), (x2, y2), 0.5, 2, tipLength=0.1)
        
        return cv2.GaussianBlur(witness, (0, 0), 2.0)


class TopologicalExecutor(PredicateExecutor):
    """Executor for TOPOLOGICAL predicates."""
    
    def execute(self, rule, graph, normal_slots):
        if not rule.predicate_args.get("entity_type") or not rule.predicate_args.get("target_entity_type"):
            return 0.0, {}, None
        
        relation = rule.predicate_name
        found = False
        violating_ids = []
        
        for rel in graph.topological_relations:
            src_type = graph.component_types.get(rel.source_id, "")
            tgt_type = graph.component_types.get(rel.target_id, "")
            
            if src_type == rule.predicate_args.get("entity_type") and tgt_type == rule.predicate_args.get("target_entity_type"):
                if rel.relation_type == relation:
                    found = True
                else:
                    violating_ids.extend([rel.source_id, rel.target_id])
        
        violation = 0.0 if found else 1.0
        details = {"found": found, "relation": relation}
        witness_map = self._make_topological_witness(graph, relation) if not found else None
        
        return violation, details, witness_map
    
    def ground(self, rule, graph, normal_slots, violation_score, details):
        return self._make_topological_witness(graph, details.get("relation"))
    
    def _make_topological_witness(self, graph, relation):
        map_size = 256
        witness = np.zeros((map_size, map_size), dtype=np.float32)
        
        for rel in graph.topological_relations:
            src_type = graph.component_types.get(rel.source_id, "")
            tgt_type = graph.component_types.get(rel.target_id, "")
            
            if src_type == "component" and tgt_type == "component":
                src = graph.components[rel.source_id]
                tgt = graph.components[rel.target_id]
                
                for comp in [src, tgt]:
                    mask = cv2.resize(comp.mask.astype(np.float32), (map_size, map_size))
                    witness = np.maximum(witness, mask)
        
        return cv2.GaussianBlur(witness, (0, 0), 2.0)


class CompositionExecutor(PredicateExecutor):
    """Executor for COMPOSITION predicates."""
    
    def execute(self, rule, graph, normal_slots):
        present_types = set(graph.component_types.values())
        violation = 0.0
        details = {}
        
        missing = []
        for req_type in rule.predicate_params.get("required_types", []):
            if req_type not in present_types:
                missing.append(req_type)
                violation = max(violation, 0.8)
        
        forbidden_violated = []
        for forbidden in rule.predicate_params.get("forbidden", []):
            if all(t in present_types for t in forbidden):
                forbidden_violated.append(forbidden)
                violation = max(violation, 1.0)
        
        details = {
            "missing_required": missing,
            "forbidden_violated": forbidden_violated,
            "present_types": list(present_types),
        }
        
        witness_map = self._make_composition_witness(graph, details) if violation > 1e-6 else None
        
        return violation, details, witness_map
    
    def ground(self, rule, graph, normal_slots, violation_score, details):
        return self._make_composition_witness(graph, details)
    
    def _make_composition_witness(self, graph, details):
        map_size = 256
        witness = np.zeros((map_size, map_size), dtype=np.float32)
        
        for missing_type in details.get("missing_required", []):
            for slot_id, slot_info in normal_slots.items():
                if slot_info.get("reference_type") == missing_type:
                    cx, cy = slot_info["mean_position"]
                    x, y = int(cx * map_size), int(cy * map_size)
                    r = int(0.05 * map_size)
                    cv2.circle(witness, (x, y), r, 1.0, -1)
        
        for forbidden in details.get("forbidden_violated", []):
            for comp_type in forbidden:
                for comp in graph.components:
                    if graph.component_types.get(id(comp), "") == comp_type:
                        mask = cv2.resize(comp.mask.astype(np.float32), (256, 256))
                        witness = np.maximum(witness, mask * 0.5)
        
        return cv2.GaussianBlur(witness, (0, 0), 2.0)


class ConditionalExecutor(PredicateExecutor):
    """Executor for CONDITIONAL predicates."""
    
    def execute(self, rule, graph, normal_slots):
        # Evaluate conditions
        context = self._build_context(graph)
        
        all_conditions_met = all(
            cond.evaluate(context) for cond in rule.conditions
        )
        
        if all_conditions_met and rule.then_rule:
            # Delegate to then_rule executor
            executor = EXECUTOR_REGISTRY.get(rule.then_rule.predicate_name)
            if executor:
                return executor.execute(rule.then_rule, graph, {})
        elif not all_conditions_met and rule.else_rule:
            executor = EXECUTOR_REGISTRY.get(rule.else_rule.predicate_name)
            if executor:
                return executor.execute(rule.else_rule, graph, {})
        
        return 0.0, {"conditions_met": all_conditions_met}, None
    
    def ground(self, rule, graph, normal_slots, violation_score, details):
        return np.zeros((256, 256), dtype=np.float32)
    
    def _build_context(self, graph):
        context = {}
        type_counts = {}
        for c in graph.components:
            t = graph.component_types.get(id(c), "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        context["counts"] = type_counts
        return context


# ============================================================
# EXECUTOR REGISTRY
# ============================================================

EXECUTOR_REGISTRY: Dict[str, PredicateExecutor] = {
    "COUNT": CountExecutor(),
    "LENGTH": AttributeExecutor(),
    "WIDTH": AttributeExecutor(),
    "HEIGHT": AttributeExecutor(),
    "AREA": AttributeExecutor(),
    "ASPECT_RATIO": AttributeExecutor(),
    "COLOR_H": AttributeExecutor(),
    "COLOR_S": AttributeExecutor(),
    "COLOR_V": AttributeExecutor(),
    "CIRCULARITY": AttributeExecutor(),
    "SOLIDITY": AttributeExecutor(),
    "LEFT_OF": SpatialExecutor(),
    "RIGHT_OF": SpatialExecutor(),
    "ABOVE": SpatialExecutor(),
    "BELOW": SpatialExecutor(),
    "NEAR": SpatialExecutor(),
    "FAR": SpatialExecutor(),
    "ALIGNED_HORIZONTAL": SpatialExecutor(),
    "ALIGNED_VERTICAL": SpatialExecutor(),
    "SYMMETRIC_ABOUT": SpatialExecutor(),
    "CONNECTED": TopologicalExecutor(),
    "TOUCHING": TopologicalExecutor(),
    "CONTAINS": TopologicalExecutor(),
    "INSIDE": TopologicalExecutor(),
    "DISJOINT": TopologicalExecutor(),
    "OVERLAPS": TopologicalExecutor(),
    "REQUIRES": CompositionExecutor(),
    "FORBIDS": CompositionExecutor(),
    "IF_THEN_ELSE": ConditionalExecutor(),
}


def register_executor(predicate_name: str, executor: PredicateExecutor) -> None:
    """Register a custom predicate executor."""
    EXECUTOR_REGISTRY[predicate_name.upper()] = executor


def get_executor(predicate_name: str) -> Optional[PredicateExecutor]:
    """Get executor for predicate."""
    return EXECUTOR_REGISTRY.get(predicate_name.upper())


# ============================================================
# MAIN RULE EXECUTOR
# ============================================================

@dataclass
class ExecutionResult:
    """Result of executing a rule set on a scene graph."""
    rule_set_id: str
    image_id: str
    overall_score: float
    rule_results: List['ViolationResult'] = field(default_factory=list)
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
    """Execute normality rules on scene graphs with extensible predicates."""
    
    def __init__(
        self,
        rule_set: RuleSet,
        normal_slots: Optional[Dict[int, Dict[str, Any]]] = None,
        map_size: int = 256,
    ):
        self.rule_set = rule_set
        self.normal_slots = normal_slots or {}
        self.map_size = map_size
    
    def execute(self, graph: SceneGraph, image_id: str = "") -> ExecutionResult:
        """Execute all rules on a scene graph."""
        rule_results = []
        rule_scores = {}
        spatial_maps = {}
        
        for rule in self.rule_set.rules:
            # Validate rule
            if not rule.validate():
                result = ViolationResult.unknown(
                    rule_id=rule.rule_id,
                    reason="Rule validation failed: predicate or args invalid",
                    grounding_strategy="none",
                )
                rule_results.append(result)
                rule_scores[rule.rule_id] = 1.0
                continue
            
            # Get executor
            executor = EXECUTOR_REGISTRY.get(rule.predicate_name.upper())
            if not executor:
                result = ViolationResult.unknown(
                    rule_id=rule.rule_id,
                    reason=f"No executor for predicate: {rule.predicate_name}",
                    grounding_strategy="none",
                )
                rule_results.append(result)
                rule_scores[rule.rule_id] = 1.0
                continue
            
            # Execute
            try:
                violation_score, details, witness_map = executor.execute(
                    rule, graph, self.normal_slots
                )
            except Exception as e:
                result = ViolationResult.unknown(
                    rule_id=rule.rule_id,
                    reason=f"Execution error: {str(e)}",
                    grounding_strategy="none",
                )
                rule_results.append(result)
                rule_scores[rule.rule_id] = 1.0
                continue
            
            # Determine status
            if violation_score > 1e-6:
                status = RuleStatus.VIOLATED
            else:
                status = RuleStatus.SATISFIED
            
            # Grounding
            witness_map = None
            if violation_score > 1e-6:
                witness_map = executor.ground(rule, graph, self.normal_slots, violation_score, {})
            
            # Determine grounding strategy
            strategy = rule.get_grounding_strategy()
            
            # Evidence confidence based on rule confidence and witness quality
            evidence_conf = rule.confidence * (1.0 - violation_score) if status == RuleStatus.SATISFIED else rule.confidence * violation_score
            
            # Create result
            result = ViolationResult(
                rule_id=rule.rule_id,
                status=RuleStatus(status),
                violation_score=violation_score,
                evidence_confidence=evidence_conf,
                witness_instances=details.get("violating_ids", []) if 'details' in locals() else [],
                grounding_strategy=rule.get_grounding_strategy(),
                anomaly_map=witness_map,
                attribution_uncertainty=1.0 - evidence_conf,
                details=details,
            )
            
            rule_results.append(result)
            rule_scores[rule.rule_id] = 1.0 - violation_score
            
            if witness_map is not None:
                spatial_maps[rule.rule_id] = witness_map
        
        # Compute overall anomaly score
        violated = [r for r in rule_results if r.status == RuleStatus.VIOLATED]
        if violated:
            weighted_sum = sum(r.violation_score * self.rule_set.rules[
                next(i for i, r in enumerate(self.rule_set.rules) if r.rule_id == r.rule_id)
            ].confidence for r in violated)
            weight_total = sum(self.rule_set.rules[
                next(i for i, r in enumerate(self.rule_set.rules) if r.rule_id == r.rule_id)
            ].confidence for r in violated)
            overall_score = weighted_sum / max(weight_total, 1e-6)
        else:
            overall_score = 0.0
        
        return ExecutionResult(
            rule_set_id=self.rule_set.category,
            image_id=image_id,
            overall_score=overall_score,
            rule_results=rule_results,
            rule_scores=rule_scores,
            spatial_maps=spatial_maps,
        )


__all__ = [
    "PredicateExecutor", "CountExecutor", "AttributeExecutor", "SpatialExecutor",
    "TopologicalExecutor", "CompositionExecutor", "ConditionalExecutor",
    "EXECUTOR_REGISTRY", "register_executor", "get_executor",
    "RuleExecutor", "ExecutionResult",
]