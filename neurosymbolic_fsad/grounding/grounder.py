"""Rule-to-Pixel Grounding for SymbolicAD.

This module provides spatial attribution for rule violations with proper
instance-level, slot-level, and relation-level grounding strategies.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from neurosymbolic_fsad.rules.generation.dsl import NormalityRule, RuleType
from neurosymbolic_fsad.perception.scene_graph import SceneGraph
from neurosymbolic_fsad.reasoning.violation_result import ViolationResult, RuleStatus


@dataclass
class GroundingResult:
    """Result of grounding a rule violation to pixel space."""
    rule_id: str
    anomaly_map: np.ndarray          # [H, W] anomaly score map
    responsibility_map: np.ndarray   # [H, W] component responsibility
    uncertainty_map: np.ndarray      # [H, W] grounding uncertainty
    strategy: str                    # grounding strategy used
    metadata: Dict[str, Any] = field(default_factory=dict)


class RuleToPixelGrounder:
    """Ground rule violations to pixel space with proper attribution."""
    
    def __init__(
        self,
        map_size: int = 256,
        blur_sigma: float = 2.0,
    ):
        self.map_size = map_size
        self.blur_sigma = blur_sigma
    
    def ground_violation(
        self,
        violation: ViolationResult,
        graph: SceneGraph,
        normal_slots: Dict[int, Dict[str, Any]],
    ) -> GroundingResult:
        """Ground a violation to pixel space based on its type."""
        rule_id = violation.rule_id
        strategy = violation.grounding_strategy
        
        # Get rule details from violation
        details = violation.details
        
        if "COUNT" in details.get("attribute", "").upper() or "COUNT" in details.get("predicate_name", "").upper():
            return self._ground_count(violation, graph, normal_slots)
        elif "SPATIAL" in details.get("predicate_name", "").upper() or "SPATIAL" in details.get("attribute", "").upper():
            return self._ground_spatial(violation, graph, normal_slots)
        elif "ATTRIBUTE" in details.get("predicate_name", "").upper() or any(attr in details.get("predicate_name", "").upper() 
            for attr in ["LENGTH", "WIDTH", "HEIGHT", "AREA", "ASPECT_RATIO", "COLOR", "CIRCULARITY", "SOLIDITY"]):
            return self._ground_attribute(violation, graph, normal_slots)
        elif "TOPOLOGICAL" in details.get("predicate_name", "").upper() or any(topo in details.get("predicate_name", "").upper() 
            for topo in ["CONNECTED", "TOUCHING", "CONTAINS", "INSIDE", "DISJOINT", "OVERLAPS"]):
            return self._ground_topological(violation, graph, normal_slots)
        elif "COMPOSITION" in details.get("predicate_name", "").upper() or "REQUIRES" in details.get("predicate_name", "").upper() or "FORBIDS" in details.get("predicate_name", "").upper():
            return self._ground_composition(violation, graph, normal_slots)
        
        # Default: use witness map if available
        if violation.anomaly_map is not None:
            anomaly_map = cv2.GaussianBlur(violation.anomaly_map, (0, 0), self.blur_sigma)
            return GroundingResult(
                rule_id=violation.rule_id,
                anomaly_map=np.clip(anomaly_map, 0, 1),
                responsibility_map=anomaly_map,
                uncertainty_map=np.ones_like(anomaly_map) * 0.5,
                strategy="witness_map",
                metadata=violation.details,
            )
        
        return GroundingResult(
            rule_id=violation.rule_id,
            anomaly_map=np.zeros((self.map_size, self.map_size)),
            responsibility_map=np.zeros((self.map_size, self.map_size)),
            uncertainty_map=np.ones((self.map_size, self.map_size)),
            strategy="none",
            metadata={},
        )
    
    def _ground_count(self, violation, graph, normal_slots) -> GroundingResult:
        """Ground count violation (missing or extra objects)."""
        anomaly_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        responsibility = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        uncertainty = np.ones((self.map_size, self.map_size), dtype=np.float32) * 0.3
        
        expected = violation.details.get("expected", 0)
        actual = violation.details.get("actual", 0)
        target_type = violation.details.get("target_type", "")
        
        if actual > expected:
            # Extra objects - highlight them
            for comp_id in violation.witness_instances:
                if comp_id < len(graph.components):
                    comp = graph.components[comp_id]
                    mask = cv2.resize(
                        comp.mask.astype(np.float32),
                        (self.map_size, self.map_size),
                    )
                    anomaly_map = np.maximum(anomaly_map, mask * violation.violation_score)
                    responsibility = np.maximum(responsibility, mask)
                    uncertainty[mask > 0] = 0.1
        
        elif actual < expected and normal_slots:
            # Missing objects - highlight expected slots
            missing_count = expected - actual
            for slot_id, slot_info in normal_slots.items():
                if slot_info.get("reference_type") == target_type and missing_count > 0:
                    cx, cy = slot_info["mean_position"]
                    x, y = int(cx * self.map_size), int(cy * self.map_size)
                    r = max(10, int(0.05 * self.map_size))
                    cv2.circle(anomaly_map, (x, y), r, violation.violation_score, -1)
                    cv2.circle(responsibility, (x, y), r, 0.5, -1)
                    uncertainty[y-r:y+r, x-r:x+r] = 0.4
                    missing_count -= 1
        
        anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), self.blur_sigma)
        responsibility = cv2.GaussianBlur(responsibility, (0, 0), self.blur_sigma)
        
        return GroundingResult(
            rule_id=violation.rule_id,
            anomaly_map=np.clip(anomaly_map, 0, 1),
            responsibility_map=np.clip(responsibility, 0, 1),
            uncertainty_map=np.clip(uncertainty, 0, 1),
            strategy="count",
            metadata={"expected": expected, "actual": actual, "target_type": target_type},
        )
    
    def _ground_attribute(self, violation, graph, normal_slots) -> GroundingResult:
        """Ground attribute violation (color, shape, size)."""
        anomaly_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        responsibility = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        uncertainty = np.ones((self.map_size, self.map_size), dtype=np.float32) * 0.2
        
        for comp_id in violation.witness_instances:
            if comp_id < len(graph.components):
                comp = graph.components[comp_id]
                mask = cv2.resize(comp.mask.astype(np.float32), (self.map_size, self.map_size))
                
                attr_name = violation.details.get("attribute", "")
                attr_val = comp.features.get(attr_name.lower(), 0)
                expected = violation.details.get("expected", 0)
                weight = min(1.0, abs(attr_val - expected) / max(abs(expected), 1e-6))
                
                anomaly_map = np.maximum(anomaly_map, mask * weight)
                responsibility = np.maximum(responsibility, mask)
                uncertainty[mask > 0] = 0.1
        
        anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), self.blur_sigma)
        responsibility = cv2.GaussianBlur(responsibility, (0, 0), self.blur_sigma)
        
        return GroundingResult(
            rule_id=violation.rule_id,
            anomaly_map=np.clip(anomaly_map, 0, 1),
            responsibility_map=np.clip(responsibility, 0, 1),
            uncertainty_map=np.clip(uncertainty, 0, 1),
            strategy="attribute",
            metadata={"attribute": violation.details.get("attribute", "")},
        )
    
    def _ground_spatial(self, violation, graph, normal_slots) -> GroundingResult:
        """Ground spatial relation violation."""
        anomaly_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        responsibility = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        uncertainty = np.ones((self.map_size, self.map_size), dtype=np.float32) * 0.3
        
        if len(violation.witness_instances) >= 2:
            src_id, tgt_id = violation.witness_instances[0], violation.witness_instances[1]
            
            for comp_id in [src_id, tgt_id]:
                if comp_id < len(graph.components):
                    comp = graph.components[comp_id]
                    mask = cv2.resize(comp.mask.astype(np.float32), (self.map_size, self.map_size))
                    anomaly_map = np.maximum(anomaly_map, mask * violation.violation_score)
                    responsibility = np.maximum(responsibility, mask)
                    uncertainty[mask > 0] = 0.2
            
            # Draw arrow between them
            src = graph.components[src_id]
            tgt = graph.components[tgt_id]
            x1, y1 = int(src.centroid[0] * self.map_size), int(src.centroid[1] * self.map_size)
            x2, y2 = int(tgt.centroid[0] * self.map_size), int(tgt.centroid[1] * self.map_size)
            cv2.arrowedLine(anomaly_map, (x1, y1), (x2, y2), violation.violation_score * 0.5, 2, tipLength=0.1)
        
        anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), self.blur_sigma)
        responsibility = cv2.GaussianBlur(responsibility, (0, 0), self.blur_sigma)
        
        return GroundingResult(
            rule_id=violation.rule_id,
            anomaly_map=np.clip(anomaly_map, 0, 1),
            responsibility_map=np.clip(responsibility, 0, 1),
            uncertainty_map=np.clip(uncertainty, 0, 1),
            strategy="spatial",
            metadata={"relation": violation.details.get("relation", "")},
        )
    
    def _ground_topological(self, violation, graph, normal_slots) -> GroundingResult:
        """Ground topological relation violation."""
        anomaly_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        responsibility = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        uncertainty = np.ones((self.map_size, self.map_size), dtype=np.float32) * 0.3
        
        for comp_id in violation.witness_instances:
            if comp_id < len(graph.components):
                comp = graph.components[comp_id]
                mask = cv2.resize(comp.mask.astype(np.float32), (self.map_size, self.map_size))
                anomaly_map = np.maximum(anomaly_map, mask * violation.violation_score)
                responsibility = np.maximum(responsibility, mask)
                uncertainty[mask > 0] = 0.2
        
        anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), self.blur_sigma)
        responsibility = cv2.GaussianBlur(responsibility, (0, 0), self.blur_sigma)
        
        return GroundingResult(
            rule_id=violation.rule_id,
            anomaly_map=np.clip(anomaly_map, 0, 1),
            responsibility_map=np.clip(responsibility, 0, 1),
            uncertainty_map=np.clip(uncertainty, 0, 1),
            strategy="topological",
            metadata={"relation": violation.details.get("relation", "")},
        )
    
    def _ground_composition(self, violation, graph, normal_slots) -> GroundingResult:
        """Ground composition violation."""
        anomaly_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        responsibility = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        uncertainty = np.ones((self.map_size, self.map_size), dtype=np.float32) * 0.4
        
        details = violation.details
        
        # Missing required components
        for missing_type in details.get("missing_required", []):
            for slot_id, slot_info in normal_slots.items():
                if slot_info.get("reference_type") == missing_type:
                    cx, cy = slot_info["mean_position"]
                    x, y = int(cx * self.map_size), int(cy * self.map_size)
                    r = max(10, int(0.05 * self.map_size))
                    cv2.circle(anomaly_map, (x, y), r, violation.violation_score, -1)
                    cv2.circle(responsibility, (x, y), r, 0.5, -1)
        
        # Forbidden combinations
        for forbidden in details.get("forbidden_violated", []):
            for comp_type in forbidden:
                for comp in graph.components:
                    if graph.component_types.get(id(comp), "") == comp_type:
                        mask = cv2.resize(comp.mask.astype(np.float32), (self.map_size, self.map_size))
                        anomaly_map = np.maximum(anomaly_map, mask * violation.violation_score * 0.7)
                        responsibility = np.maximum(responsibility, mask * 0.5)
        
        anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), self.blur_sigma)
        responsibility = cv2.GaussianBlur(responsibility, (0, 0), self.blur_sigma)
        
        return GroundingResult(
            rule_id=violation.rule_id,
            anomaly_map=np.clip(anomaly_map, 0, 1),
            responsibility_map=np.clip(responsibility, 0, 1),
            uncertainty_map=np.clip(uncertainty, 0, 1),
            strategy="composition",
            metadata={"details": details},
        )


class GroundingFusion:
    """Fuse multiple grounding results into unified logic anomaly map."""
    
    def __init__(
        self,
        map_size: int = 256,
        fusion_method: str = "max",
    ):
        self.map_size = map_size
        self.fusion_method = fusion_method
    
    def fuse(
        self,
        grounding_results: List[GroundingResult],
        rule_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Fuse multiple grounding results."""
        if not grounding_results:
            return (
                np.zeros((self.map_size, self.map_size)),
                {},
            )
        
        maps = []
        weights = []
        
        for result in grounding_results:
            maps.append(result.anomaly_map)
            if rule_weights and result.rule_id in rule_weights:
                weights.append(rule_weights[result.rule_id])
            else:
                weights.append(1.0)
        
        maps = np.stack(maps, axis=0)  # [N, H, W]
        weights = np.array(weights)
        
        if self.fusion_method == "max":
            fused = np.max(maps, axis=0)
        elif self.fusion_method == "weighted_max":
            weighted_maps = maps * weights[:, None, None]
            fused = np.max(weighted_maps, axis=0)
        elif self.fusion_method == "mean":
            fused = np.mean(maps, axis=0)
        elif self.fusion_method == "weighted_mean":
            fused = np.sum(maps * weights[:, None, None], axis=0) / np.sum(weights)
        elif self.fusion_method == "probabilistic_or":
            fused = 1.0 - np.prod(1.0 - maps, axis=0)
        else:
            fused = np.max(maps, axis=0)
        
        # Per-rule contributions
        contributions = {}
        for i, result in enumerate(grounding_results):
            if self.fusion_method in ["max", "weighted_max"]:
                max_map = np.max(maps, axis=0)
                contributions[result.rule_id] = np.where(
                    np.abs(result.anomaly_map - max_map) < 1e-6,
                    result.anomaly_map, 0.0
                )
            else:
                contributions[result.rule_id] = result.anomaly_map
        
        return np.clip(fused, 0, 1), contributions


def create_logic_anomaly_map(
    execution_result: Any,
    graph: SceneGraph,
    normal_slots: Dict[int, Dict[str, Any]],
    grounder: RuleToPixelGrounder,
    fusion: GroundingFusion,
    rule_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Create unified logic anomaly map from rule execution results."""
    grounding_results = []
    
    for violation in execution_result.rule_results:
        if violation.violation_score > 1e-6:
            result = grounder.ground_violation(violation, graph, normal_slots)
            grounding_results.append(result)
    
    fused_map, contributions = fusion.fuse(grounding_results, rule_weights)
    
    return fused_map, contributions


def compute_grounding_confidence(
    logic_map: np.ndarray,
    visual_map: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """Compute grounding confidence map.
    
    High confidence where both logic and visual anomalies agree.
    Low confidence where logic fires but visual doesn't.
    """
    visual_evidence = np.clip(visual_map / (threshold + 1e-6), 0, 1)
    logic_strength = logic_map
    confidence = 0.5 + 0.5 * visual_evidence * logic_strength
    return confidence