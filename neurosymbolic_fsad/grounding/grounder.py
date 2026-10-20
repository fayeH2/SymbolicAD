"""Rule-to-Pixel Grounding module for spatial anomaly attribution."""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..rules.generation.dsl import NormalityRule, RuleType
from ..perception.scene_graph import SceneGraph
from ..perception.segmentation import ComponentInstance
from ..reasoning.executor import ViolationEvidence


@dataclass
class GroundingResult:
    """Result of grounding a rule violation to pixel space."""
    rule_id: str
    anomaly_map: np.ndarray  # [H, W] in [0, 1]
    responsibility_map: np.ndarray  # [H, W] component responsibility
    uncertainty_map: np.ndarray  # [H, W] grounding uncertainty
    metadata: Dict[str, Any] = field(default_factory=dict)


class RuleToPixelGrounder:
    """Ground rule violations to pixel-level anomaly maps."""
    
    def __init__(
        self,
        map_size: int = 256,
        blur_sigma: float = 2.0,
    ):
        self.map_size = map_size
        self.blur_sigma = blur_sigma
    
    def ground_violation(
        self,
        evidence: ViolationEvidence,
        graph: SceneGraph,
        normal_slots: Dict[int, Dict[str, Any]],
    ) -> GroundingResult:
        """Ground a violation evidence to pixel space."""
        rule_type = evidence.rule_type
        
        if rule_type == RuleType.COUNT:
            return self._ground_count_violation(evidence, graph, normal_slots)
        elif rule_type == RuleType.ATTRIBUTE:
            return self._ground_attribute_violation(evidence, graph)
        elif rule_type == RuleType.SPATIAL:
            return self._ground_spatial_violation(evidence, graph)
        elif rule_type == RuleType.TOPOLOGICAL:
            return self._ground_topological_violation(evidence, graph)
        elif rule_type == RuleType.COMPOSITION:
            return self._ground_composition_violation(evidence, graph)
        
        # Default: use witness map if available
        if evidence.witness_map is not None:
            anomaly_map = cv2.GaussianBlur(
                evidence.witness_map, (0, 0), self.blur_sigma
            )
            return GroundingResult(
                rule_id=evidence.rule_id,
                anomaly_map=np.clip(anomaly_map, 0, 1),
                responsibility_map=anomaly_map,
                uncertainty_map=np.ones_like(anomaly_map) * 0.5,
            )
        
        return GroundingResult(
            rule_id=evidence.rule_id,
            anomaly_map=np.zeros((self.map_size, self.map_size)),
            responsibility_map=np.zeros((self.map_size, self.map_size)),
            uncertainty_map=np.ones((self.map_size, self.map_size)),
        )
    
    def _ground_count_violation(
        self,
        evidence: ViolationEvidence,
        graph: SceneGraph,
        normal_slots: Dict[int, Dict[str, Any]],
    ) -> GroundingResult:
        """Ground count violation (missing or extra objects)."""
        anomaly_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        responsibility = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        uncertainty = np.ones((self.map_size, self.map_size), dtype=np.float32) * 0.3
        
        expected = evidence.expected
        actual = evidence.actual
        target_type = evidence.details.get("target_type", "")
        
        if actual > expected:
            # Extra objects - mark them
            for comp_id in evidence.component_ids:
                if comp_id < len(graph.components):
                    comp = graph.components[comp_id]
                    mask = self._resize_mask(comp.mask)
                    anomaly_map = np.maximum(anomaly_map, mask * evidence.violation_score)
                    responsibility = np.maximum(responsibility, mask)
                    uncertainty[mask > 0] = 0.1  # High certainty for visible objects
        
        elif actual < expected and normal_slots:
            # Missing objects - mark expected slots
            missing_count = expected - actual
            for slot_id, slot_info in normal_slots.items():
                if slot_info.get("reference_type") == target_type and missing_count > 0:
                    cx, cy = slot_info["mean_position"]
                    x, y = int(cx * self.map_size), int(cy * self.map_size)
                    r = max(10, int(0.05 * self.map_size))
                    cv2.circle(anomaly_map, (x, y), r, evidence.violation_score, -1)
                    cv2.circle(responsibility, (x, y), r, 0.5, -1)
                    uncertainty[y-r:y+r, x-r:x+r] = 0.4  # Medium uncertainty
                    missing_count -= 1
        
        anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), self.blur_sigma)
        responsibility = cv2.GaussianBlur(responsibility, (0, 0), self.blur_sigma)
        
        return GroundingResult(
            rule_id=evidence.rule_id,
            anomaly_map=np.clip(anomaly_map, 0, 1),
            responsibility_map=np.clip(responsibility, 0, 1),
            uncertainty_map=np.clip(uncertainty, 0, 1),
            metadata={"type": "count", "expected": expected, "actual": actual},
        )
    
    def _ground_attribute_violation(
        self,
        evidence: ViolationEvidence,
        graph: SceneGraph,
    ) -> GroundingResult:
        """Ground attribute violation (color, shape, size)."""
        anomaly_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        responsibility = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        uncertainty = np.ones((self.map_size, self.map_size), dtype=np.float32) * 0.2
        
        for comp_id in evidence.component_ids:
            if comp_id < len(graph.components):
                comp = graph.components[comp_id]
                mask = self._resize_mask(comp.mask)
                
                # Weight by attribute deviation
                attr_name = evidence.details.get("attribute", "")
                attr_val = comp.features.get(attr_name, 0)
                expected = evidence.expected or 0
                weight = min(1.0, abs(attr_val - expected) / max(abs(expected), 1e-6))
                
                anomaly_map = np.maximum(anomaly_map, mask * weight)
                responsibility = np.maximum(responsibility, mask)
                uncertainty[mask > 0] = 0.1
        
        anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), self.blur_sigma)
        responsibility = cv2.GaussianBlur(responsibility, (0, 0), self.blur_sigma)
        
        return GroundingResult(
            rule_id=evidence.rule_id,
            anomaly_map=np.clip(anomaly_map, 0, 1),
            responsibility_map=np.clip(responsibility, 0, 1),
            uncertainty_map=np.clip(uncertainty, 0, 1),
            metadata={"type": "attribute", "attribute": evidence.details.get("attribute")},
        )
    
    def _ground_spatial_violation(
        self,
        evidence: ViolationEvidence,
        graph: SceneGraph,
    ) -> GroundingResult:
        """Ground spatial relation violation."""
        anomaly_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        responsibility = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        uncertainty = np.ones((self.map_size, self.map_size), dtype=np.float32) * 0.3
        
        if len(evidence.component_ids) >= 2:
            src_id, tgt_id = evidence.component_ids[0], evidence.component_ids[1]
            
            for comp_id in [src_id, tgt_id]:
                if comp_id < len(graph.components):
                    comp = graph.components[comp_id]
                    mask = self._resize_mask(comp.mask)
                    anomaly_map = np.maximum(anomaly_map, mask * evidence.violation_score)
                    responsibility = np.maximum(responsibility, mask)
                    uncertainty[mask > 0] = 0.2
            
            # Draw connection line showing expected vs actual
            src = graph.components[src_id]
            tgt = graph.components[tgt_id]
            x1, y1 = int(src.centroid[0] * self.map_size), int(src.centroid[1] * self.map_size)
            x2, y2 = int(tgt.centroid[0] * self.map_size), int(tgt.centroid[1] * self.map_size)
            cv2.line(anomaly_map, (x1, y1), (x2, y2), evidence.violation_score * 0.5, 2)
        
        anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), self.blur_sigma)
        responsibility = cv2.GaussianBlur(responsibility, (0, 0), self.blur_sigma)
        
        return GroundingResult(
            rule_id=evidence.rule_id,
            anomaly_map=np.clip(anomaly_map, 0, 1),
            responsibility_map=np.clip(responsibility, 0, 1),
            uncertainty_map=np.clip(uncertainty, 0, 1),
            metadata={"type": "spatial", "relation": evidence.expected},
        )
    
    def _ground_topological_violation(
        self,
        evidence: ViolationEvidence,
        graph: SceneGraph,
    ) -> GroundingResult:
        """Ground topological relation violation."""
        anomaly_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        responsibility = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        uncertainty = np.ones((self.map_size, self.map_size), dtype=np.float32) * 0.3
        
        for comp_id in evidence.component_ids:
            if comp_id < len(graph.components):
                comp = graph.components[comp_id]
                mask = self._resize_mask(comp.mask)
                anomaly_map = np.maximum(anomaly_map, mask * evidence.violation_score)
                responsibility = np.maximum(responsibility, mask)
                uncertainty[mask > 0] = 0.2
        
        anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), self.blur_sigma)
        responsibility = cv2.GaussianBlur(responsibility, (0, 0), self.blur_sigma)
        
        return GroundingResult(
            rule_id=evidence.rule_id,
            anomaly_map=np.clip(anomaly_map, 0, 1),
            responsibility_map=np.clip(responsibility, 0, 1),
            uncertainty_map=np.clip(uncertainty, 0, 1),
            metadata={"type": "topological", "relation": evidence.expected},
        )
    
    def _ground_composition_violation(
        self,
        evidence: ViolationEvidence,
        graph: SceneGraph,
    ) -> GroundingResult:
        """Ground composition violation."""
        anomaly_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        responsibility = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        uncertainty = np.ones((self.map_size, self.map_size), dtype=np.float32) * 0.4
        
        details = evidence.details
        
        # Missing required components
        for missing_type in details.get("missing_required", []):
            for comp in graph.components:
                if graph.component_types.get(id(comp), "") == missing_type:
                    mask = self._resize_mask(comp.mask)
                    anomaly_map = np.maximum(anomaly_map, mask * evidence.violation_score)
                    responsibility = np.maximum(responsibility, mask)
        
        # Forbidden combinations
        for forbidden in details.get("forbidden_violated", []):
            for comp in graph.components:
                comp_type = graph.component_types.get(id(comp), "")
                if comp_type in forbidden:
                    mask = self._resize_mask(comp.mask)
                    anomaly_map = np.maximum(anomaly_map, mask * evidence.violation_score * 0.7)
                    responsibility = np.maximum(responsibility, mask * 0.5)
        
        anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), self.blur_sigma)
        responsibility = cv2.GaussianBlur(responsibility, (0, 0), self.blur_sigma)
        
        return GroundingResult(
            rule_id=evidence.rule_id,
            anomaly_map=np.clip(anomaly_map, 0, 1),
            responsibility_map=np.clip(responsibility, 0, 1),
            uncertainty_map=np.clip(uncertainty, 0, 1),
            metadata={"type": "composition", "details": details},
        )
    
    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Resize mask to map_size."""
        if mask.shape[:2] == (self.map_size, self.map_size):
            return mask.astype(np.float32)
        return cv2.resize(
            mask.astype(np.float32),
            (self.map_size, self.map_size),
            interpolation=cv2.INTER_LINEAR,
        )


class GroundingFusion:
    """Fuse multiple grounding results into a unified logic anomaly map."""
    
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
        
        # Collect anomaly maps
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
            # Probabilistic OR: 1 - prod(1 - p_i)
            fused = 1.0 - np.prod(1.0 - maps, axis=0)
        else:
            fused = np.max(maps, axis=0)
        
        # Per-rule contribution maps
        contributions = {}
        for i, result in enumerate(grounding_results):
            if self.fusion_method in ["max", "weighted_max"]:
                # Contribution where this rule is the maximum
                max_map = np.max(maps, axis=0)
                contributions[result.rule_id] = np.where(
                    np.abs(result.anomaly_map - max_map) < 1e-6,
                    result.anomaly_map, 0.0
                )
            else:
                contributions[result.rule_id] = result.anomaly_map
        
        return np.clip(fused, 0, 1), contributions


def create_logic_anomaly_map(
    execution_result: Any,  # RuleExecutionResult
    graph: SceneGraph,
    normal_slots: Dict[int, Dict[str, Any]],
    grounder: RuleToPixelGrounder,
    fusion: GroundingFusion,
    rule_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Create unified logic anomaly map from rule execution results."""
    grounding_results = []
    
    for evidence in execution_result.violations:
        result = grounder.ground_violation(evidence, graph, normal_slots)
        grounding_results.append(result)
    
    fused_map, contributions = fusion.fuse(grounding_results, rule_weights)
    
    return fused_map, contributions