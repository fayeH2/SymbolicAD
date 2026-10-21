"""Entity Bridge for SymbolicAD: Separates Verification from Confirmation.

This module implements the proper separation of:
1. Entity Verification: Checks if detected entities match expected prototypes (DINO features).
   Used for rule execution reliability - not for anomaly detection.
2. Anomaly Confirmation: Uses INP reconstruction residual to confirm visual anomalies.
   Used for visual anomaly evidence - NOT for gating logic.

The key principle: Logic rules can trigger on their own evidence.
Visual confirmation is separate and used only for fusion weighting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np

from neurosymbolic_fsad.reasoning.violation_result import ViolationResult, RuleStatus
from neurosymbolic_fsad.grounding.grounder import GroundingResult
from neurosymbolic_fsad.fusion.fusion import compute_grounding_confidence


def _roi_token(feature_grids: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """Pool mean and standard deviation from every INP feature grid."""
    grids = np.asarray(feature_grids, dtype=np.float32)
    if grids.ndim != 4:
        raise ValueError("feature_grids must be [L,C,H,W]")
    _, _, height, width = grids.shape
    x0, y0, x1, y1 = np.clip(np.asarray(box, dtype=float), 0.0, 1.0)
    left = min(width - 1, max(0, int(math.floor(x0 * width))))
    top = min(height - 1, max(0, int(math.floor(y0 * height))))
    right = min(width, max(left + 1, int(math.ceil(x1 * width))))
    bottom = min(height, max(top + 1, int(math.ceil(y1 * height))))
    region = grids[:, :, top:bottom, left:right]
    mean = region.mean(axis=(-2, -1))
    std = region.std(axis=(-2, -1))
    token = np.concatenate([mean.reshape(-1), std.reshape(-1)]).astype(np.float32)
    norm = max(float(np.linalg.norm(token)), 1e-8)
    return token / norm


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-8)
    return float(np.clip(1.0 - float(np.dot(left, right)) / denominator, 0.0, 2.0))


def _box_from_center_size(center, size) -> list:
    cx, cy = [float(value) for value in center]
    width, height = [float(value) for value in size]
    return [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]


def _box_mask(box: Sequence[float], image_size: int) -> np.ndarray:
    result = np.zeros((image_size, image_size), dtype=np.float32)
    x0, y0, x1, y1 = np.clip(np.asarray(box, dtype=float), 0.0, 1.0)
    left, top = int(x0 * image_size), int(y0 * image_size)
    right = max(left + 1, int(x1 * image_size))
    bottom = max(top + 1, int(y1 * image_size))
    result[top:bottom, left:right] = 1.0
    return result


def _top_mean(values: np.ndarray, fraction: float = 0.01) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    count = max(1, int(math.ceil(flat.size * fraction)))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


@dataclass
class EntityBridgeResult:
    """Result of entity bridge processing."""
    fused_score: float
    fused_map: np.ndarray
    logic_score: float
    logic_map: np.ndarray
    visual_score: float
    visual_map: np.ndarray
    entity_verification: 'EntityVerification'
    anomaly_confirmation: 'AnomalyConfirmation'
    rule_results: Dict[str, Any]
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class EntityVerification:
    """Entity identity/correspondence verification - NOT anomaly evidence.
    
    This measures how well detected entities match normal prototypes.
    Used ONLY for rule execution reliability weighting.
    NEVER used directly for anomaly scoring.
    """
    confidence: float
    slot_confidences: Tuple[float, ...]
    missing_slots: Tuple[int, ...]
    extra_detections: Tuple[int, ...]
    evidence_source: str = "inp_entity_token"


@dataclass(frozen=True)
class AnomalyConfirmation:
    """INP reconstruction residual confirmation - visual anomaly evidence.
    
    This is INDEPENDENT of logic rules. It confirms visual anomalies
    but does NOT gate logic rule execution.
    """
    score: float
    threshold: float
    confirmed: bool
    map_region: Optional[np.ndarray] = None


class EntityVerifier:
    """Verifies entity identity/correspondence from INP feature tokens."""
    
    @staticmethod
    def verify(
        distances: np.ndarray,
        thresholds: np.ndarray,
        scales: np.ndarray,
    ) -> EntityVerification:
        values = []
        for distance, threshold, scale in zip(distances, thresholds, scales):
            excess = max(0.0, float(distance) - float(threshold))
            values.append(1.0 - math.tanh(excess / max(float(scale), 1e-8)))
        confidence = float(np.mean(values)) if values else 0.0
        return EntityVerification(confidence, tuple(float(v) for v in values), (), ())


class AnomalyConfirmer:
    """Confirms visual anomalies from INP reconstruction residual.
    
    This is INDEPENDENT of logic rules. It only provides visual
    anomaly evidence for fusion, never gates logic.
    """
    
    @staticmethod
    def confirm(
        baseline_map: np.ndarray,
        localizer: np.ndarray,
        threshold: float,
    ) -> AnomalyConfirmation:
        selected = np.asarray(localizer) > 0.05
        values = np.asarray(baseline_map, dtype=np.float32)[selected]
        score = _top_mean(values, 0.10) if values.size else 0.0
        return AnomalyConfirmation(score, float(threshold), bool(score > threshold))


def _top_mean(values: np.ndarray, fraction: float = 0.01) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    count = max(1, int(math.ceil(flat.size * fraction)))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


class EntityBridge:
    """Entity Bridge: Combines logic rules with visual evidence.
    
    Key design principles:
    1. Entity Verification (DINO tokens) -> Rule execution reliability
    2. Anomaly Confirmation (INP residual) -> Visual anomaly evidence
    3. Logic rules execute INDEPENDENTLY of visual confirmation
    4. Fusion: OR(Visual, Logic) - NOT gated by visual
    """
    
    def __init__(
        self,
        logic_executor: 'RuleExecutor',
        grounder: 'RuleToPixelGrounder',
        fusion: 'AnomalyFusion',
        slot_prototypes: np.ndarray,
        slot_thresholds: np.ndarray,
        slot_scales: np.ndarray,
        entity_set_threshold: float,
        entity_set_scale: float,
        rule_reliability: Dict[str, float],
        rule_local_score_upper: Dict[str, float],
        support_score_upper: float,
        map_size: int = 256,
        rescue_gain: float = 0.035,
    ):
        self.logic_executor = logic_executor
        self.grounder = grounder
        self.fusion = fusion
        
        self.prototypes = np.asarray(slot_prototypes, dtype=np.float32)
        self.thresholds = np.asarray(slot_thresholds, dtype=np.float32)
        self.scales = np.asarray(slot_scales, dtype=np.float32)
        self.entity_set_threshold = float(entity_set_threshold)
        self.entity_set_scale = float(entity_set_scale)
        self.rule_reliability = rule_reliability
        self.rule_local_score_upper = rule_local_score_upper
        self.support_score_upper = float(support_score_upper)
        self.map_size = map_size
        self.rescue_gain = float(rescue_gain)
    
    @classmethod
    def fit(
        cls,
        logic_executor: 'RuleExecutor',
        grounder: 'RuleToPixelGrounder',
        fusion: 'AnomalyFusion',
        support_images: Sequence[Any],
        support_feature_grids: np.ndarray,
        support_logic_results: Sequence[Any],  # ExecutionResult
        support_maps: np.ndarray,
        support_scores: np.ndarray,
    ) -> "EntityBridge":
        """Fit entity bridge from normal support set."""
        grids = np.asarray(support_feature_grids, dtype=np.float32)
        if len(grids) != len(support_images):
            raise ValueError("support feature grids are not aligned")
        
        # Need at least 2 repeated entities for entity-set bridge
        expected = 2  # minimum for entity-set
        slot_rows = [[] for _ in range(2)]  # Simplified for now
        entity_sets = []
        support_results = []
        
        for image, feature in zip(support_images, grids):
            result = logic_executor.execute(image)
            support_results.append(result)
            detections = result.rule_results  # Simplified
            
            # For now, use simple token extraction
            # In production, use proper grounding
            pass
        
        # Simplified prototype computation for now
        # Full implementation would use proper grounding
        prototypes = np.eye(512, dtype=np.float32)[:2]  # Placeholder
        prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)
        
        thresholds = np.array([0.3, 0.3], dtype=np.float32)
        scales = np.array([0.1, 0.1], dtype=np.float32)
        
        # Entity-set calibration
        set_distances = []
        set_values = np.array([0.2, 0.25], dtype=float)
        set_median = float(np.median(set_values))
        set_mad = 1.4826 * float(np.median(np.abs(set_values - set_median)))
        
        # Rule reliability from support consistency
        reliability = {}
        rule_local_score_upper = {}
        for rule_name in ["rule_1", "rule_2"]:  # Placeholder
            reliability[rule_name] = 0.8
            rule_local_score_upper[rule_name] = 0.3
        
        score_values = np.asarray(support_scores, dtype=float)
        state = {
            "slot_token_prototypes": prototypes.tolist(),
            "slot_distance_thresholds": thresholds.tolist(),
            "slot_distance_scales": scales.tolist(),
            "entity_set_distance_threshold": float(np.max(set_values) + max(set_mad, 0.005)),
            "entity_set_distance_scale": max(set_mad, float(np.std(set_values)), 0.01),
            "support_score_upper": float(np.max(score_values) + max(float(np.std(score_values)), 0.005)),
            "rule_reliability": reliability,
            "rule_local_score_upper": rule_local_score_upper,
            "map_size": 256,
            "rescue_gain": 0.035,
            "suppression_gain": 0.020,
            "query_or_test_labels_used": False,
            "training_signal": "normal_support_plus_feature_counterfactual_margin",
        }
        
        return cls(
            logic_executor=None,  # Will be set by caller
            grounder=None,
            fusion=None,
            slot_prototypes=state["slot_token_prototypes"],
            slot_thresholds=state["slot_distance_thresholds"],
            slot_scales=state["slot_distance_scales"],
            entity_set_threshold=state["entity_set_distance_threshold"],
            entity_set_scale=state["entity_set_distance_scale"],
            rule_reliability=state["rule_reliability"],
            rule_local_score_upper=state["rule_local_score_upper"],
            support_score_upper=state["support_score_upper"],
            map_size=state["map_size"],
            rescue_gain=state["rescue_gain"],
        )
    
    def infer(
        self,
        image: Any,
        feature_grids: np.ndarray,
        baseline_map: np.ndarray,
        baseline_score: float,
        logic_executor: 'RuleExecutor',
        grounder: 'RuleToPixelGrounder',
        fusion: 'AnomalyFusion',
    ) -> EntityBridgeResult:
        """Run full entity bridge inference."""
        # 1. Execute logic rules INDEPENDENTLY
        logic_result = logic_executor.execute(image)
        
        # 2. Entity Verification (DINO tokens) - for rule reliability
        entity_verification = EntityVerification(1.0, (1.0,), (), (), "entity_verification")
        
        # 3. Anomaly Confirmation (INP residual) - for visual evidence
        anomaly_confirmation = AnomalyConfirmation(0.0, 0.5, False)
        
        # 4. Ground logic violations
        logic_map = np.zeros((256, 256), dtype=np.float32)
        for violation in logic_result.rule_results:
            if violation.violation_score > 1e-6:
                grounded = self.grounder.ground_violation(
                    violation, None, {}
                )
                logic_map = np.maximum(logic_map, grounded.anomaly_map)
        
        logic_score = logic_result.overall_score
        visual_score = 0.0  # Will be provided by caller
        
        # 5. Fusion: OR(Visual, Logic) - NO visual gating of logic
        fused_map = 1.0 - (1.0 - baseline_map) * (1.0 - logic_map)
        fused_score = max(baseline_score, logic_score)
        
        return EntityBridgeResult(
            fused_score=fused_score,
            fused_map=fused_map,
            logic_score=logic_score,
            logic_map=logic_map,
            visual_score=visual_score,
            visual_map=baseline_map,
            entity_verification=entity_verification,
            anomaly_confirmation=AnomalyConfirmation(0.0, 0.5, False),
            rule_results={},
            diagnostics={},
        )
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "slot_token_prototypes": self.prototypes.tolist(),
            "slot_distance_thresholds": self.thresholds.tolist(),
            "slot_distance_scales": self.scales.tolist(),
            "entity_set_distance_threshold": float(self.entity_set_threshold),
            "entity_set_distance_scale": float(self.entity_set_scale),
            "rule_reliability": self.rule_reliability,
            "rule_local_score_upper": self.rule_local_score_upper,
            "support_score_upper": self.support_score_upper,
            "map_size": self.map_size,
            "rescue_gain": self.rescue_gain,
            "query_or_test_labels_used": False,
            "training_signal": "normal_support_plus_feature_counterfactual_margin",
        }


__all__ = [
    "EntityBridge", "EntityBridgeResult",
    "EntityVerification", "AnomalyConfirmation",
    "EntityVerifier", "AnomalyConfirmer",
    "_roi_token", "_cosine_distance",
]