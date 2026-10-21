"""Anomaly fusion module for combining visual and logic anomalies.

Key principle: Logic rules can detect anomalies INDEPENDENTLY of visual evidence.
Fusion combines evidence but does NOT gate logic by visual.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class FusionConfig:
    """Configuration for anomaly fusion."""
    # Visual anomaly weight
    visual_weight: float = 1.0
    
    # Logic anomaly weight
    logic_weight: float = 1.0
    
    # Fusion method: "or", "weighted_or", "weighted_sum", "max"
    # NO "grounded_and_or" - logic is NOT gated by visual
    fusion_method: str = "or"
    
    # Calibration parameters
    visual_calibration: Optional[Dict[str, float]] = None
    logic_calibration: Optional[Dict[str, float]] = None
    
    # Per-rule weights
    rule_weights: Optional[Dict[str, float]] = None
    
    # Spatial alignment
    align_maps: bool = True


@dataclass
class FusionResult:
    """Result of anomaly fusion."""
    fused_map: np.ndarray      # [H, W] fused anomaly map
    fused_score: float         # Image-level anomaly score
    visual_map: np.ndarray
    logic_map: np.ndarray
    visual_score: float
    logic_score: float
    contributions: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnomalyFusion:
    """Fuse INP-Former visual anomaly with logic-based anomaly.
    
    Key principle: Logic rules can detect anomalies INDEPENDENTLY of visual evidence.
    Fusion combines evidence but does NOT gate logic by visual.
    
    Supported methods:
    - "or": Probabilistic OR = 1 - (1-v)(1-l)
    - "weighted_or": Weighted probabilistic OR
    - "max": Element-wise maximum
    - "weighted_sum": Weighted average
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
    
    def fuse(
        self,
        visual_map: np.ndarray,
        visual_score: float,
        logic_map: np.ndarray,
        logic_score: float,
        rule_contributions: Optional[Dict[str, np.ndarray]] = None,
    ) -> FusionResult:
        """Fuse visual and logic anomaly maps.
        
        Logic is NOT gated by visual. Both contribute independently.
        """
        
        # Ensure maps are same size
        if visual_map.shape != logic_map.shape:
            logic_map = cv2.resize(
                logic_map, (visual_map.shape[1], visual_map.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Calibrate scores
        visual_score_cal = self._calibrate_score(visual_score, "visual")
        logic_score_cal = self._calibrate_score(logic_score, "logic")
        
        # Calibrate maps
        visual_map_cal = self._calibrate_map(visual_map, "visual")
        logic_map_cal = self._calibrate_map(logic_map, "logic")
        
        # NO grounding gating - logic contributes independently
        # logic_map_cal is used as-is
        
        # Fuse maps
        if self.config.fusion_method == "or":
            # Probabilistic OR
            fused_map = 1.0 - (1.0 - visual_map_cal) * (1.0 - logic_map_cal)
            fused_score = 1.0 - (1.0 - visual_score_cal) * (1.0 - logic_score_cal)
        
        elif self.config.fusion_method == "weighted_or":
            # Weighted probabilistic OR
            fused_map = 1.0 - (1.0 - self.config.visual_weight * visual_map_cal) * \
                               (1.0 - self.config.logic_weight * logic_map_cal)
            fused_score = 1.0 - (1.0 - self.config.visual_weight * visual_score_cal) * \
                               (1.0 - self.config.logic_weight * logic_score_cal)
        
        elif self.config.fusion_method == "max":
            fused_map = np.maximum(visual_map_cal, logic_map_cal)
            fused_score = max(visual_score_cal, logic_score_cal)
        
        elif self.config.fusion_method == "weighted_sum":
            total_w = self.config.visual_weight + self.config.logic_weight
            fused_map = (self.config.visual_weight * visual_map_cal + 
                         self.config.logic_weight * logic_map_cal) / total_w
            fused_score = (self.config.visual_weight * visual_score_cal + 
                           self.config.logic_weight * logic_score_cal) / total_w
        
        else:
            fused_map = visual_map_cal
            fused_score = visual_score_cal
        
        fused_map = np.clip(fused_map, 0, 1)
        fused_score = np.clip(fused_score, 0, 1)
        
        # Per-rule contributions
        contributions = {}
        if rule_contributions:
            for rule_id, rule_map in rule_contributions.items():
                if rule_map.shape != fused_map.shape:
                    rule_map = cv2.resize(
                        rule_map, (fused_map.shape[1], fused_map.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                contributions[rule_id] = rule_map
        
        return FusionResult(
            fused_map=fused_map,
            fused_score=fused_score,
            visual_map=visual_map_cal,
            logic_map=logic_map_cal,
            visual_score=visual_score_cal,
            logic_score=logic_score_cal,
            contributions=contributions,
            metadata={
                "fusion_method": self.config.fusion_method,
                "visual_weight": self.config.visual_weight,
                "logic_weight": self.config.logic_weight,
            },
        )
    
    def _calibrate_score(self, score: float, modality: str) -> float:
        """Calibrate raw score to [0, 1] probability."""
        if modality == "visual" and self.config.visual_calibration:
            cal = self.config.visual_calibration
            center = cal.get("center", 0.5)
            scale = cal.get("scale", 1.0)
            return 1.0 / (1.0 + np.exp(-(score - center) / scale))
        elif modality == "logic" and self.config.logic_calibration:
            cal = self.config.logic_calibration
            center = cal.get("center", 0.5)
            scale = cal.get("scale", 1.0)
            return 1.0 / (1.0 + np.exp(-(score - center) / scale))
        return np.clip(score, 0, 1)
    
    def _calibrate_map(self, map_: np.ndarray, modality: str) -> np.ndarray:
        """Calibrate anomaly map."""
        if modality == "visual" and self.config.visual_calibration:
            cal = self.config.visual_calibration
            center = cal.get("center", 0.5)
            scale = cal.get("scale", 1.0)
            return 1.0 / (1.0 + np.exp(-(map_ - center) / scale))
        elif modality == "logic" and self.config.logic_calibration:
            cal = self.config.logic_calibration
            center = cal.get("center", 0.5)
            scale = cal.get("scale", 1.0)
            return 1.0 / (1.0 + np.exp(-(map_ - center) / scale))
        return np.clip(map_, 0, 1)
    
    def calibrate_from_normal(
        self,
        normal_visual_maps: List[np.ndarray],
        normal_logic_maps: List[np.ndarray],
        normal_visual_scores: List[float],
        normal_logic_scores: List[float],
    ) -> "AnomalyFusion":
        """Calibrate fusion parameters from normal samples."""
        visual_scores = np.array(normal_visual_scores)
        logic_scores = np.array(normal_logic_scores)
        
        visual_center = float(np.median(visual_scores))
        visual_scale = float(np.std(visual_scores) * 1.4826)
        
        logic_center = float(np.median(logic_scores))
        logic_scale = float(np.std(logic_scores) * 1.4826)
        
        self.config.visual_calibration = {
            "center": visual_center,
            "scale": max(visual_scale, 1e-6),
        }
        self.config.logic_calibration = {
            "center": logic_center,
            "scale": max(logic_scale, 1e-6),
        }
        
        return self


class AdaptiveFusion(AnomalyFusion):
    """Adaptive fusion that learns weights from validation data."""
    
    def __init__(self, config: Optional[FusionConfig] = None):
        super().__init__(config)
        self.learned_weights: Optional[Dict[str, float]] = None
    
    def learn_weights(
        self,
        val_visual_maps: List[np.ndarray],
        val_logic_maps: List[np.ndarray],
        val_labels: List[int],
        val_masks: List[np.ndarray],
    ) -> Dict[str, float]:
        """Learn optimal fusion weights on validation set."""
        from sklearn.metrics import roc_auc_score
        from scipy.optimize import minimize
        
        def objective(weights):
            w_vis, w_log = weights
            if w_vis < 0 or w_log < 0:
                return 1e6
            
            aucs = []
            for v_map, l_map, label, mask in zip(
                val_visual_maps, val_logic_maps, val_labels, val_masks
            ):
                fused = (w_vis * v_map + w_log * l_map) / (w_vis + w_log)
                pix_auc = roc_auc_score(mask.flatten(), fused.flatten())
                aucs.append(pix_auc)
            
            return -np.mean(aucs)
        
        result = minimize(objective, [1.0, 1.0], bounds=[(0, 5), (0, 5)])
        
        if result.success:
            w_vis, w_log = result.x
            self.config.visual_weight = float(w_vis)
            self.config.logic_weight = float(w_log)
            self.learned_weights = {"visual": float(w_vis), "logic": float(w_log)}
        
        return self.learned_weights or {"visual": 1.0, "logic": 1.0}


def compute_grounding_confidence(
    logic_map: np.ndarray,
    visual_map: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """Compute grounding confidence map.
    
    This is for DIAGNOSTIC PURPOSES ONLY - not for gating logic.
    High confidence where both logic and visual anomalies agree.
    Low confidence where logic fires but visual doesn't.
    
    Returns confidence in [0, 1] per pixel.
    """
    visual_evidence = np.clip(visual_map / (threshold + 1e-6), 0, 1)
    logic_strength = logic_map
    confidence = 0.5 + 0.5 * visual_evidence * logic_strength
    return confidence


def compute_logic_only_confidence(
    logic_map: np.ndarray,
    visual_map: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """Compute confidence that logic detection is genuine (not false positive).
    
    Returns high confidence when:
    - Logic fires AND visual confirms (true positive)
    - Logic fires BUT visual doesn't (uncertain - could be logic-only anomaly)
    
    Returns low confidence when:
    - Logic fires but visual strongly contradicts (likely false positive)
    """
    visual_evidence = np.clip(visual_map / (threshold + 1e-6), 0, 1)
    logic_strength = logic_map
    
    # High confidence when both agree
    agreement = visual_evidence * logic_strength
    
    # Medium confidence when logic fires alone (logic-only anomaly)
    logic_only = logic_strength * (1.0 - visual_evidence)
    
    # Low confidence when visual contradicts logic
    contradiction = (1.0 - logic_strength) * visual_evidence
    
    confidence = 0.8 * agreement + 0.5 * logic_only + 0.1 * contradiction
    return np.clip(confidence, 0, 1)