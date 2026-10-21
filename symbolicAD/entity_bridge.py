"""连接规则证据与 INP 视觉证据，但禁止两者相互门控。

There are deliberately two independent questions:

``EntityVerifier``
    Are the detected instances and their correspondences reliable enough for a
    rule executor to make a statement?  Its confidence is passed into
    :class:`ExecutionContext`.  Low confidence therefore produces ``UNKNOWN``.

``AnomalyConfirmer``
    Does the INP reconstruction branch see unusual local appearance?  This is
    visual evidence and a diagnostic; it never changes SATISFIED/VIOLATED.

只有已知的逻辑违反贡献 logic map。融合使用单调 probabilistic OR，因此逻辑分支
可以暴露“外观正常”的结构/数量异常，也绝不会擦除 INP 已发现的热点。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .reasoning.constraint_executor import ConstraintExecution
from .reasoning.violation_result import RuleStatus


def _roi_token(feature_grids: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """Pool a normalized INP token from a normalized ``[x0,y0,x1,y1]`` ROI."""
    grids = np.asarray(feature_grids, dtype=np.float32)
    if grids.ndim != 4:
        raise ValueError("feature_grids must have shape [layers, channels, H, W]")
    _, _, height, width = grids.shape
    x0, y0, x1, y1 = np.clip(np.asarray(box, dtype=float), 0.0, 1.0)
    left = min(width - 1, max(0, int(math.floor(x0 * width))))
    top = min(height - 1, max(0, int(math.floor(y0 * height))))
    right = min(width, max(left + 1, int(math.ceil(x1 * width))))
    bottom = min(height, max(top + 1, int(math.ceil(y1 * height))))
    region = grids[:, :, top:bottom, left:right]
    token = np.concatenate([
        region.mean(axis=(-2, -1)).reshape(-1),
        region.std(axis=(-2, -1)).reshape(-1),
    ]).astype(np.float32)
    return token / max(float(np.linalg.norm(token)), 1e-8)


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-8)
    return float(np.clip(1.0 - float(np.dot(left, right)) / denominator, 0.0, 2.0))


def _top_fraction_mean(values: np.ndarray, fraction: float = 0.01) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if not flat.size:
        return 0.0
    count = max(1, int(math.ceil(flat.size * float(fraction))))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


@dataclass(frozen=True)
class EntityVerification:
    confidence: float
    instance_confidences: Tuple[float, ...]
    evidence_source: str = "inp_entity_token"

    def for_execution(self, entity_type: str) -> Dict[str, float]:
        """Build the exact mapping consumed by ``ExecutionContext``."""
        return {str(entity_type): float(np.clip(self.confidence, 0.0, 1.0))}


class EntityVerifier:
    """把原型距离变成“规则是否可执行”的可靠性，而不是异常分数。"""

    @staticmethod
    def verify(
        observed_tokens: Sequence[np.ndarray],
        prototypes: Sequence[np.ndarray],
        thresholds: Sequence[float],
        scales: Sequence[float],
    ) -> EntityVerification:
        if not (
            len(observed_tokens) == len(prototypes)
            == len(thresholds) == len(scales)
        ):
            return EntityVerification(0.0, tuple(), "incomplete_correspondence")
        confidences = []
        for token, prototype, threshold, scale in zip(
            observed_tokens, prototypes, thresholds, scales
        ):
            distance = _cosine_distance(token, prototype)
            excess = max(0.0, distance - float(threshold))
            confidences.append(
                1.0 - math.tanh(excess / max(float(scale), 1e-8))
            )
        confidence = float(np.mean(confidences)) if confidences else 0.0
        return EntityVerification(confidence, tuple(confidences))


@dataclass(frozen=True)
class AnomalyConfirmation:
    score: float
    threshold: float
    confirmed: bool
    evidence_source: str = "inp_reconstruction_residual"


class AnomalyConfirmer:
    """测量局部 INP 残差作为辅助证据，但不修改逻辑真值。"""

    @staticmethod
    def confirm(
        visual_map: np.ndarray,
        region: Optional[np.ndarray] = None,
        *,
        threshold: float,
        top_fraction: float = 0.10,
    ) -> AnomalyConfirmation:
        values = np.asarray(visual_map, dtype=np.float32)
        if region is not None:
            selected = np.asarray(region) > 0.05
            values = values[selected]
        score = _top_fraction_mean(values, top_fraction)
        return AnomalyConfirmation(score, float(threshold), score > threshold)


@dataclass(frozen=True)
class EntityBridgeResult:
    fused_score: float
    fused_map: np.ndarray
    visual_score: float
    visual_map: np.ndarray
    logic_score: float
    logic_map: np.ndarray
    logic_status: RuleStatus
    anomaly_confirmation: AnomalyConfirmation
    diagnostics: Mapping[str, object]


class EntityBridge:
    """融合已经执行完的 typed constraints 与视觉 baseline。"""

    def fuse(
        self,
        visual_map: np.ndarray,
        visual_score: float,
        logic: ConstraintExecution,
        *,
        confirmation_threshold: float = 0.5,
    ) -> EntityBridgeResult:
        visual = np.clip(np.asarray(visual_map, dtype=np.float32), 0.0, 1.0)
        logic_map = np.asarray(logic.anomaly_map, dtype=np.float32)
        if logic_map.shape != visual.shape:
            logic_map = cv2.resize(
                logic_map, visual.shape[::-1], interpolation=cv2.INTER_LINEAR
            )
        # UNKNOWN is neither satisfied nor a soft anomaly.  The constraint
        # executor already guarantees a zero map, and this check fails closed
        # if a malformed external result violates that contract.
        if logic.status != RuleStatus.VIOLATED:
            logic_map = np.zeros_like(visual)
            logic_score = 0.0
        else:
            logic_map = np.clip(logic_map, 0.0, 1.0)
            logic_score = float(np.clip(logic.violation_score, 0.0, 1.0))
        confirmation = AnomalyConfirmer.confirm(
            visual, logic_map if np.any(logic_map) else None,
            threshold=confirmation_threshold,
        )
        fused = (
            visual.copy() if not np.any(logic_map)
            else 1.0 - (1.0 - visual) * (1.0 - logic_map)
        )
        fused_score = 1.0 - (
            (1.0 - float(np.clip(visual_score, 0.0, 1.0)))
            * (1.0 - logic_score)
        )
        return EntityBridgeResult(
            fused_score=float(fused_score),
            fused_map=fused.astype(np.float32),
            visual_score=float(np.clip(visual_score, 0.0, 1.0)),
            visual_map=visual,
            logic_score=logic_score,
            logic_map=logic_map,
            logic_status=logic.status,
            anomaly_confirmation=confirmation,
            diagnostics={
                "logic_truth_depends_on_inp_residual": False,
                "fusion_is_monotone": True,
                "unknown_contribution": 0.0,
            },
        )


__all__ = [
    "EntityVerification", "EntityVerifier", "AnomalyConfirmation",
    "AnomalyConfirmer", "EntityBridgeResult", "EntityBridge",
    "_roi_token", "_cosine_distance",
]
