"""视觉/逻辑证据的单调融合。

No method in this file multiplies a rule violation by the INP residual.  The
logical map must already come from a known ``VIOLATED`` result.  ``UNKNOWN``
and ``SATISFIED`` are forced to zero logical contribution. 此 API 刻意不接收
验证标签或 GT mask，因此不能在测试集上自适应调权。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np

from ..reasoning.violation_result import RuleStatus


@dataclass(frozen=True)
class FusionConfig:
    method: str = "probabilistic_or"  # probabilistic_or | maximum

    def __post_init__(self) -> None:
        if self.method not in {"probabilistic_or", "maximum"}:
            raise ValueError("fusion method must be probabilistic_or or maximum")


@dataclass(frozen=True)
class FusionResult:
    fused_map: np.ndarray
    fused_score: float
    visual_map: np.ndarray
    logic_map: np.ndarray
    visual_score: float
    logic_score: float
    logical_status: RuleStatus
    logical_contribution: float
    rule_contributions: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AnomalyFusion:
    """合并相互独立的证据，且任何一个输入热点都不会被另一分支压低。"""

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()

    def fuse(
        self,
        visual_map: np.ndarray,
        visual_score: float,
        logic_map: np.ndarray,
        logic_score: float,
        *,
        logical_status: RuleStatus,
        rule_contributions: Optional[Mapping[str, np.ndarray]] = None,
    ) -> FusionResult:
        visual = np.clip(np.asarray(visual_map, dtype=np.float32), 0.0, 1.0)
        logic = np.asarray(logic_map, dtype=np.float32)
        if logic.shape != visual.shape:
            logic = cv2.resize(logic, visual.shape[::-1], interpolation=cv2.INTER_LINEAR)

        # 三态语义边界：UNKNOWN 不等于 0.5 异常，也不等于“规则满足”。
        status = RuleStatus(logical_status)
        if status != RuleStatus.VIOLATED:
            logic = np.zeros_like(visual)
            logic_score = 0.0
        else:
            logic = np.clip(logic, 0.0, 1.0)
            logic_score = float(np.clip(logic_score, 0.0, 1.0))

        if not np.any(logic):
            fused = visual.copy()  # 保证 UNKNOWN 路径逐位保持 baseline。
        elif self.config.method == "maximum":
            fused = np.maximum(visual, logic)
        else:
            fused = 1.0 - (1.0 - visual) * (1.0 - logic)
        visual_score = float(np.clip(visual_score, 0.0, 1.0))
        fused_score = (
            max(visual_score, logic_score)
            if self.config.method == "maximum"
            else 1.0 - (1.0 - visual_score) * (1.0 - logic_score)
        )
        contributions = {}
        for rule_id, rule_map in (rule_contributions or {}).items():
            value = np.asarray(rule_map, dtype=np.float32)
            if value.shape != visual.shape:
                value = cv2.resize(value, visual.shape[::-1], interpolation=cv2.INTER_LINEAR)
            contributions[str(rule_id)] = np.clip(value, 0.0, 1.0)
        return FusionResult(
            fused.astype(np.float32), float(fused_score), visual, logic,
            visual_score, logic_score, status, logic_score,
            contributions, {"method": self.config.method, "visual_gates_logic": False},
        )


__all__ = ["FusionConfig", "FusionResult", "AnomalyFusion"]
