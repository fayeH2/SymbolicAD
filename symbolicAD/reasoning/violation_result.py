"""所有规则执行器必须返回的统一三态、空间化结果。

``status`` 是逻辑结论，``violation_score`` 是已确认违反的强度，
``evidence_confidence`` 是视觉事实可靠性，三者不能互相替代。特别地，
``UNKNOWN`` 不是 0.5 异常，也不是规则满足；它强制输出零逻辑分数/零图，并把
不确定性和原因单独保存。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class RuleStatus(str, Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNKNOWN = "UNKNOWN"


@dataclass
class ViolationResult:
    """规则级证据包；witness 字段让像素图能够追溯到实例、关系或缺失槽位。"""
    rule_id: str
    status: RuleStatus
    violation_score: float
    evidence_confidence: float
    witness_instances: List[str] = field(default_factory=list)
    witness_relations: List[str] = field(default_factory=list)
    missing_slots: List[str] = field(default_factory=list)
    grounding_strategy: str = "none"
    anomaly_map: Optional[np.ndarray] = None
    attribution_uncertainty: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.status = RuleStatus(self.status)
        self.violation_score = float(np.clip(self.violation_score, 0.0, 1.0))
        self.evidence_confidence = float(np.clip(self.evidence_confidence, 0.0, 1.0))
        self.attribution_uncertainty = float(
            np.clip(self.attribution_uncertainty, 0.0, 1.0)
        )
        if self.status != RuleStatus.VIOLATED:
            # UNKNOWN is not a soft anomaly. SATISFIED and UNKNOWN contribute
            # no logical anomaly score; uncertainty is carried separately.
            self.violation_score = 0.0
        if self.anomaly_map is not None:
            array = np.asarray(self.anomaly_map, dtype=np.float32)
            if array.ndim != 2:
                raise ValueError("anomaly_map must be a two-dimensional array")
            self.anomaly_map = np.clip(array, 0.0, 1.0)

    @classmethod
    def unknown(
        cls,
        rule_id: str,
        reason: str,
        *,
        evidence_confidence: float = 0.0,
        grounding_strategy: str = "none",
        map_size: int = 256,
        details: Optional[Dict[str, Any]] = None,
    ) -> "ViolationResult":
        # UNKNOWN 使用统一构造器，避免某个 executor 偷偷返回非零异常图。
        payload = dict(details or {})
        payload["unknown_reason"] = reason
        return cls(
            rule_id=rule_id,
            status=RuleStatus.UNKNOWN,
            violation_score=0.0,
            evidence_confidence=evidence_confidence,
            grounding_strategy=grounding_strategy,
            anomaly_map=np.zeros((map_size, map_size), dtype=np.float32),
            attribution_uncertainty=1.0,
            details=payload,
        )

    def to_dict(self, *, include_map: bool = False) -> Dict[str, Any]:
        result = {
            "rule_id": self.rule_id,
            "status": self.status.value,
            "violation_score": self.violation_score,
            "evidence_confidence": self.evidence_confidence,
            "witness_instances": list(self.witness_instances),
            "witness_relations": list(self.witness_relations),
            "missing_slots": list(self.missing_slots),
            "grounding_strategy": self.grounding_strategy,
            "attribution_uncertainty": self.attribution_uncertainty,
            "details": dict(self.details),
        }
        if include_map:
            result["anomaly_map"] = (
                None if self.anomaly_map is None else self.anomaly_map.tolist()
            )
        else:
            result["anomaly_map"] = None if self.anomaly_map is None else {
                "shape": list(self.anomaly_map.shape),
                "minimum": float(np.min(self.anomaly_map)),
                "maximum": float(np.max(self.anomaly_map)),
            }
        return result


__all__ = ["RuleStatus", "ViolationResult"]
