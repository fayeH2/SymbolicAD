"""规则在“生成 -> 编译 -> 执行 -> 定位”整条链路中的唯一数据协议。

审计要点：自然语言模型只允许产生 :class:`RuleSpec` 候选，不能直接产生
异常分数。``predicate`` 必须在 registry 中精确命中，``support_evidence`` 保存
规则来自哪些正常样本统计，``dependencies`` 则描述规则执行前必须成立的前置规则。
因此一条规则从哪里来、能否执行、为何被拒绝，都可以沿字段回溯。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Tuple


class PredicateKind(str, Enum):
    """受控谓词名称；枚举存在不等于已经具有 executor。"""
    COUNT_EQ = "count_eq"
    COUNT_IN = "count_in"
    EXISTS = "exists"
    ABOVE = "above"
    LEFT_OF = "left_of"
    ALIGNED_VERTICAL = "aligned_vertical"
    DISJOINT = "disjoint"
    LENGTH_RANGE = "length_range"
    ASPECT_RANGE = "aspect_range"
    COLOR_RANGE = "color_range"
    RELATIVE_POSITION = "relative_position"
    INSIDE = "inside"
    ONE_TO_ONE = "one_to_one"
    SLOT_OCCUPANCY = "slot_occupancy"
    ATTRIBUTE_COMPATIBLE = "attribute_compatible"


@dataclass(frozen=True)
class RuleSpec:
    """一条候选正常性约束，不携带任何 query/GT 标签信息。"""
    rule_id: str
    predicate: str
    entity_types: Tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    family: str = "unspecified"
    confidence: float = 1.0
    source: str = "support_induction"
    dependencies: Tuple[str, ...] = field(default_factory=tuple)
    grounding_strategy: str = "none"
    support_evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.rule_id).strip():
            raise ValueError("rule_id cannot be empty")
        if not str(self.predicate).strip():
            raise ValueError("predicate cannot be empty")
        if not self.entity_types or any(not str(item).strip() for item in self.entity_types):
            raise ValueError("entity_types must contain at least one non-empty type")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")

    @property
    def canonical_predicate(self) -> str:
        return str(self.predicate).strip().lower()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "predicate": self.canonical_predicate,
            "entity_types": list(self.entity_types),
            "parameters": dict(self.parameters),
            "family": self.family,
            "confidence": float(self.confidence),
            "source": self.source,
            "dependencies": list(self.dependencies),
            "grounding_strategy": self.grounding_strategy,
            "support_evidence": dict(self.support_evidence),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuleSpec":
        entity_types = payload.get("entity_types")
        if entity_types is None and payload.get("entity") is not None:
            entity_types = [payload["entity"]]
        if not isinstance(entity_types, (list, tuple)):
            raise ValueError("rule must declare entity_types")
        return cls(
            rule_id=str(payload.get("rule_id") or payload.get("id") or ""),
            predicate=str(payload.get("predicate") or ""),
            entity_types=tuple(str(item) for item in entity_types),
            parameters=dict(payload.get("parameters") or {}),
            family=str(payload.get("family") or "unspecified"),
            confidence=float(payload.get("confidence", 1.0)),
            source=str(payload.get("source") or "proposal"),
            dependencies=tuple(str(item) for item in payload.get("dependencies", [])),
            grounding_strategy=str(payload.get("grounding_strategy") or "none"),
            support_evidence=dict(payload.get("support_evidence") or {}),
        )


@dataclass(frozen=True)
class RuleSetSpec:
    category: str
    rules: Tuple[RuleSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_rules(
        cls, category: str, rules: Iterable[RuleSpec], **metadata: Any
    ) -> "RuleSetSpec":
        return cls(category=category, rules=tuple(rules), metadata=metadata)


__all__ = ["PredicateKind", "RuleSpec", "RuleSetSpec"]
