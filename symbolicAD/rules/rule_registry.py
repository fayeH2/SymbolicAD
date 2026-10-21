"""谓词能力注册表：这里是系统“确实会执行什么”的唯一事实来源。

候选规则可以比注册表更丰富，但未登记的候选只能成为 ``UNSUPPORTED``。
禁止模糊匹配和同义替换，例如 ``length_range`` 绝不能因为未实现而退化成
``aspect_range``。这条 fail-closed 原则正是第一阶段回归测试要验证的能力边界。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Protocol, Tuple


class ExecutorProtocol(Protocol):
    def execute(self, rule: Any, context: Any) -> Any: ...


@dataclass(frozen=True)
class RuleCapability:
    """把谓词名、所需观测、执行器和定位策略绑定为一个可审计能力。"""
    predicate: str
    executor: ExecutorProtocol
    required_observations: Tuple[str, ...]
    grounding_strategy: str
    version: str = "1"


class RuleRegistry:
    def __init__(self) -> None:
        self._capabilities: Dict[str, RuleCapability] = {}

    def register(self, capability: RuleCapability) -> None:
        name = str(capability.predicate).strip().lower()
        if not name:
            raise ValueError("predicate name cannot be empty")
        if name in self._capabilities:
            raise ValueError(f"predicate already registered: {name}")
        self._capabilities[name] = capability

    def resolve(self, predicate: str) -> Optional[RuleCapability]:
        # Deliberately exact: length_range must never fall back to aspect_range.
        return self._capabilities.get(str(predicate).strip().lower())

    def predicates(self) -> Tuple[str, ...]:
        return tuple(sorted(self._capabilities))

    def capability_report(self, requested: Iterable[str]) -> Dict[str, str]:
        return {
            str(name): (
                "EXECUTABLE" if self.resolve(str(name)) is not None else "UNSUPPORTED"
            )
            for name in requested
        }


def default_rule_registry() -> RuleRegistry:
    """构造当前最小可信能力集；新增谓词必须在此显式接入执行器。"""
    from ..reasoning.predicate_executor import (
        AspectRangeExecutor,
        ColorRangeExecutor,
        CountEqExecutor,
        LengthRangeExecutor,
    )

    registry = RuleRegistry()
    registry.register(RuleCapability(
        "count_eq", CountEqExecutor(), ("instances", "observation_completeness"),
        "instance_mask_or_slot_probability",
    ))
    registry.register(RuleCapability(
        "length_range", LengthRangeExecutor(), ("instances", "length"),
        "instance_mask_endpoint_difference",
    ))
    registry.register(RuleCapability(
        "aspect_range", AspectRangeExecutor(), ("instances", "aspect_ratio"),
        "instance_mask",
    ))
    registry.register(RuleCapability(
        "color_range", ColorRangeExecutor(), ("instances", "color_h"),
        "instance_mask",
    ))
    return registry


__all__ = ["ExecutorProtocol", "RuleCapability", "RuleRegistry", "default_rule_registry"]
