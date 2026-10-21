"""Explicit predicate capability registry. No implicit predicate aliases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Protocol, Tuple


class ExecutorProtocol(Protocol):
    def execute(self, rule: Any, context: Any) -> Any: ...


@dataclass(frozen=True)
class RuleCapability:
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
