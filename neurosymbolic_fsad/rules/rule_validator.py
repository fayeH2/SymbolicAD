"""Support-only validation without treating UNKNOWN as normal."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Sequence, Tuple

from ..reasoning.violation_result import RuleStatus, ViolationResult
from .rule_registry import RuleCapability
from .rule_schema import RuleSpec

if TYPE_CHECKING:
    from ..reasoning.predicate_executor import ExecutionContext


class ValidationStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class RuleValidation:
    rule: RuleSpec
    status: ValidationStatus
    support_results: Tuple[ViolationResult, ...]
    known_fraction: float
    reason: str


class RuleValidator:
    def __init__(self, minimum_known_fraction: float = 0.75):
        self.minimum_known_fraction = float(minimum_known_fraction)

    def validate(
        self,
        rule: RuleSpec,
        capability: RuleCapability,
        support_contexts: Sequence[ExecutionContext],
    ) -> RuleValidation:
        results = tuple(
            capability.executor.execute(rule, context) for context in support_contexts
        )
        if not results:
            return RuleValidation(
                rule, ValidationStatus.INSUFFICIENT_EVIDENCE, results, 0.0,
                "no_support_contexts",
            )
        known = [row for row in results if row.status != RuleStatus.UNKNOWN]
        known_fraction = len(known) / len(results)
        if known_fraction < self.minimum_known_fraction:
            return RuleValidation(
                rule, ValidationStatus.INSUFFICIENT_EVIDENCE, results,
                known_fraction, "too_many_unknown_support_results",
            )
        if any(row.status == RuleStatus.VIOLATED for row in known):
            return RuleValidation(
                rule, ValidationStatus.REJECTED, results, known_fraction,
                "normal_support_violates_candidate",
            )
        return RuleValidation(
            rule, ValidationStatus.ACCEPTED, results, known_fraction,
            "normal_support_satisfied_candidate",
        )


__all__ = ["ValidationStatus", "RuleValidation", "RuleValidator"]
