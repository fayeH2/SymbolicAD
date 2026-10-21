"""Support-only rule induction and auditable compilation.

The compiler never substitutes one predicate for another.  In particular, an
explicit length proposal either reaches the length executor or remains visible
as unsupported; it can never silently become an aspect or count rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .rule_registry import RuleCapability, RuleRegistry, default_rule_registry
from .rule_schema import RuleSpec
from .rule_validator import RuleValidation, RuleValidator, ValidationStatus

if TYPE_CHECKING:
    from ..reasoning.predicate_executor import ExecutionContext


class CompileDisposition(str, Enum):
    EXECUTABLE = "EXECUTABLE"
    UNSUPPORTED = "UNSUPPORTED"
    REJECTED = "REJECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class CompileRecord:
    rule: RuleSpec
    disposition: CompileDisposition
    reason_code: str
    capability: Optional[RuleCapability] = None
    validation: Optional[RuleValidation] = None


@dataclass(frozen=True)
class CompiledRuleSet:
    records: Tuple[CompileRecord, ...]
    substitutions: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def executable_records(self) -> Tuple[CompileRecord, ...]:
        return tuple(
            row for row in self.records
            if row.disposition == CompileDisposition.EXECUTABLE
        )

    @property
    def executable_rules(self) -> Tuple[RuleSpec, ...]:
        return tuple(row.rule for row in self.executable_records)

    def by_rule_id(self) -> Dict[str, CompileRecord]:
        return {row.rule.rule_id: row for row in self.records}

    def audit(self) -> Dict[str, Dict[str, Any]]:
        return {
            row.rule.rule_id: {
                "predicate": row.rule.canonical_predicate,
                "execution_status": row.disposition.value,
                "reason_code": row.reason_code,
                "validation_status": (
                    None if row.validation is None else row.validation.status.value
                ),
                "known_support_fraction": (
                    None if row.validation is None else row.validation.known_fraction
                ),
            }
            for row in self.records
        }


class RuleCompiler:
    def __init__(
        self,
        registry: Optional[RuleRegistry] = None,
        validator: Optional[RuleValidator] = None,
    ) -> None:
        self.registry = registry or default_rule_registry()
        self.validator = validator or RuleValidator()

    def compile(
        self,
        candidates: Iterable[RuleSpec],
        support_contexts: Sequence[ExecutionContext],
        *,
        validate_support: bool = True,
    ) -> CompiledRuleSet:
        records = []
        seen_ids = set()
        for rule in candidates:
            if rule.rule_id in seen_ids:
                records.append(CompileRecord(
                    rule, CompileDisposition.REJECTED, "DUPLICATE_RULE_ID"
                ))
                continue
            seen_ids.add(rule.rule_id)
            capability = self.registry.resolve(rule.canonical_predicate)
            if capability is None:
                records.append(CompileRecord(
                    rule, CompileDisposition.UNSUPPORTED,
                    "NO_REGISTERED_EXECUTOR",
                ))
                continue
            if not validate_support:
                records.append(CompileRecord(
                    rule, CompileDisposition.EXECUTABLE, "REGISTERED_EXECUTOR",
                    capability=capability,
                ))
                continue
            validation = self.validator.validate(
                rule, capability, support_contexts
            )
            if validation.status == ValidationStatus.ACCEPTED:
                disposition = CompileDisposition.EXECUTABLE
                reason = "VALIDATED_ON_NORMAL_SUPPORT"
            elif validation.status == ValidationStatus.INSUFFICIENT_EVIDENCE:
                disposition = CompileDisposition.INSUFFICIENT_EVIDENCE
                reason = validation.reason.upper()
            else:
                disposition = CompileDisposition.REJECTED
                reason = validation.reason.upper()
            records.append(CompileRecord(
                rule, disposition, reason,
                capability=capability, validation=validation,
            ))
        return CompiledRuleSet(tuple(records))


@dataclass(frozen=True)
class InductionDecision:
    predicate: str
    entity_type: str
    proposed: bool
    reason_code: str
    rule: Optional[RuleSpec]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InductionResult:
    decisions: Tuple[InductionDecision, ...]

    @property
    def proposed_rules(self) -> Tuple[RuleSpec, ...]:
        return tuple(row.rule for row in self.decisions if row.rule is not None)

    @property
    def structure_signature(self) -> Tuple[str, ...]:
        return tuple(sorted(rule.canonical_predicate for rule in self.proposed_rules))


class SupportRuleInducer:
    """Find stable normal constraints without category-specific branches."""

    ATTRIBUTE_PREDICATES = {
        "length_range": "length",
        "aspect_range": "aspect_ratio",
        "color_range": "color_h",
    }

    def __init__(
        self,
        *,
        maximum_relative_span: float = 0.08,
        minimum_scenes: int = 3,
        range_margin_fraction: float = 0.02,
    ) -> None:
        self.maximum_relative_span = float(maximum_relative_span)
        self.minimum_scenes = int(minimum_scenes)
        self.range_margin_fraction = float(range_margin_fraction)

    @staticmethod
    def _entity_types(contexts: Sequence[ExecutionContext]) -> Tuple[str, ...]:
        values = {
            context.scene.component_type(index)
            for context in contexts
            for index in range(len(context.scene.components))
            if context.scene.component_type(index)
        }
        return tuple(sorted(values))

    @staticmethod
    def _relative_span(values: Sequence[float]) -> float:
        array = np.asarray(values, dtype=float)
        center = float(np.median(array))
        return float(np.ptp(array) / max(abs(center), 1e-6))

    def _attribute_decision(
        self,
        predicate: str,
        attribute: str,
        entity_type: str,
        contexts: Sequence[ExecutionContext],
    ) -> InductionDecision:
        scene_values = []
        flattened = []
        for context in contexts:
            values = []
            for instance in context.scene.instances_of(entity_type):
                value = getattr(instance, "attributes", {}).get(attribute)
                if value is not None and np.isfinite(float(value)):
                    values.append(float(value))
            if not values:
                return InductionDecision(
                    predicate, entity_type, False, "MISSING_SUPPORT_OBSERVATION",
                    None, {"attribute": attribute},
                )
            scene_values.append(float(np.median(values)))
            flattened.extend(values)
        span = self._relative_span(flattened)
        diagnostics = {
            "attribute": attribute,
            "relative_span": span,
            "scene_medians": scene_values,
            "minimum": float(np.min(flattened)),
            "maximum": float(np.max(flattened)),
        }
        if span > self.maximum_relative_span:
            return InductionDecision(
                predicate, entity_type, False, "UNSTABLE_ON_SUPPORT",
                None, diagnostics,
            )
        lower, upper = float(np.min(flattened)), float(np.max(flattened))
        margin = max(
            (upper - lower) * 0.5,
            abs(float(np.median(flattened))) * self.range_margin_fraction,
            1e-4,
        )
        rule = RuleSpec(
            rule_id=f"auto_{entity_type}_{predicate}",
            predicate=predicate,
            entity_types=(entity_type,),
            parameters={"lower": lower - margin, "upper": upper + margin},
            family="shape" if attribute != "color_h" else "appearance",
            source="support_induction",
            grounding_strategy=(
                "instance_mask_endpoint_difference"
                if predicate == "length_range" else "instance_mask"
            ),
            support_evidence=diagnostics,
        )
        return InductionDecision(
            predicate, entity_type, True, "STABLE_ON_SUPPORT", rule, diagnostics
        )

    def _count_decision(
        self,
        entity_type: str,
        contexts: Sequence[ExecutionContext],
    ) -> InductionDecision:
        counts = [len(context.scene.instances_of(entity_type)) for context in contexts]
        diagnostics = {"counts": counts}
        if len(set(counts)) != 1:
            return InductionDecision(
                "count_eq", entity_type, False, "UNSTABLE_ON_SUPPORT",
                None, diagnostics,
            )
        rule = RuleSpec(
            rule_id=f"auto_{entity_type}_count_eq",
            predicate="count_eq", entity_types=(entity_type,),
            parameters={"expected": counts[0]}, family="cardinality",
            source="support_induction",
            grounding_strategy="instance_mask_or_slot_probability",
            support_evidence=diagnostics,
        )
        return InductionDecision(
            "count_eq", entity_type, True, "STABLE_ON_SUPPORT", rule, diagnostics
        )

    def induce(self, contexts: Sequence[ExecutionContext]) -> InductionResult:
        if len(contexts) < self.minimum_scenes:
            return InductionResult(tuple())
        decisions = []
        for entity_type in self._entity_types(contexts):
            decisions.append(self._count_decision(entity_type, contexts))
            for predicate, attribute in self.ATTRIBUTE_PREDICATES.items():
                decisions.append(self._attribute_decision(
                    predicate, attribute, entity_type, contexts
                ))
        return InductionResult(tuple(decisions))


__all__ = [
    "CompileDisposition", "CompileRecord", "CompiledRuleSet", "RuleCompiler",
    "InductionDecision", "InductionResult", "SupportRuleInducer",
]
