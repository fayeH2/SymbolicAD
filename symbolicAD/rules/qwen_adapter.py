"""Conservative adapter from Qwen's proposal schema to the typed rule core.

Qwen proposes semantics; it does not grant execution capability.  The adapter
therefore never coerces an unimplemented meaning into the nearest implemented
predicate: an unsupported proposal keeps its own predicate name and stays
``UNSUPPORTED`` in the admission audit.

Two ambiguities are handled explicitly rather than papered over:

``COUNT(entity, region)``
    A two-argument count is ``count_in`` (container-scoped).  It is never
    widened into the global ``count_eq``.

``COUNT`` whose text only asserts presence
    Qwen writes "Granola is present in right compartment" as ``COUNT=1``.
    Exact-count semantics would also reject a normal scene containing two
    almonds.  When the description is unambiguously about presence, and the
    expected count/tolerance are ``1``/``0``, the adapter emits ``exists``
    instead and records the rewrite.  Widening tolerance is *not* used.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .rule_schema import RuleSpec


class QwenAdaptationStatus(str, Enum):
    ADAPTED = "ADAPTED"
    PRESERVED_UNSUPPORTED = "PRESERVED_UNSUPPORTED"
    INVALID = "INVALID"


@dataclass(frozen=True)
class QwenAdaptation:
    source_rule_id: str
    source_predicate: str
    status: QwenAdaptationStatus
    reason_code: str
    rule: Optional[RuleSpec]


_RANGE_PREDICATES = {
    "LENGTH": ("length_range", "shape", "instance_mask_endpoint_difference"),
    "ASPECT_RATIO": ("aspect_range", "shape", "instance_mask"),
    "COLOR_H": ("color_range", "appearance", "instance_mask"),
}

# Predicates whose typed name and parameter semantics are already exact.
_RELATION_PREDICATES = {
    "ABOVE": "above",
    "LEFT_OF": "left_of",
    "ALIGNED_VERTICAL": "aligned_vertical",
    "DISJOINT": "disjoint",
}

_PRESENCE_MARKERS = (
    "is present", "are present", "presence", "exists", "is absent", "are absent",
    "is missing", "are missing",
)


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _entity_types(arguments: Mapping[str, Any]) -> Tuple[str, ...]:
    values = []
    for key in ("entity_type", "target_entity_type"):
        value = str(arguments.get(key, "")).strip()
        if value and value not in values:
            values.append(value)
    return tuple(values)


def _entity_pair(arguments: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """Raw ``(source, target)`` pair; self-relations keep both positions.

    ``ALIGNED_VERTICAL(orange, orange)`` means "the two oranges are aligned",
    so de-duplicating the pair would silently turn a binary relation into an
    arity-1 predicate.
    """
    source = str(arguments.get("entity_type", "")).strip()
    target = str(arguments.get("target_entity_type", "")).strip()
    if not source or not target:
        return None
    return source, target


def _is_presence_claim(description: str) -> bool:
    lowered = description.lower()
    return any(marker in lowered for marker in _PRESENCE_MARKERS)


def adapt_qwen_rule(payload: Mapping[str, Any]) -> QwenAdaptation:
    """Adapt one proposal without weakening or broadening its semantics."""
    rule_id = str(payload.get("rule_id", "")).strip()
    predicate = str(payload.get("predicate_name", "")).strip().upper()
    arguments = payload.get("predicate_args") or {}
    parameters = payload.get("predicate_params") or {}
    if (
        not rule_id or not predicate
        or not isinstance(arguments, Mapping)
        or not isinstance(parameters, Mapping)
    ):
        return QwenAdaptation(
            rule_id, predicate, QwenAdaptationStatus.INVALID,
            "MALFORMED_QWEN_RULE", None,
        )
    entities = _entity_types(arguments)
    if not entities:
        return QwenAdaptation(
            rule_id, predicate, QwenAdaptationStatus.INVALID,
            "MISSING_ENTITY_TYPES", None,
        )
    description = str(payload.get("description", ""))

    common = {
        "rule_id": rule_id,
        "entity_types": entities,
        "confidence": float(np.clip(
            _finite_number(payload.get("confidence")) or 0.0, 0.0, 1.0
        )),
        "source": "qwen_proposal",
        "support_evidence": {
            "description": description,
            "qwen_predicate": predicate,
            "qwen_arguments": dict(arguments),
            "qwen_parameters": dict(parameters),
            "qwen_support_evidence": list(payload.get("support_evidence") or []),
        },
    }

    if predicate == "COUNT":
        expected = _finite_number(parameters.get("expected_count"))
        tolerance = _finite_number(parameters.get("tolerance", 0))
        if (
            expected is None or tolerance is None
            or expected < 0 or tolerance < 0
            or int(expected) != expected or int(tolerance) != tolerance
        ):
            return QwenAdaptation(
                rule_id, predicate, QwenAdaptationStatus.INVALID,
                "COUNT_PARAMETERS_NOT_NUMERIC", None,
            )
        scoped = "target_entity_type" in arguments
        presence_claim = (
            scoped
            and _is_presence_claim(description)
            and int(expected) == 1 and int(tolerance) == 0
        )
        if presence_claim:
            rule = RuleSpec(
                predicate="exists",
                parameters={"minimum_count": 1},
                family="cardinality", grounding_strategy="presence_region_mask",
                **common,
            )
            return QwenAdaptation(
                rule_id, predicate, QwenAdaptationStatus.ADAPTED,
                "PRESENCE_CLAIM_MAPPED_TO_EXISTS_NOT_EXACT_COUNT", rule,
            )
        if scoped:
            rule = RuleSpec(
                predicate="count_in",
                parameters={"expected": expected, "tolerance": tolerance},
                family="cardinality", grounding_strategy="region_mask_and_instance_witness",
                **common,
            )
            return QwenAdaptation(
                rule_id, predicate, QwenAdaptationStatus.ADAPTED,
                "SCOPED_COUNT_SEMANTICS_PRESERVED", rule,
            )
        rule = RuleSpec(
            predicate="count_eq",
            parameters={"expected": expected, "tolerance": tolerance},
            family="cardinality", grounding_strategy="instance_mask_or_slot_probability",
            **common,
        )
        return QwenAdaptation(
            rule_id, predicate, QwenAdaptationStatus.ADAPTED,
            "GLOBAL_COUNT_SEMANTICS_PRESERVED", rule,
        )

    if predicate == "EXISTS":
        minimum = _finite_number(parameters.get("minimum_count", 1))
        if minimum is None or minimum < 1 or int(minimum) != minimum:
            minimum = 1.0
        rule = RuleSpec(
            predicate="exists", parameters={"minimum_count": int(minimum)},
            family="cardinality", grounding_strategy="presence_region_mask", **common,
        )
        return QwenAdaptation(
            rule_id, predicate, QwenAdaptationStatus.ADAPTED,
            "EXISTS_SEMANTICS_PRESERVED", rule,
        )

    if predicate in _RANGE_PREDICATES:
        expected = _finite_number(parameters.get("expected_value"))
        tolerance = _finite_number(parameters.get("tolerance"))
        if expected is None or tolerance is None or tolerance < 0:
            return QwenAdaptation(
                rule_id, predicate, QwenAdaptationStatus.INVALID,
                "RANGE_PARAMETERS_NOT_EXECUTABLE", None,
            )
        typed_predicate, family, grounding = _RANGE_PREDICATES[predicate]
        rule = RuleSpec(
            predicate=typed_predicate,
            parameters={"lower": expected - tolerance, "upper": expected + tolerance},
            family=family, grounding_strategy=grounding, **common,
        )
        return QwenAdaptation(
            rule_id, predicate, QwenAdaptationStatus.ADAPTED,
            "NUMERIC_RANGE_SEMANTICS_PRESERVED", rule,
        )

    if predicate in _RELATION_PREDICATES:
        pair = _entity_pair(arguments)
        if pair is None:
            return QwenAdaptation(
                rule_id, predicate, QwenAdaptationStatus.INVALID,
                "RELATION_REQUIRES_SOURCE_AND_TARGET", None,
            )
        # DISJOINT carries no Qwen threshold; the executor compares pixel
        # overlap and needs an explicit numerical tolerance, not a semantic
        # relaxation.  Record that it was introduced by the adapter.
        default_threshold = 0.01 if predicate == "DISJOINT" else None
        threshold = _finite_number(parameters.get("threshold", default_threshold))
        if threshold is None or threshold < 0:
            return QwenAdaptation(
                rule_id, predicate, QwenAdaptationStatus.INVALID,
                "RELATION_THRESHOLD_NOT_EXECUTABLE", None,
            )
        relation_common = dict(common)
        relation_common["entity_types"] = pair
        rule = RuleSpec(
            predicate=_RELATION_PREDICATES[predicate],
            parameters={"threshold": threshold},
            family="spatial" if predicate != "DISJOINT" else "topological",
            grounding_strategy="relation_centroid", **relation_common,
        )
        reason = (
            "RELATION_SEMANTICS_PRESERVED"
            if "threshold" in parameters
            else "RELATION_SEMANTICS_PRESERVED_WITH_EXPLICIT_OVERLAP_TOLERANCE"
        )
        return QwenAdaptation(
            rule_id, predicate, QwenAdaptationStatus.ADAPTED, reason, rule,
        )

    # Preserve every other meaning verbatim.  It stays UNSUPPORTED unless an
    # exact executor is later registered under this exact name.
    rule = RuleSpec(
        predicate=predicate.lower(), parameters=dict(parameters),
        family=str(payload.get("rule_type") or "qwen"),
        grounding_strategy=str(payload.get("grounding_strategy") or "none"),
        **common,
    )
    return QwenAdaptation(
        rule_id, predicate, QwenAdaptationStatus.PRESERVED_UNSUPPORTED,
        "NO_EXACT_TYPED_ADAPTER", rule,
    )


def adapt_qwen_candidates(
    payloads: Sequence[Mapping[str, Any]],
) -> Tuple[QwenAdaptation, ...]:
    return tuple(adapt_qwen_rule(row) for row in payloads)


def payloads_from_audit(report: Mapping[str, Any]) -> Tuple[dict, ...]:
    """Reconstruct the original Qwen payloads recorded in an admission audit.

    ``audit.json`` stores the untranslated ``qwen_arguments`` /
    ``qwen_parameters`` inside each candidate's ``support_evidence``, so the
    candidate ledger can be rebuilt exactly without the raw LLM response.
    """
    payloads = []
    for row in report.get("candidates", []):
        typed = row.get("typed_rule") or {}
        evidence = typed.get("support_evidence") or {}
        payloads.append({
            "rule_id": row.get("rule_id", ""),
            "description": row.get("description", ""),
            "rule_type": typed.get("family") or "qwen",
            "predicate_name": evidence.get("qwen_predicate", row.get("source_predicate", "")),
            "predicate_args": dict(evidence.get("qwen_arguments") or {}),
            "predicate_params": dict(evidence.get("qwen_parameters") or {}),
            "confidence": typed.get("confidence", 1.0),
            "support_evidence": list(evidence.get("qwen_support_evidence") or []),
        })
    return tuple(payloads)


__all__ = [
    "QwenAdaptationStatus", "QwenAdaptation", "adapt_qwen_rule",
    "adapt_qwen_candidates", "payloads_from_audit",
]
