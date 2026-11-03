"""只利用正常 support 对候选规则做执行性验证。

这里回答的是“该约束是否被已知的正常证据支持”，不是用人工异常拟合搜索奖励。
``UNKNOWN`` 不算满足：若可判定 support 的比例不足，规则进入
``INSUFFICIENT_EVIDENCE``；只要已知的正常 support 违反候选，规则就被拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple

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
    """在正常支持集上验证规则，同时保留每张 support 的三态结果。"""
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
        # UNKNOWN 表示当前感知证据不能判断，不能为了提高通过率把它当 SATISFIED。
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


@dataclass(frozen=True)
class EntityIdentityAudit:
    """Exact, support-only audit of the entity names referenced by one rule.

    This deliberately is not an alias resolver.  If Qwen says ``apple`` while
    the fixed perception ontology contains and observes ``peach``, the rule is
    rejected instead of silently changing its meaning.
    """

    rule_id: str
    accepted: bool
    reason_codes: Tuple[str, ...]
    entity_evidence: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "entity_evidence": {
                name: dict(row) for name, row in self.entity_evidence.items()
            },
        }


def audit_rule_entity_identity(
    rule: RuleSpec,
    support_records: Sequence[Mapping[str, Any]],
    *,
    vocabulary_entities: Sequence[str],
    supertype_members: Optional[Mapping[str, Sequence[str]]] = None,
    derived_entities: Sequence[str] = (),
    minimum_presence_fraction: float = 0.75,
) -> EntityIdentityAudit:
    """Reject unknown or unsupported entity names using exact observations.

    ``minimum_presence_fraction`` measures whether an ontology entity is
    actually observable in the declared normal supports.  Supertypes are
    observed through their explicitly declared members; derived scene entities
    (for example fixed compartments) do not require detector boxes.  No string
    similarity, synonym guessing, or LLM self-verification is performed.
    """
    threshold = float(minimum_presence_fraction)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum_presence_fraction must be in [0, 1]")
    vocabulary = {str(name) for name in vocabulary_entities}
    derived = {str(name) for name in derived_entities}
    supertypes = {
        str(name): tuple(str(member) for member in members)
        for name, members in (supertype_members or {}).items()
    }
    observed_per_support = []
    for record in support_records:
        observed_per_support.append({
            str(row.get("entity_type", ""))
            for row in record.get("detections", ())
            if str(row.get("entity_type", ""))
        })

    evidence: Dict[str, Mapping[str, Any]] = {}
    reasons = []
    # Preserve semantic arity but audit a repeated self-relation only once.
    for entity in dict.fromkeys(str(name) for name in rule.entity_types):
        if entity in derived:
            evidence[entity] = {
                "ontology_status": "DERIVED_SCENE_ENTITY",
                "observation_source": "deterministic_scene_geometry",
                "support_presence_fraction": 1.0,
                "minimum_presence_fraction": threshold,
            }
            continue
        if entity in supertypes:
            members = supertypes[entity]
            hits = sum(bool(set(members) & observed) for observed in observed_per_support)
            fraction = hits / len(observed_per_support) if observed_per_support else 0.0
            evidence[entity] = {
                "ontology_status": "DECLARED_SUPERTYPE",
                "ontology_members": list(members),
                "observation_source": "exact_member_detections",
                "support_presence_fraction": fraction,
                "minimum_presence_fraction": threshold,
            }
            if fraction < threshold:
                reasons.append(f"ENTITY_NOT_OBSERVED_ON_SUPPORT:{entity}")
            continue
        if entity not in vocabulary:
            evidence[entity] = {
                "ontology_status": "UNKNOWN_ENTITY",
                "observation_source": "none",
                "support_presence_fraction": 0.0,
                "minimum_presence_fraction": threshold,
            }
            reasons.append(f"ENTITY_NOT_IN_PERCEPTION_ONTOLOGY:{entity}")
            continue
        hits = sum(entity in observed for observed in observed_per_support)
        fraction = hits / len(observed_per_support) if observed_per_support else 0.0
        evidence[entity] = {
            "ontology_status": "EXACT_VOCABULARY_ENTITY",
            "observation_source": "exact_detector_entity_type",
            "support_presence_fraction": fraction,
            "minimum_presence_fraction": threshold,
        }
        if fraction < threshold:
            reasons.append(f"ENTITY_NOT_OBSERVED_ON_SUPPORT:{entity}")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return EntityIdentityAudit(
        rule_id=rule.rule_id,
        accepted=not unique_reasons,
        reason_codes=unique_reasons,
        entity_evidence=evidence,
    )


__all__ = [
    "ValidationStatus", "RuleValidation", "RuleValidator",
    "EntityIdentityAudit", "audit_rule_entity_identity",
]
