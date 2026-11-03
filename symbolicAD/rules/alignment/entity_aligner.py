"""Semantic--Visual Atomic Entity Alignment (SVEA).

This module binds entity names proposed by Qwen to GroundingDINO observations.
It only decides entity identity and granularity.  It does not validate a rule
semantics and it does not invent a new rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..generation.ontology import (
    CategoryOntology, Canonicalization, ontology_for_category,
)
from ..rule_schema import RuleSpec
from .entity_extractor import AtomicEntityUse, extract_atomic_entities


ACCEPTED_STATUSES = {"ALIGNED", "ALIGNED_WITH_CANONICALIZATION"}


@dataclass(frozen=True)
class VisualCandidate:
    image_id: str
    candidate_id: str
    entity_type: str
    phrase: str
    score: float
    box_xyxy: Tuple[float, float, float, float]
    granularity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "candidate_id": self.candidate_id,
            "entity_type": self.entity_type,
            "phrase": self.phrase,
            "score": float(self.score),
            "box_xyxy": [float(v) for v in self.box_xyxy],
            "granularity": self.granularity,
        }


@dataclass(frozen=True)
class EntityBinding:
    source_name: str
    canonical_id: Optional[str]
    granularity: Optional[str]
    status: str
    reason_codes: Tuple[str, ...]
    canonicalization: Canonicalization
    candidates: Tuple[VisualCandidate, ...] = ()
    support_presence: Mapping[str, int] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status in ACCEPTED_STATUSES

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "canonical_id": self.canonical_id,
            "granularity": self.granularity,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "canonicalization": self.canonicalization.to_dict(),
            "candidate_count": self.candidate_count,
            "support_presence": dict(self.support_presence),
            "candidates": [row.to_dict() for row in self.candidates],
        }


def _candidates_for(
    detection_records: Mapping[str, Mapping[str, Any]],
    ontology: CategoryOntology,
    entity_types: Iterable[str],
) -> Tuple[VisualCandidate, ...]:
    accepted = {str(name) for name in entity_types}
    rows: List[VisualCandidate] = []
    for image_id, record in detection_records.items():
        for index, detection in enumerate(record.get("detections", ())):
            entity_type = str(detection.get("entity_type", ""))
            if entity_type not in accepted:
                continue
            box = detection.get("box_xyxy") or ()
            if len(box) != 4:
                continue
            rows.append(VisualCandidate(
                image_id=str(image_id),
                candidate_id=f"{image_id}::{index}::{entity_type}",
                entity_type=entity_type,
                phrase=str(detection.get("phrase", "")),
                score=float(detection.get("score", 0.0)),
                box_xyxy=tuple(float(v) for v in box),
                granularity=ontology.granularity(entity_type) or "unknown",
            ))
    return tuple(rows)


def _binding_for(
    use: AtomicEntityUse,
    ontology: CategoryOntology,
    detection_records: Mapping[str, Mapping[str, Any]],
) -> EntityBinding:
    canonicalization = ontology.canonicalize(use.source_name)
    if canonicalization.status == "UNRESOLVED":
        return EntityBinding(
            source_name=use.source_name,
            canonical_id=None,
            granularity=None,
            status="UNRESOLVED",
            reason_codes=(canonicalization.reason_code,),
            canonicalization=canonicalization,
        )
    if canonicalization.status == "AMBIGUOUS":
        return EntityBinding(
            source_name=use.source_name,
            canonical_id=None,
            granularity=None,
            status="AMBIGUOUS",
            reason_codes=(canonicalization.reason_code,),
            canonicalization=canonicalization,
        )

    canonical_id = str(canonicalization.canonical_id)
    if ontology.is_derived(canonical_id):
        return EntityBinding(
            source_name=use.source_name,
            canonical_id=canonical_id,
            granularity=ontology.granularity(canonical_id),
            status="ALIGNED",
            reason_codes=("DERIVED_SCENE_ENTITY",),
            canonicalization=canonicalization,
        )

    if canonical_id in ontology.supertypes:
        members = tuple(ontology.supertypes[canonical_id])
        candidates = _candidates_for(detection_records, ontology, members)
        reason = "SUPERTYPE_EXPANSION"
    else:
        candidates = _candidates_for(detection_records, ontology, (canonical_id,))
        reason = canonicalization.reason_code

    support_presence: Dict[str, int] = {}
    for candidate in candidates:
        support_presence[candidate.image_id] = (
            support_presence.get(candidate.image_id, 0) + 1
        )

    if not candidates:
        return EntityBinding(
            source_name=use.source_name,
            canonical_id=canonical_id,
            granularity=ontology.granularity(canonical_id),
            status="NOT_DETECTED",
            reason_codes=(reason, "NO_VISUAL_CANDIDATE"),
            canonicalization=canonicalization,
            support_presence=support_presence,
        )

    granularity = ontology.granularity(canonical_id)
    if granularity == "region" and len(candidates) > 1:
        return EntityBinding(
            source_name=use.source_name,
            canonical_id=canonical_id,
            granularity=granularity,
            status="AMBIGUOUS",
            reason_codes=(reason, "MULTIPLE_REGION_CANDIDATES"),
            canonicalization=canonicalization,
            candidates=candidates,
            support_presence=support_presence,
        )

    return EntityBinding(
        source_name=use.source_name,
        canonical_id=canonical_id,
        granularity=granularity,
        status=(
            "ALIGNED" if canonicalization.status == "ALIGNED"
            else "ALIGNED_WITH_CANONICALIZATION"
        ),
        reason_codes=(reason,),
        canonicalization=canonicalization,
        candidates=candidates,
        support_presence=support_presence,
    )


def align_atomic_entities(
    rules: Sequence[RuleSpec],
    detection_records: Mapping[str, Mapping[str, Any]],
    ontology: Optional[CategoryOntology] = None,
    *,
    category: Optional[str] = None,
) -> Dict[str, EntityBinding]:
    """Bind every unique rule entity to visual candidates and ontology status."""
    selected = ontology
    if selected is None:
        if category is None:
            raise ValueError("category or ontology is required")
        selected = ontology_for_category(category)
    if selected is None:
        raise ValueError(f"no ontology registered for category {category!r}")
    return {
        use.source_name: _binding_for(use, selected, detection_records)
        for use in extract_atomic_entities(rules)
    }


def build_targeted_detector_vocabulary(
    rules: Sequence[RuleSpec],
    ontology: Optional[CategoryOntology] = None,
    *,
    category: Optional[str] = None,
) -> Dict[str, Tuple[str, ...]]:
    """Build GroundingDINO prompts from the entities actually used by Qwen.

    The container entity is always included because the scene builder needs a
    coordinate frame.  If every LLM entity is unresolved, fall back to the full
    ontology so the run still produces a diagnosable detection record.
    """
    selected = ontology
    if selected is None:
        if category is None:
            raise ValueError("category or ontology is required")
        selected = ontology_for_category(category)
    if selected is None:
        raise ValueError(f"no ontology registered for category {category!r}")

    detector_ids = []
    for use in extract_atomic_entities(rules):
        canonicalization = selected.canonicalize(use.source_name)
        if not canonicalization.accepted or canonicalization.canonical_id is None:
            continue
        canonical_id = str(canonicalization.canonical_id)
        if canonical_id in selected.supertypes:
            detector_ids.extend(selected.supertypes[canonical_id])
        elif not selected.is_derived(canonical_id):
            detector_ids.append(canonical_id)

    containers = [
        spec.canonical_id for spec in selected.entities.values()
        if spec.granularity == "container" and not spec.derived
    ]
    detector_ids.extend(containers)
    if not detector_ids:
        detector_ids = [
            spec.canonical_id for spec in selected.entities.values()
            if not spec.derived
        ]
    return selected.detector_vocabulary(sorted(set(detector_ids)))


@dataclass(frozen=True)
class CanonicalRuleDecision:
    rule_id: str
    status: str
    reason_codes: Tuple[str, ...]
    original_entity_types: Tuple[str, ...]
    canonical_entity_types: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "original_entity_types": list(self.original_entity_types),
            "canonical_entity_types": list(self.canonical_entity_types),
        }


def canonicalize_rules_with_bindings(
    rules: Sequence[RuleSpec],
    bindings: Mapping[str, EntityBinding],
) -> Tuple[Tuple[RuleSpec, ...], Tuple[CanonicalRuleDecision, ...]]:
    """Replace entities by canonical IDs, quarantining unresolved rules."""
    canonical_rules: List[RuleSpec] = []
    decisions: List[CanonicalRuleDecision] = []
    for rule in rules:
        reasons = []
        canonical_ids = []
        alignment_evidence = {}
        for entity in rule.entity_types:
            binding = bindings.get(str(entity))
            if binding is None:
                reasons.append(f"MISSING_ENTITY_BINDING:{entity}")
                continue
            alignment_evidence[str(entity)] = binding.to_dict()
            if not binding.accepted or binding.canonical_id is None:
                reasons.append(f"ENTITY_{binding.status}:{entity}")
                reasons.extend(binding.reason_codes)
                continue
            canonical_ids.append(str(binding.canonical_id))
        if reasons:
            decisions.append(CanonicalRuleDecision(
                rule_id=rule.rule_id,
                status="QUARANTINED",
                reason_codes=tuple(dict.fromkeys(reasons)),
                original_entity_types=tuple(rule.entity_types),
            ))
            continue
        evidence = dict(rule.support_evidence)
        evidence["original_entity_types"] = list(rule.entity_types)
        evidence["entity_alignment"] = alignment_evidence
        canonical_rule = replace(
            rule,
            entity_types=tuple(canonical_ids),
            support_evidence=evidence,
        )
        canonical_rules.append(canonical_rule)
        decisions.append(CanonicalRuleDecision(
            rule_id=rule.rule_id,
            status="CANONICALIZED",
            reason_codes=(),
            original_entity_types=tuple(rule.entity_types),
            canonical_entity_types=tuple(canonical_ids),
        ))
    return tuple(canonical_rules), tuple(decisions)


__all__ = [
    "ACCEPTED_STATUSES", "VisualCandidate", "EntityBinding",
    "align_atomic_entities", "build_targeted_detector_vocabulary",
    "CanonicalRuleDecision", "canonicalize_rules_with_bindings",
]
