"""Semantic--Visual Atomic Entity Alignment and validated rule-bank compilation."""

from .entity_extractor import AtomicEntityUse, extract_atomic_entities
from .entity_aligner import (
    ACCEPTED_STATUSES, EntityBinding, VisualCandidate, align_atomic_entities,
    build_targeted_detector_vocabulary, canonicalize_rules_with_bindings,
    CanonicalRuleDecision,
)
from .rule_bank import ValidatedRuleBank, build_validated_rule_bank

__all__ = [
    "AtomicEntityUse", "extract_atomic_entities",
    "ACCEPTED_STATUSES", "EntityBinding", "VisualCandidate",
    "align_atomic_entities", "build_targeted_detector_vocabulary",
    "canonicalize_rules_with_bindings", "CanonicalRuleDecision",
    "ValidatedRuleBank", "build_validated_rule_bank",
]
