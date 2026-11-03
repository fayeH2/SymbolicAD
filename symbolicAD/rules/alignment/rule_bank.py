"""Compile an aligned, validated category-level static rule bank.

The bank keeps three separate layers of evidence:

1. semantic--visual entity alignment (``EntityBinding``);
2. canonical rule rewriting (``CanonicalRuleDecision``);
3. executable / support-validated compilation (``CompiledRuleSet``).

A rule is accepted only if all three layers succeed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..generation.ontology import CategoryOntology, ontology_for_category
from ..rule_compiler import RuleCompiler, RuleValidator
from ..rule_registry import RuleRegistry, default_rule_registry
from ..rule_schema import RuleSpec
from ..search.symbolicdet_evolution import hard_constraint_reasons
from .entity_aligner import (
    CanonicalRuleDecision, EntityBinding, align_atomic_entities,
    build_targeted_detector_vocabulary, canonicalize_rules_with_bindings,
)


@dataclass(frozen=True)
class ValidatedRuleBank:
    category: str
    ontology: CategoryOntology
    bindings: Mapping[str, EntityBinding]
    canonical_decisions: Tuple[CanonicalRuleDecision, ...]
    accepted_rules: Tuple[RuleSpec, ...]
    compile_ledger: Mapping[str, Mapping[str, Any]]
    quarantined_rules: Tuple[Mapping[str, Any], ...]
    detector_vocabulary: Mapping[str, Tuple[str, ...]]
    validation_scope: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "symbolicad-validated-rule-bank-v1",
            "category": self.category,
            "validation_scope": self.validation_scope,
            "ontology": self.ontology.to_dict(),
            "detector_vocabulary": {
                key: list(value) for key, value in self.detector_vocabulary.items()
            },
            "entity_bindings": {
                name: binding.to_dict()
                for name, binding in self.bindings.items()
            },
            "canonical_decisions": [
                decision.to_dict() for decision in self.canonical_decisions
            ],
            "accepted_rules": [rule.to_dict() for rule in self.accepted_rules],
            "compile_ledger": {
                key: dict(value) for key, value in self.compile_ledger.items()
            },
            "quarantined_rules": [dict(row) for row in self.quarantined_rules],
            "summary": {
                "accepted_rule_count": len(self.accepted_rules),
                "quarantined_rule_count": len(self.quarantined_rules),
                "aligned_entity_count": sum(
                    1 for binding in self.bindings.values() if binding.accepted
                ),
                "unresolved_entity_count": sum(
                    1 for binding in self.bindings.values()
                    if not binding.accepted
                ),
            },
        }


def build_validated_rule_bank(
    rules: Sequence[RuleSpec],
    detection_records: Mapping[str, Mapping[str, Any]],
    *,
    category: str,
    ontology: Optional[CategoryOntology] = None,
    support_contexts: Optional[Sequence[Any]] = None,
    registry: Optional[RuleRegistry] = None,
    minimum_known_fraction: float = 0.75,
) -> ValidatedRuleBank:
    selected = ontology or ontology_for_category(category)
    if selected is None:
        raise ValueError(f"no ontology registered for category {category!r}")
    registry = registry or default_rule_registry()

    bindings = align_atomic_entities(
        rules, detection_records, selected, category=category
    )
    canonical_rules, canonical_decisions = canonicalize_rules_with_bindings(
        rules, bindings
    )
    quarantined = [
        dict(decision.to_dict())
        for decision in canonical_decisions
        if decision.status != "CANONICALIZED"
    ]

    # Remove exact duplicates after canonicalization.  Qwen frequently emits the
    # same global count twice with different descriptions.
    deduped = []
    seen_signatures = {}
    for rule in canonical_rules:
        signature = (
            rule.canonical_predicate,
            tuple(rule.entity_types),
            json.dumps(
                dict(rule.parameters), sort_keys=True, ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        if signature in seen_signatures:
            quarantined.append({
                "rule_id": rule.rule_id,
                "status": "DUPLICATE_RULE",
                "reason_codes": [f"DUPLICATE_OF:{seen_signatures[signature]}"],
            })
            continue
        seen_signatures[signature] = rule.rule_id
        deduped.append(rule)
    canonical_rules = tuple(deduped)

    if support_contexts is None:
        accepted = []
        ledger: Dict[str, Dict[str, Any]] = {}
        for rule in canonical_rules:
            reasons = list(hard_constraint_reasons((rule,), registry))
            if reasons:
                ledger[rule.rule_id] = {
                    "execution_status": "UNSUPPORTED" if any(
                        reason.startswith("UNSUPPORTED_PREDICATE") for reason in reasons
                    ) else "REJECTED",
                    "reason_codes": reasons,
                }
                quarantined.append({
                    "rule_id": rule.rule_id,
                    "status": "COMPILE_REJECTED",
                    "reason_codes": reasons,
                })
            else:
                accepted.append(rule)
                ledger[rule.rule_id] = {
                    "execution_status": "EXECUTABLE",
                    "reason_codes": [],
                }
        validation_scope = "hard_constraints_only"
    else:
        compiler = RuleCompiler(
            registry=registry,
            validator=RuleValidator(minimum_known_fraction=minimum_known_fraction),
        )
        compiled = compiler.compile(canonical_rules, support_contexts)
        accepted = list(compiled.executable_rules)
        ledger = compiled.audit()
        for record in compiled.records:
            if record.disposition.value == "EXECUTABLE":
                continue
            quarantined.append({
                "rule_id": record.rule.rule_id,
                "status": f"COMPILE_{record.disposition.value}",
                "reason_codes": [record.reason_code],
                "validation_status": (
                    None if record.validation is None
                    else record.validation.status.value
                ),
                "known_support_fraction": (
                    None if record.validation is None
                    else record.validation.known_fraction
                ),
            })
        validation_scope = "normal_support_validated"

    detector_vocabulary = build_targeted_detector_vocabulary(
        rules, selected, category=category
    )
    return ValidatedRuleBank(
        category=str(category),
        ontology=selected,
        bindings=bindings,
        canonical_decisions=canonical_decisions,
        accepted_rules=tuple(accepted),
        compile_ledger=ledger,
        quarantined_rules=tuple(quarantined),
        detector_vocabulary=detector_vocabulary,
        validation_scope=validation_scope,
    )


__all__ = ["ValidatedRuleBank", "build_validated_rule_bank"]
