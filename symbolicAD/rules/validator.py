"""Structural and support-consistency validation for generated rule banks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from .schema import ALLOWED_RELATIONS, RuleKB, RuleSchemaError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from symbolicAD.grounding.support_memory import SupportMemory


@dataclass
class ValidationReport:
    issues: List[str] = field(default_factory=list)
    dropped_atoms: List[str] = field(default_factory=list)
    accepted: bool = True

    def add(self, message: str) -> None:
        self.issues.append(message)

    @property
    def ok(self) -> bool:
        return self.accepted and not self.dropped_atoms

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "ok": self.ok,
            "issues": list(self.issues),
            "dropped_atoms": list(self.dropped_atoms),
        }


def validate_rule_kb(rule_kb: RuleKB) -> tuple[RuleKB, ValidationReport]:
    """Validate a rule bank in place, zero-weighting unusable atoms.

    A broken rule must never terminate the pipeline: invalid atoms are disabled
    (weight 0) and reported, while the remaining formula still runs.
    """
    report = ValidationReport()
    component_names = {item.name for item in rule_kb.components}

    if not rule_kb.category:
        report.accepted = False
        report.add("rule bank has no category")

    if not rule_kb.components:
        report.accepted = False
        report.add("rule bank has no components")

    for symbol in rule_kb.components:
        if not symbol.all_prompts():
            symbol.prompts = [symbol.name.replace("_", " ")]
            report.add(f"component {symbol.name!r} had no prompts; default prompt added")

    seen_atoms: set[str] = set()
    atom_by_id = {}
    for atom in rule_kb.atoms:
        if atom.atom_id in seen_atoms:
            report.add(f"duplicate atom id {atom.atom_id!r}; atom disabled")
            atom.weight = 0.0
            continue
        seen_atoms.add(atom.atom_id)
        atom_by_id[atom.atom_id] = atom

        if atom.kind not in ALLOWED_RELATIONS and atom.kind != "SpatialRelation":
            report.add(f"atom {atom.atom_id!r} uses unsupported kind {atom.kind!r}")
            atom.weight = 0.0
            continue

        if atom.kind == "SpatialRelation" and atom.spatial_relation not in ALLOWED_RELATIONS:
            report.add(
                f"atom {atom.atom_id!r} uses unsupported spatial relation "
                f"{atom.spatial_relation!r}"
            )
            atom.weight = 0.0

        for component in atom.components:
            if component not in component_names:
                report.add(
                    f"atom {atom.atom_id!r} references unknown component {component!r}"
                )
                atom.weight = 0.0

        if atom.kind in {"CountEq", "CountInRange", "CountRatio"}:
            numeric = [value for value in atom.args[1:] if isinstance(value, (int, float))]
            if not numeric or any(not math.isfinite(float(value)) for value in numeric):
                report.add(f"atom {atom.atom_id!r} has no finite numeric threshold")
                atom.weight = 0.0
            elif atom.kind == "CountEq" and float(numeric[0]) < 0:
                report.add(f"atom {atom.atom_id!r} has a negative target count")
                atom.weight = 0.0
            elif atom.kind == "CountInRange":
                low, high = float(atom.args[1]), float(atom.args[2])
                if low < 0 or high < low:
                    report.add(f"atom {atom.atom_id!r} has invalid count range")
                    atom.weight = 0.0
            elif atom.kind == "CountRatio" and float(numeric[0]) < 0:
                report.add(f"atom {atom.atom_id!r} has a negative target ratio")
                atom.weight = 0.0

        if atom.kind == "AttributeMatch":
            if not atom.attribute or not atom.attribute_value:
                report.add(f"atom {atom.atom_id!r} has an empty attribute/value")
                atom.weight = 0.0
        if atom.kind == "PositionMatch" and not atom.position_region:
            report.add(f"atom {atom.atom_id!r} has an empty position region")
            atom.weight = 0.0

        atom.weight = min(1.0, max(0.0, float(atom.weight)))
        if atom.weight == 0.0:
            report.dropped_atoms.append(atom.atom_id)

    for atom_id in rule_kb.formula.atom_ids():
        if atom_id not in atom_by_id:
            report.add(f"formula references unknown atom {atom_id!r}")

    if not rule_kb.formula.atom_ids():
        report.accepted = False
        report.add("formula is empty")

    if not report.accepted:
        # The pipeline still needs an executable object.  A disabled formula is
        # represented by a single neutral AND over no leaves and is handled by
        # the executor as truth 1.0.
        from .schema import LogicNode

        rule_kb.formula = LogicNode(op="AND", children=[])
    return rule_kb, report


def validate_against_support(
    rule_kb: RuleKB,
    support_memory: "SupportMemory",
    count_tolerance: float = 0.5,
    ratio_tolerance: float = 0.5,
    disable_on_mismatch: bool = False,
    min_attribute_support_score: float = 0.05,
    min_atom_support_truth: float = 0.55,
) -> ValidationReport:
    """Report or disable count atoms that contradict normal support evidence.

    The rule compiler repairs counts from support statistics first, so the
    default here is report-only.  ``disable_on_mismatch=True`` restores the
    old, more aggressive behaviour for callers that want it.
    """
    report = ValidationReport()
    for atom in rule_kb.atoms:
        if atom.weight <= 0:
            continue
        if atom.kind == "CountEq" and atom.components:
            target = float(atom.args[1]) if len(atom.args) > 1 else 0.0
            memory = support_memory.components.get(atom.components[0])
            if memory is None:
                continue
            expected = float(memory.expected_count)
            std = float(memory.count_std)
            tolerance = max(float(count_tolerance), 0.25 * max(target, 1.0))
            if abs(expected - target) > tolerance:
                message = (
                    f"CountEq({atom.components[0]},{target:g}) does not match "
                    f"support count {expected:.2f} +/- {std:.2f}"
                )
                report.add(message)
                if disable_on_mismatch and abs(expected - target) > tolerance + std:
                    atom.weight = 0.0
                    report.dropped_atoms.append(atom.atom_id)
        elif atom.kind == "CountInRange" and atom.components:
            low = float(atom.args[1])
            high = float(atom.args[2])
            memory = support_memory.components.get(atom.components[0])
            if memory is None:
                continue
            expected = float(memory.expected_count)
            if expected < low - count_tolerance or expected > high + count_tolerance:
                message = (
                    f"CountInRange({atom.components[0]},{low:g},{high:g}) does not "
                    f"match support count {expected:.2f}"
                )
                report.add(message)
                if disable_on_mismatch:
                    atom.weight = 0.0
                    report.dropped_atoms.append(atom.atom_id)
        elif atom.kind == "CountRatio" and len(atom.components) >= 2:
            target = float(atom.args[2]) if len(atom.args) > 2 else 0.0
            mem_a = support_memory.components.get(atom.components[0])
            mem_b = support_memory.components.get(atom.components[1])
            if mem_a is None or mem_b is None:
                continue
            expected = float(mem_a.expected_count) / max(float(mem_b.expected_count), 1e-6)
            if abs(expected - target) > max(ratio_tolerance, 0.5 * max(target, 1.0)):
                message = (
                    f"CountRatio({atom.components[0]},{atom.components[1]},{target:g}) "
                    f"does not match support ratio {expected:.2f}"
                )
                report.add(message)
                if disable_on_mismatch:
                    atom.weight = 0.0
                    report.dropped_atoms.append(atom.atom_id)
        elif atom.kind == "AttributeMatch":
            stats = support_memory.attribute_stats.get(atom.atom_id, {})
            median = stats.get("q50")
            if median is None:
                message = f"{atom.expression} has no grounded support evidence"
                report.add(message)
                if disable_on_mismatch:
                    atom.weight = 0.0
                    report.dropped_atoms.append(atom.atom_id)
            elif float(median) < float(min_attribute_support_score):
                message = (
                    f"{atom.expression} support grounding score {float(median):.4g} "
                    f"is below {float(min_attribute_support_score):.4g}"
                )
                report.add(message)
                if disable_on_mismatch:
                    atom.weight = 0.0
                    report.dropped_atoms.append(atom.atom_id)

    atom_stats = getattr(support_memory, "atom_stats", {}) or {}
    for atom in rule_kb.atoms:
        if atom.weight <= 0 or atom.kind == "AttributeMatch":
            continue
        stats = atom_stats.get(atom.atom_id)
        if not stats:
            continue
        median = float(stats.get("q50", 1.0))
        support_failure = float(stats.get("support_failure_rate", 0.0))
        if median < float(min_atom_support_truth) or support_failure > 0.5:
            message = (
                f"{atom.expression} normal-support truth {median:.3f}, "
                f"leave-one-out failure rate {support_failure:.2f} "
                f"(grounding failure)"
            )
            report.add(message)
            if disable_on_mismatch:
                atom.weight = 0.0
                report.dropped_atoms.append(atom.atom_id)

    for atom_id in rule_kb.formula.atom_ids():
        atom = rule_kb.atom(atom_id)
        if atom is None or atom.weight <= 0:
            report.add(f"formula atom {atom_id!r} is disabled")
    return report
