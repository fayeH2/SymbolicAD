"""Dependency-aware, de-duplicating greedy rule-set baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .rule_compiler import CompiledRuleSet
from .rule_schema import RuleSpec


class RuleDependencyCycle(ValueError):
    pass


class UnsatisfiedRuleDependency(ValueError):
    pass


class RuleDependencyGraph:
    def __init__(self, rules: Iterable[RuleSpec]):
        self.dependencies = {rule.rule_id: tuple(rule.dependencies) for rule in rules}
        self._validate_references()
        self._validate_acyclic()

    def _validate_references(self) -> None:
        ids = set(self.dependencies)
        for rule_id, dependencies in self.dependencies.items():
            missing = set(dependencies) - ids
            if missing:
                raise UnsatisfiedRuleDependency(
                    f"{rule_id} has unavailable dependencies: {sorted(missing)}"
                )

    def _validate_acyclic(self) -> None:
        active, complete = set(), set()

        def visit(rule_id: str) -> None:
            if rule_id in active:
                raise RuleDependencyCycle(f"dependency cycle contains {rule_id}")
            if rule_id in complete:
                return
            active.add(rule_id)
            for dependency in self.dependencies[rule_id]:
                visit(dependency)
            active.remove(rule_id)
            complete.add(rule_id)

        for rule_id in self.dependencies:
            visit(rule_id)

    def closure(self, rule_id: str) -> Set[str]:
        result = {rule_id}
        for dependency in self.dependencies[rule_id]:
            result.update(self.closure(dependency))
        return result

    def is_closed(self, selected: Iterable[str]) -> bool:
        values = set(selected)
        return all(set(deps).issubset(values) for rule_id, deps in self.dependencies.items()
                   if rule_id in values)

    def validate(self, selected: Iterable[str]) -> None:
        if not self.is_closed(selected):
            raise UnsatisfiedRuleDependency("selected rule set is not dependency-closed")


@dataclass(frozen=True)
class RuleOptimizationReport:
    selected_rules: Tuple[RuleSpec, ...]
    unique_response_pattern_count: int
    diagnostic_intervention_coverage: Optional[float]
    real_anomaly_coverage: None = None
    method: str = "greedy_dependency_dedup"

    @property
    def selected_ids(self) -> Tuple[str, ...]:
        return tuple(rule.rule_id for rule in self.selected_rules)


class GreedyRuleSetOptimizer:
    """Simple baseline; diagnostic interventions are not real anomaly labels."""

    def select(
        self,
        compiled: CompiledRuleSet,
        *,
        budget: int,
        diagnostic_responses: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> RuleOptimizationReport:
        rules = tuple(record.rule for record in compiled.executable_records)
        graph = RuleDependencyGraph(rules)
        by_id = {rule.rule_id: rule for rule in rules}
        responses = {
            rule_id: np.asarray(values, dtype=float) > 0
            for rule_id, values in (diagnostic_responses or {}).items()
            if rule_id in by_id
        }
        selected: Set[str] = set()
        covered = np.zeros(
            max((len(row) for row in responses.values()), default=0), dtype=bool
        )
        seen_patterns = set()
        while len(selected) < int(budget):
            best = None
            best_key = None
            for rule in rules:
                if rule.rule_id in selected:
                    continue
                pattern = tuple(responses.get(rule.rule_id, np.asarray([], dtype=bool)))
                if pattern and pattern in seen_patterns:
                    continue
                addition = graph.closure(rule.rule_id) - selected
                if len(selected | addition) > int(budget):
                    continue
                response = responses.get(rule.rule_id)
                marginal = int(np.count_nonzero(response & ~covered)) if response is not None else 0
                family_gain = int(rule.family not in {by_id[item].family for item in selected})
                # Stable input order is the final tie-breaker; this prevents a
                # later textual alias from replacing the first canonical rule.
                key = (marginal, family_gain, float(rule.confidence), -len(addition))
                if best_key is None or key > best_key:
                    best, best_key = rule, key
            if best is None:
                break
            addition = graph.closure(best.rule_id) - selected
            for rule_id in sorted(addition):
                selected.add(rule_id)
                response = responses.get(rule_id)
                if response is not None:
                    covered |= response
                    seen_patterns.add(tuple(response))
        graph.validate(selected)
        ordered = tuple(rule for rule in rules if rule.rule_id in selected)
        total = covered.size
        coverage = float(np.mean(covered)) if total else None
        unique = len({
            tuple(responses[rule.rule_id]) for rule in ordered
            if rule.rule_id in responses
        }) if responses else len(ordered)
        return RuleOptimizationReport(
            ordered, unique, coverage, real_anomaly_coverage=None
        )


__all__ = [
    "RuleDependencyCycle", "UnsatisfiedRuleDependency", "RuleDependencyGraph",
    "RuleOptimizationReport", "GreedyRuleSetOptimizer",
]
