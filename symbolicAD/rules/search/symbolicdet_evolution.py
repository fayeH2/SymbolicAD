"""Validation-driven evolutionary search for a *small* typed rule set.

Only two quantities are optimized:

``L_normal``
    Mean rule-set violation on independent normal facts.  It prevents strict,
    brittle rules from surviving merely because they react to interventions.

``L_cf``
    Margin ranking loss between a rendered normal image and the same image
    after a validated semantic intervention.  It prevents the empty/widest
    possible rule set from winning on normal consistency alone.

Executability, parameter legality, dependency closure, proven redundancies and
unconditional contradictions are deterministic hard constraints.  They are
not converted into extra weighted losses.  Consequently an unsupported Qwen
proposal remains visible in the admission audit but can never receive a
favourable evolutionary fitness by accident.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ...reasoning.predicate_executor import ExecutionContext
from ...reasoning.constraint_executor import ConstraintOperator, combine_statuses
from ...reasoning.violation_result import RuleStatus, ViolationResult
from ..rule_registry import RuleRegistry, default_rule_registry
from ..rule_schema import RuleSpec


@dataclass(frozen=True)
class CounterfactualPair:
    """A visual before/after pair whose facts were inferred independently."""

    pair_id: str
    normal: ExecutionContext
    counterfactual: ExecutionContext
    intervention: str
    validated: bool = True
    validation_evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleSetEvaluation:
    """Two-objective fitness plus the evidence needed to reproduce it."""

    rules: Tuple[RuleSpec, ...]
    normal_loss: float
    counterfactual_loss: float
    normal_known_fraction: float
    counterfactual_known_fraction: float
    normal_scores: Tuple[Optional[float], ...]
    counterfactual_margins: Tuple[float, ...]
    rule_known_fractions: Mapping[str, float]
    rule_counterfactual_paired_fractions: Mapping[str, float]

    @property
    def objectives(self) -> Tuple[float, float]:
        return self.normal_loss, self.counterfactual_loss

    def scalar(self, lambda_cf: float = 1.0) -> float:
        """Reporting/tie-break score; Pareto selection still uses both axes."""
        return float(self.normal_loss + float(lambda_cf) * self.counterfactual_loss)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_ids": [rule.rule_id for rule in self.rules],
            "rules": [rule.to_dict() for rule in self.rules],
            "L_normal": self.normal_loss,
            "L_cf": self.counterfactual_loss,
            "normal_known_fraction": self.normal_known_fraction,
            "counterfactual_known_fraction": self.counterfactual_known_fraction,
            "normal_scores": list(self.normal_scores),
            "counterfactual_margins": list(self.counterfactual_margins),
            "rule_known_fractions": dict(self.rule_known_fractions),
            "rule_counterfactual_paired_fractions": dict(
                self.rule_counterfactual_paired_fractions
            ),
        }


@dataclass(frozen=True)
class SearchConfig:
    population_size: int = 32
    generations: int = 40
    mutation_probability: float = 0.25
    crossover_probability: float = 0.8
    tolerance_grid: Tuple[int, ...] = (0, 1)
    minimum_known_fraction: float = 0.8
    counterfactual_margin: float = 0.5
    lambda_cf: float = 1.0
    seed: int = 1

    def __post_init__(self) -> None:
        if int(self.population_size) < 2:
            raise ValueError("population_size must be at least 2")
        if int(self.generations) < 1:
            raise ValueError("generations must be positive")
        for name in ("mutation_probability", "crossover_probability"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        if not self.tolerance_grid or any(
            not _integer_parameter(value) for value in self.tolerance_grid
        ):
            raise ValueError("tolerance_grid must contain non-negative integers")
        known = float(self.minimum_known_fraction)
        if not np.isfinite(known) or not 0.0 < known <= 1.0:
            raise ValueError("minimum_known_fraction must be finite and in (0, 1]")
        margin = float(self.counterfactual_margin)
        if not np.isfinite(margin) or not 0.0 < margin <= 1.0:
            raise ValueError("counterfactual_margin must be finite and in (0, 1]")
        weight = float(self.lambda_cf)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("lambda_cf must be finite and positive")


@dataclass(frozen=True)
class SearchOutcome:
    selected: RuleSetEvaluation
    pareto_front: Tuple[RuleSetEvaluation, ...]
    history: Tuple[Mapping[str, Any], ...]
    evaluated_genomes: int
    method: str = "pareto_genetic_add_drop_tolerance"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "selected": self.selected.to_dict(),
            "pareto_front": [row.to_dict() for row in self.pareto_front],
            "history": [dict(row) for row in self.history],
            "evaluated_genomes": self.evaluated_genomes,
        }


def _parameter_signature(parameters: Mapping[str, Any]) -> str:
    return json.dumps(dict(parameters), ensure_ascii=False, sort_keys=True, default=str)


def _canonical_entity_types(rule: RuleSpec) -> Tuple[str, ...]:
    """Return the semantic entity tuple used by equality checks.

    ``aligned_vertical(x, x)`` is Qwen's binary spelling of "the observed
    ``x`` instances share one column".  Its executor deliberately de-duplicates
    instances, so it is semantically the same constraint as the unary spelling
    ``aligned_vertical(x)``.  The raw pair remains in :class:`RuleSpec` and its
    support evidence; only search identity is canonicalized here.
    """
    entities = tuple(rule.entity_types)
    if (
        rule.canonical_predicate == "aligned_vertical"
        and len(entities) == 2
        and entities[0] == entities[1]
    ):
        return (entities[0],)
    return entities


def _rule_signature(rule: RuleSpec) -> Tuple[str, Tuple[str, ...], str, Tuple[str, ...]]:
    return (
        rule.canonical_predicate,
        _canonical_entity_types(rule),
        _parameter_signature(rule.parameters),
        tuple(sorted(set(rule.dependencies))),
    )


def rule_set_phenotype(
    rules: Sequence[RuleSpec],
) -> Tuple[Tuple[str, str, Tuple[str, ...], str, Tuple[str, ...]], ...]:
    """Order-independent rule-set identity including searched parameters."""
    rows = (
        (
            rule.rule_id, rule.canonical_predicate, _canonical_entity_types(rule),
            _parameter_signature(rule.parameters),
            tuple(sorted(set(rule.dependencies))),
        )
        for rule in rules
    )
    return tuple(sorted(rows))


def _cyclic_dependency_ids(rule_set: Sequence[RuleSpec]) -> Tuple[str, ...]:
    """Return all identifiers that belong to a dependency cycle.

    Tarjan's strongly-connected components algorithm makes rejection
    independent of proposal order and identifies the cycle members without
    unnecessarily rejecting unrelated rules.
    """
    graph = {
        rule.rule_id: tuple(sorted(set(rule.dependencies))) for rule in rule_set
    }
    next_index = 0
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    cyclic: set[str] = set()

    def visit(rule_id: str) -> None:
        nonlocal next_index
        indices[rule_id] = next_index
        lowlinks[rule_id] = next_index
        next_index += 1
        stack.append(rule_id)
        on_stack.add(rule_id)
        for dependency in graph.get(rule_id, ()):
            if dependency not in graph:
                continue
            if dependency not in indices:
                visit(dependency)
                lowlinks[rule_id] = min(lowlinks[rule_id], lowlinks[dependency])
            elif dependency in on_stack:
                lowlinks[rule_id] = min(lowlinks[rule_id], indices[dependency])
        if lowlinks[rule_id] != indices[rule_id]:
            return
        component = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == rule_id:
                break
        if len(component) > 1 or rule_id in graph.get(rule_id, ()):
            cyclic.update(component)

    for rule_id in sorted(graph):
        if rule_id not in indices:
            visit(rule_id)
    return tuple(sorted(cyclic))


def _integer_parameter(value: Any, *, minimum: int = 0) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number == int(number) and number >= minimum)


def structurally_valid(rule_set: Sequence[RuleSpec]) -> bool:
    """Check schema-level invariants without executing visual facts."""
    if not rule_set:
        return False
    ids = [rule.rule_id for rule in rule_set]
    if len(ids) != len(set(ids)):
        return False
    selected = set(ids)
    if any(not set(rule.dependencies).issubset(selected) for rule in rule_set):
        return False
    if _cyclic_dependency_ids(rule_set):
        return False
    for rule in rule_set:
        if rule.canonical_predicate in {"count_eq", "count_in"}:
            if not _integer_parameter(rule.parameters.get("expected")):
                return False
            if not _integer_parameter(rule.parameters.get("tolerance", 0)):
                return False
            required_arity = 2 if rule.canonical_predicate == "count_in" else 1
            if len(rule.entity_types) != required_arity:
                return False
            expected = int(float(rule.parameters["expected"]))
            tolerance = int(float(rule.parameters.get("tolerance", 0)))
            if expected > 0 and tolerance >= expected:
                # If tolerance reaches the positive expected count, complete
                # absence becomes SATISFIED and the normality claim is void.
                return False
    for rule in rule_set:
        predicate = rule.canonical_predicate
        if predicate == "exists":
            if len(rule.entity_types) not in {1, 2}:
                return False
            if not _integer_parameter(rule.parameters.get("minimum_count", 1)):
                return False
            if int(float(rule.parameters.get("minimum_count", 1))) < 1:
                # "At least zero" is a tautology, not a normality constraint.
                return False
        elif predicate in {"above", "left_of", "disjoint"}:
            if len(rule.entity_types) != 2:
                return False
            try:
                threshold = float(rule.parameters.get("threshold", 0.0))
            except (TypeError, ValueError):
                return False
            if not np.isfinite(threshold) or threshold < 0.0:
                return False
        elif predicate == "aligned_vertical":
            if len(rule.entity_types) not in {1, 2}:
                return False
            try:
                threshold = float(rule.parameters.get("threshold", 0.0))
            except (TypeError, ValueError):
                return False
            if not np.isfinite(threshold) or threshold < 0.0:
                return False
    return True


def candidate_structurally_valid(rule: RuleSpec) -> bool:
    """Validate one candidate without pretending its dependencies are absent.

    Dependency closure is a rule-*set* hard constraint.  Candidate admission
    checks the rule's own schema and leaves closure to materialized genomes.
    """
    return structurally_valid((replace(rule, dependencies=()),))


def executable(rule_set: Sequence[RuleSpec], registry: RuleRegistry) -> bool:
    return all(registry.resolve(rule.canonical_predicate) is not None for rule in rule_set)


def _evidence_parameter_signature(
    rule: RuleSpec, constraint_parameters: Sequence[str],
) -> str:
    """Parameters that affect executability rather than the truth interval."""
    parameters = {
        key: value for key, value in rule.parameters.items()
        if key not in set(constraint_parameters)
    }
    # Count and presence executors share this default.  Normalizing it makes
    # an omitted default equivalent to an explicitly serialized default while
    # still keeping different evidence gates distinct.
    parameters.setdefault("minimum_confidence", 0.5)
    return _parameter_signature(parameters)


def _count_interval(rule: RuleSpec) -> Optional[Tuple[int, int]]:
    try:
        expected_value = float(rule.parameters["expected"])
        tolerance_value = float(rule.parameters.get("tolerance", 0))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        not np.isfinite(expected_value) or not np.isfinite(tolerance_value)
        or expected_value != int(expected_value)
        or tolerance_value != int(tolerance_value)
        or expected_value < 0 or tolerance_value < 0
    ):
        return None
    expected, tolerance = int(expected_value), int(tolerance_value)
    return max(0, expected - tolerance), expected + tolerance


def _exists_lower_bound(rule: RuleSpec) -> Optional[int]:
    try:
        value = float(rule.parameters.get("minimum_count", 1))
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(value) or value != int(value) or value < 1:
        return None
    return int(value)


def _provably_implies(stronger: RuleSpec, weaker: RuleSpec) -> bool:
    """Conservative implication over one identical unconditional scope.

    No ontology arithmetic, cross-entity reasoning, or response correlation is
    used.  We only compare integer intervals interpreted by the same executor
    under the same dependency and evidence-confidence contract.
    """
    if stronger.canonical_predicate != weaker.canonical_predicate:
        return False
    if _canonical_entity_types(stronger) != _canonical_entity_types(weaker):
        return False
    if tuple(sorted(set(stronger.dependencies))) != tuple(
        sorted(set(weaker.dependencies))
    ):
        return False
    predicate = stronger.canonical_predicate
    if predicate in {"count_eq", "count_in"}:
        if _evidence_parameter_signature(
            stronger, ("expected", "tolerance")
        ) != _evidence_parameter_signature(weaker, ("expected", "tolerance")):
            return False
        strict_interval = _count_interval(stronger)
        loose_interval = _count_interval(weaker)
        if strict_interval is None or loose_interval is None:
            return False
        return (
            strict_interval[0] >= loose_interval[0]
            and strict_interval[1] <= loose_interval[1]
        )
    if predicate == "exists":
        if _evidence_parameter_signature(
            stronger, ("minimum_count",)
        ) != _evidence_parameter_signature(weaker, ("minimum_count",)):
            return False
        strict_minimum = _exists_lower_bound(stronger)
        loose_minimum = _exists_lower_bound(weaker)
        if strict_minimum is None or loose_minimum is None:
            return False
        return strict_minimum >= loose_minimum
    return False


def contains_redundant_rules(rule_set: Sequence[RuleSpec]) -> bool:
    """Reject exact aliases and conservatively proven interval implications."""
    signatures = [_rule_signature(rule) for rule in rule_set]
    if len(signatures) != len(set(signatures)):
        return True
    for index, left in enumerate(rule_set):
        for right in rule_set[index + 1:]:
            if _provably_implies(left, right) or _provably_implies(right, left):
                return True
    return False


def contains_unconditional_contradiction(rule_set: Sequence[RuleSpec]) -> bool:
    """Detect disjoint count intervals on the same unconditional scope."""
    groups: Dict[Tuple[str, Tuple[str, ...]], list[Tuple[int, int]]] = {}
    for rule in rule_set:
        if rule.canonical_predicate not in {"count_eq", "count_in"}:
            continue
        expected = int(float(rule.parameters["expected"]))
        tolerance = int(float(rule.parameters.get("tolerance", 0)))
        groups.setdefault(
            (rule.canonical_predicate, tuple(rule.entity_types)), []
        ).append((max(0, expected - tolerance), expected + tolerance))
    for intervals in groups.values():
        if len(intervals) < 2:
            continue
        lower = max(row[0] for row in intervals)
        upper = min(row[1] for row in intervals)
        if lower > upper:
            return True
    return False


def hard_constraint_reasons(
    rule_set: Sequence[RuleSpec],
    registry: Optional[RuleRegistry] = None,
) -> Tuple[str, ...]:
    registry = registry or default_rule_registry()
    reasons = []
    if not structurally_valid(rule_set):
        reasons.append("STRUCTURALLY_INVALID")
    if any(rule.dependencies for rule in rule_set):
        reasons.append("DEPENDENCIES_NOT_EXECUTABLE")
    unavailable = sorted({
        rule.canonical_predicate for rule in rule_set
        if registry.resolve(rule.canonical_predicate) is None
    })
    reasons.extend(f"UNSUPPORTED_PREDICATE:{name}" for name in unavailable)
    if contains_redundant_rules(rule_set):
        reasons.append("PROVEN_LOGICAL_REDUNDANCY")
    if structurally_valid(rule_set) and contains_unconditional_contradiction(rule_set):
        reasons.append("UNCONDITIONAL_SEMANTIC_CONTRADICTION")
    return tuple(reasons)


def _execute(
    rule_set: Sequence[RuleSpec],
    context: ExecutionContext,
    registry: RuleRegistry,
    atomic_cache: Optional[Dict[Tuple[Any, int], ViolationResult]] = None,
) -> Tuple[ViolationResult, ...]:
    rows = []
    for rule in rule_set:
        key = ((rule.rule_id, _rule_signature(rule)), id(context))
        result = None if atomic_cache is None else atomic_cache.get(key)
        if result is None:
            result = registry.resolve(
                rule.canonical_predicate
            ).executor.execute(rule, context)
            if atomic_cache is not None:
                atomic_cache[key] = result
        rows.append(result)
    return tuple(rows)


def _aggregate(results: Sequence[ViolationResult]) -> Tuple[float, bool]:
    status = combine_statuses(
        (row.status for row in results), ConstraintOperator.AND
    )
    if status == RuleStatus.UNKNOWN:
        return 0.0, False
    if status == RuleStatus.SATISFIED:
        return 0.0, True
    violated = [row for row in results if row.status == RuleStatus.VIOLATED]
    return float(max(row.violation_score for row in violated)), True


def evaluate_rule_set(
    rule_set: Sequence[RuleSpec],
    normal_facts: Sequence[ExecutionContext],
    counterfactuals: Sequence[CounterfactualPair],
    *,
    registry: Optional[RuleRegistry] = None,
    minimum_known_fraction: float = 0.8,
    counterfactual_margin: float = 0.5,
    atomic_cache: Optional[Dict[Tuple[Any, int], ViolationResult]] = None,
) -> Optional[RuleSetEvaluation]:
    """Evaluate the two losses, returning ``None`` for any hard violation."""
    if not 0.0 < float(minimum_known_fraction) <= 1.0:
        raise ValueError("minimum_known_fraction must be in (0, 1]")
    registry = registry or default_rule_registry()
    rules = tuple(rule_set)
    if hard_constraint_reasons(rules, registry):
        return None
    if not normal_facts or not counterfactuals:
        return None
    if any(not pair.validated for pair in counterfactuals):
        return None

    normal_rows = [
        _execute(rules, context, registry, atomic_cache) for context in normal_facts
    ]
    per_rule_known = {}
    for index, rule in enumerate(rules):
        fraction = float(np.mean([
            rows[index].status != RuleStatus.UNKNOWN for rows in normal_rows
        ]))
        per_rule_known[rule.rule_id] = fraction
        if fraction < float(minimum_known_fraction):
            return None
    normal_aggregates = [_aggregate(rows) for rows in normal_rows]
    normal_known = float(np.mean([known for _, known in normal_aggregates]))
    if normal_known < float(minimum_known_fraction):
        return None
    normal_scores = tuple(
        score if known else None for score, known in normal_aggregates
    )
    decided_normal_scores = [
        score for score, known in normal_aggregates if known
    ]
    normal_loss = float(np.mean(decided_normal_scores))

    margins = []
    known_pairs = 0
    paired_counts = {rule.rule_id: 0 for rule in rules}
    for pair in counterfactuals:
        before_results = _execute(rules, pair.normal, registry, atomic_cache)
        after_results = _execute(
            rules, pair.counterfactual, registry, atomic_cache
        )
        paired = [
            before.status != RuleStatus.UNKNOWN
            and after.status != RuleStatus.UNKNOWN
            for before, after in zip(before_results, after_results)
        ]
        for rule, is_paired in zip(rules, paired):
            paired_counts[rule.rule_id] += int(is_paired)
        # A ranking margin is valid only when the same complete rule set is
        # executable on both members.  Cross-rule known-set swaps are not
        # comparable evidence and are rejected as a hard coverage condition.
        if not all(paired):
            continue
        before_score, before_known = _aggregate(before_results)
        after_score, after_known = _aggregate(after_results)
        if before_known and after_known:
            known_pairs += 1
            margins.append(float(after_score - before_score))
    pair_fraction = known_pairs / len(counterfactuals)
    if pair_fraction < float(minimum_known_fraction):
        return None
    margin_target = max(float(counterfactual_margin), 1e-8)
    cf_loss = float(np.mean([
        max(0.0, margin_target - observed) / margin_target for observed in margins
    ]))
    return RuleSetEvaluation(
        rules=rules,
        normal_loss=normal_loss,
        counterfactual_loss=cf_loss,
        normal_known_fraction=normal_known,
        counterfactual_known_fraction=float(pair_fraction),
        normal_scores=normal_scores,
        counterfactual_margins=tuple(margins),
        rule_known_fractions=per_rule_known,
        rule_counterfactual_paired_fractions={
            rule_id: count / len(counterfactuals)
            for rule_id, count in paired_counts.items()
        },
    )


def pareto_front(
    evaluations: Iterable[RuleSetEvaluation],
) -> Tuple[RuleSetEvaluation, ...]:
    """Return deterministic non-dominated solutions for minimization."""
    unique: Dict[Any, RuleSetEvaluation] = {}
    for row in evaluations:
        key = rule_set_phenotype(row.rules)
        unique[key] = row
    rows = list(unique.values())
    front = []
    for candidate in rows:
        dominated = any(
            other is not candidate
            and other.normal_loss <= candidate.normal_loss
            and other.counterfactual_loss <= candidate.counterfactual_loss
            and (
                other.normal_loss < candidate.normal_loss
                or other.counterfactual_loss < candidate.counterfactual_loss
            )
            for other in rows
        )
        if not dominated:
            front.append(candidate)
    return tuple(sorted(
        front,
        key=lambda row: (
            row.normal_loss, row.counterfactual_loss, len(row.rules),
            rule_set_phenotype(row.rules),
        ),
    ))


Genome = Tuple[Optional[int], ...]


def _gene_choices(rule: RuleSpec, tolerance_grid: Sequence[int]) -> Tuple[Optional[int], ...]:
    """Only count predicates own a numeric tolerance gene.

    Relation/presence rules still have an add/drop gene, but duplicating them
    once per tolerance value would create identical phenotypes and inflate the
    exhaustive oracle from 2^m to 3^m for no semantic reason.
    """
    if rule.canonical_predicate in {"count_eq", "count_in"}:
        legal = []
        for value in dict.fromkeys(int(item) for item in tolerance_grid):
            parameters = dict(rule.parameters)
            parameters["tolerance"] = value
            if candidate_structurally_valid(replace(rule, parameters=parameters)):
                legal.append(value)
        return (None,) + tuple(legal)
    return (None, 0)


def _materialize(
    candidates: Sequence[RuleSpec], genome: Genome
) -> Tuple[RuleSpec, ...]:
    rules = []
    for rule, tolerance in zip(candidates, genome):
        if tolerance is None:
            continue
        if rule.canonical_predicate in {"count_eq", "count_in"}:
            parameters = dict(rule.parameters)
            parameters["tolerance"] = int(tolerance)
            rule = replace(rule, parameters=parameters)
        rules.append(rule)
    return tuple(rules)


def _genome_key(genome: Genome) -> Tuple[int, ...]:
    return tuple(-1 if value is None else int(value) for value in genome)


def _canonicalize_exact_aliases(
    rules: Sequence[RuleSpec],
) -> Tuple[Tuple[RuleSpec, ...], Mapping[str, str]]:
    """Collapse exact aliases and rewrite dependency references deterministically.

    The lexicographically smallest identifier is the canonical representative.
    Repeating to a fixed point also catches rules that become aliases only
    after their dependency aliases have been rewritten.
    """
    current = tuple(sorted(rules, key=lambda rule: rule.rule_id))
    aliases: Dict[str, str] = {}
    while current:
        groups: Dict[
            Tuple[str, Tuple[str, ...], str, Tuple[str, ...]], list[RuleSpec]
        ] = {}
        for rule in current:
            groups.setdefault(_rule_signature(rule), []).append(rule)
        step: Dict[str, str] = {}
        for group in groups.values():
            if len(group) < 2:
                continue
            canonical_id = min(rule.rule_id for rule in group)
            for rule in group:
                if rule.rule_id != canonical_id:
                    step[rule.rule_id] = canonical_id
        if not step:
            break

        def resolve(rule_id: str) -> str:
            seen = set()
            while rule_id in step and rule_id not in seen:
                seen.add(rule_id)
                rule_id = step[rule_id]
            return rule_id

        aliases = {key: resolve(value) for key, value in aliases.items()}
        aliases.update({key: resolve(value) for key, value in step.items()})
        rewritten = []
        for rule in current:
            if rule.rule_id in step:
                continue
            dependencies = tuple(sorted({
                resolve(dependency) for dependency in rule.dependencies
            }))
            rewritten.append(replace(rule, dependencies=dependencies))
        current = tuple(sorted(rewritten, key=lambda rule: rule.rule_id))
    return current, dict(sorted(aliases.items()))


def _prune_invalid_dependency_candidates(
    rules: Sequence[RuleSpec],
) -> Tuple[RuleSpec, ...]:
    """Remove missing/cyclic dependency roots and every dependent orphan."""
    current = tuple(sorted(rules, key=lambda rule: rule.rule_id))
    while current:
        identifiers = {rule.rule_id for rule in current}
        missing = {
            rule.rule_id for rule in current
            if not set(rule.dependencies).issubset(identifiers)
        }
        retained = tuple(
            rule for rule in current if rule.rule_id not in missing
        )
        cyclic = set(_cyclic_dependency_ids(retained))
        retained = tuple(
            rule for rule in retained if rule.rule_id not in cyclic
        )
        if len(retained) == len(current):
            return retained
        current = retained
    return tuple()


def _admissible_candidates(
    candidates: Sequence[RuleSpec], registry: RuleRegistry
) -> Tuple[RuleSpec, ...]:
    identifiers = [rule.rule_id for rule in candidates]
    duplicates = sorted({
        rule_id for rule_id in identifiers if identifiers.count(rule_id) > 1
    })
    if duplicates:
        raise ValueError(f"duplicate candidate rule_id(s): {duplicates}")

    locally_valid = []
    for rule in sorted(candidates, key=lambda row: row.rule_id):
        if registry.resolve(rule.canonical_predicate) is None:
            continue
        # Dependency fields are preserved in the schema for future
        # conditional rules, but the present executor cannot yet enforce
        # "dependency must be SATISFIED" semantics.  Fail closed instead of
        # treating mere co-selection as logical truth.
        if rule.dependencies:
            continue
        # Candidate-level schema check uses its proposed tolerance.
        if not candidate_structurally_valid(rule):
            continue
        locally_valid.append(rule)
    canonical, _alias_map = _canonicalize_exact_aliases(locally_valid)
    return _prune_invalid_dependency_candidates(canonical)


class EvolutionaryRuleSearch:
    """Deterministic small-population Pareto GA over add/drop/tolerance edits."""

    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        registry: Optional[RuleRegistry] = None,
    ) -> None:
        self.config = config or SearchConfig()
        self.registry = registry or default_rule_registry()

    def _random_genome(
        self, candidates: Sequence[RuleSpec], rng: random.Random
    ) -> Genome:
        values = tuple(
            rng.choice(_gene_choices(rule, self.config.tolerance_grid))
            for rule in candidates
        )
        if all(value is None for value in values):
            index = rng.randrange(len(candidates))
            replacement = _gene_choices(
                candidates[index], self.config.tolerance_grid
            )[1]
            values = tuple(
                replacement if i == index else value
                for i, value in enumerate(values)
            )
        return values

    def _mutate(
        self, genome: Genome, candidates: Sequence[RuleSpec], rng: random.Random
    ) -> Genome:
        values = list(genome)
        for index in range(len(values)):
            if rng.random() < self.config.mutation_probability:
                choices = _gene_choices(
                    candidates[index], self.config.tolerance_grid
                )
                alternatives = tuple(value for value in choices if value != values[index])
                values[index] = rng.choice(alternatives)
        if all(value is None for value in values):
            index = rng.randrange(len(values))
            values[index] = _gene_choices(
                candidates[index], self.config.tolerance_grid
            )[1]
        return tuple(values)

    def search(
        self,
        candidates: Sequence[RuleSpec],
        normal_facts: Sequence[ExecutionContext],
        counterfactuals: Sequence[CounterfactualPair],
    ) -> SearchOutcome:
        admitted = _admissible_candidates(tuple(candidates), self.registry)
        admitted = tuple(
            rule for rule in admitted
            if len(_gene_choices(rule, self.config.tolerance_grid)) > 1
        )
        if not admitted:
            raise ValueError("no executable, structurally valid rule candidate")
        if not self.config.tolerance_grid:
            raise ValueError("tolerance_grid cannot be empty")
        rng = random.Random(self.config.seed)
        population = []
        # Seed every singleton and the full proposal, then fill randomly.
        for index in range(len(admitted)):
            population.append(tuple(
                _gene_choices(rule, self.config.tolerance_grid)[1]
                if row == index else None
                for row, rule in enumerate(admitted)
            ))
        population.append(tuple(
            _gene_choices(rule, self.config.tolerance_grid)[1]
            for rule in admitted
        ))
        while len(population) < self.config.population_size:
            population.append(self._random_genome(admitted, rng))

        cache: Dict[Genome, Optional[RuleSetEvaluation]] = {}
        atomic_cache: Dict[Tuple[Any, int], ViolationResult] = {}
        history = []
        archive: Tuple[RuleSetEvaluation, ...] = tuple()
        for generation in range(self.config.generations):
            for genome in population:
                if genome not in cache:
                    cache[genome] = evaluate_rule_set(
                        _materialize(admitted, genome), normal_facts, counterfactuals,
                        registry=self.registry,
                        minimum_known_fraction=self.config.minimum_known_fraction,
                        counterfactual_margin=self.config.counterfactual_margin,
                        atomic_cache=atomic_cache,
                    )
            feasible = [cache[genome] for genome in population if cache[genome] is not None]
            archive = pareto_front(tuple(archive) + tuple(feasible))
            if not feasible and generation + 1 == self.config.generations:
                break
            ranked = sorted(
                ((genome, cache[genome]) for genome in population if cache[genome] is not None),
                key=lambda item: (
                    item[1].scalar(self.config.lambda_cf),
                    item[1].normal_loss,
                    item[1].counterfactual_loss,
                    len(item[1].rules), _genome_key(item[0]),
                ),
            )
            history.append({
                "generation": generation,
                "population": len(population),
                "feasible": len(feasible),
                "pareto_size": len(archive),
                "best_L_normal": None if not feasible else min(row.normal_loss for row in feasible),
                "best_L_cf": None if not feasible else min(row.counterfactual_loss for row in feasible),
                "best_scalar": None if not feasible else min(
                    row.scalar(self.config.lambda_cf) for row in feasible
                ),
            })
            if not ranked:
                population = [
                    self._random_genome(admitted, rng)
                    for _ in range(self.config.population_size)
                ]
                continue
            survivor_count = max(2, min(len(ranked), self.config.population_size // 2))
            survivors = [item[0] for item in ranked[:survivor_count]]
            next_population = list(survivors)
            while len(next_population) < self.config.population_size:
                left, right = rng.choice(survivors), rng.choice(survivors)
                if rng.random() < self.config.crossover_probability and len(admitted) > 1:
                    cut = rng.randrange(1, len(admitted))
                    child = left[:cut] + right[cut:]
                else:
                    child = left
                next_population.append(self._mutate(child, admitted, rng))
            population = next_population
        if not archive:
            raise ValueError("evolution produced no feasible rule set")
        selected = min(
            archive,
            key=lambda row: (
                row.scalar(self.config.lambda_cf), row.normal_loss,
                row.counterfactual_loss, len(row.rules),
                rule_set_phenotype(row.rules),
            ),
        )
        return SearchOutcome(
            selected=selected,
            pareto_front=archive,
            history=tuple(history),
            evaluated_genomes=len(cache),
        )


def exhaustive_rule_search(
    candidates: Sequence[RuleSpec],
    normal_facts: Sequence[ExecutionContext],
    counterfactuals: Sequence[CounterfactualPair],
    *,
    config: Optional[SearchConfig] = None,
    registry: Optional[RuleRegistry] = None,
    maximum_states: int = 100_000,
) -> SearchOutcome:
    """Small-candidate oracle used only to verify that the GA found the front."""
    config = config or SearchConfig()
    registry = registry or default_rule_registry()
    admitted = _admissible_candidates(tuple(candidates), registry)
    admitted = tuple(
        rule for rule in admitted
        if len(_gene_choices(rule, config.tolerance_grid)) > 1
    )
    choices = tuple(
        _gene_choices(rule, config.tolerance_grid) for rule in admitted
    )
    state_count = int(np.prod([len(row) for row in choices], dtype=np.int64)) - 1
    if state_count <= 0:
        raise ValueError("no admitted candidates")
    if state_count > int(maximum_states):
        raise ValueError(f"exhaustive state count {state_count} exceeds safety limit")
    evaluations = []
    atomic_cache: Dict[Tuple[Any, int], ViolationResult] = {}
    for genome in itertools.product(*choices):
        if all(value is None for value in genome):
            continue
        row = evaluate_rule_set(
            _materialize(admitted, genome), normal_facts, counterfactuals,
            registry=registry,
            minimum_known_fraction=config.minimum_known_fraction,
            counterfactual_margin=config.counterfactual_margin,
            atomic_cache=atomic_cache,
        )
        if row is not None:
            evaluations.append(row)
    front = pareto_front(evaluations)
    if not front:
        raise ValueError("exhaustive search found no feasible rule set")
    selected = min(
        front,
        key=lambda row: (
            row.scalar(config.lambda_cf), row.normal_loss,
            row.counterfactual_loss, len(row.rules),
            rule_set_phenotype(row.rules),
        ),
    )
    return SearchOutcome(
        selected=selected, pareto_front=front, history=tuple(),
        evaluated_genomes=state_count, method="exhaustive_add_drop_tolerance_oracle",
    )


def greedy_rule_search(
    candidates: Sequence[RuleSpec],
    normal_facts: Sequence[ExecutionContext],
    counterfactuals: Sequence[CounterfactualPair],
    *,
    config: Optional[SearchConfig] = None,
    registry: Optional[RuleRegistry] = None,
) -> SearchOutcome:
    """Deterministic forward-selection control for the evolutionary search.

    This deliberately simple baseline starts from the best feasible singleton,
    then accepts only a one-gene add/tolerance edit that strictly lowers
    ``L_normal + lambda_cf * L_cf``.  It has no crossover, mutation, diversity
    bonus or hidden loss.  A genetic search that does not reliably outperform
    this control must not be promoted as the project's main algorithm.
    """
    config = config or SearchConfig()
    registry = registry or default_rule_registry()
    if not config.tolerance_grid:
        raise ValueError("tolerance_grid cannot be empty")
    if not 0.0 < float(config.minimum_known_fraction) <= 1.0:
        raise ValueError("minimum_known_fraction must be in (0, 1]")
    admitted = _admissible_candidates(tuple(candidates), registry)
    admitted = tuple(
        rule for rule in admitted
        if len(_gene_choices(rule, config.tolerance_grid)) > 1
    )
    if not admitted:
        raise ValueError("no executable, structurally valid rule candidate")

    empty: Genome = tuple(None for _ in admitted)
    current_genome = empty
    current: Optional[RuleSetEvaluation] = None
    cache: Dict[Genome, Optional[RuleSetEvaluation]] = {}
    atomic_cache: Dict[Tuple[Any, int], ViolationResult] = {}
    visited: set[Genome] = {empty}
    feasible_archive: list[RuleSetEvaluation] = []
    history = []

    while True:
        neighbours = []
        for index, rule in enumerate(admitted):
            choices = _gene_choices(rule, config.tolerance_grid)
            if current_genome[index] is None:
                proposed_values = choices[1:]
            else:
                # Once admitted, a rule may adjust only its legal numeric
                # tolerance.  Forward selection never deletes an accepted rule.
                proposed_values = tuple(
                    value for value in choices[1:]
                    if value != current_genome[index]
                )
            for value in proposed_values:
                trial = tuple(
                    value if row == index else present
                    for row, present in enumerate(current_genome)
                )
                if trial in visited:
                    continue
                visited.add(trial)
                evaluation = evaluate_rule_set(
                    _materialize(admitted, trial), normal_facts,
                    counterfactuals, registry=registry,
                    minimum_known_fraction=config.minimum_known_fraction,
                    counterfactual_margin=config.counterfactual_margin,
                    atomic_cache=atomic_cache,
                )
                cache[trial] = evaluation
                if evaluation is not None:
                    feasible_archive.append(evaluation)
                    neighbours.append((trial, evaluation))
        if not neighbours:
            break
        best_genome, best = min(
            neighbours,
            key=lambda item: (
                item[1].scalar(config.lambda_cf), item[1].normal_loss,
                item[1].counterfactual_loss, len(item[1].rules),
                rule_set_phenotype(item[1].rules), _genome_key(item[0]),
            ),
        )
        if (
            current is not None
            and best.scalar(config.lambda_cf)
            >= current.scalar(config.lambda_cf) - 1e-12
        ):
            break
        current_genome, current = best_genome, best
        history.append({
            "step": len(history),
            "selected_phenotype": rule_set_phenotype(current.rules),
            "L_normal": current.normal_loss,
            "L_cf": current.counterfactual_loss,
            "scalar": current.scalar(config.lambda_cf),
        })

    if current is None:
        raise ValueError("greedy search found no feasible rule set")
    return SearchOutcome(
        selected=current,
        pareto_front=pareto_front(feasible_archive),
        history=tuple(history),
        evaluated_genomes=len(cache),
        method="greedy_forward_add_tolerance_control",
    )


__all__ = [
    "CounterfactualPair", "RuleSetEvaluation", "SearchConfig", "SearchOutcome",
    "EvolutionaryRuleSearch", "evaluate_rule_set", "exhaustive_rule_search",
    "greedy_rule_search",
    "pareto_front", "structurally_valid", "executable",
    "contains_redundant_rules", "contains_unconditional_contradiction",
    "hard_constraint_reasons", "candidate_structurally_valid",
    "rule_set_phenotype",
]
