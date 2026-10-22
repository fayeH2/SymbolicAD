"""Validation-driven evolutionary search for a *small* typed rule set.

Only two quantities are optimized:

``L_normal``
    Mean rule-set violation on independent normal facts.  It prevents strict,
    brittle rules from surviving merely because they react to interventions.

``L_cf``
    Margin ranking loss between a rendered normal image and the same image
    after a validated semantic intervention.  It prevents the empty/widest
    possible rule set from winning on normal consistency alone.

Executability, parameter legality, dependency closure, proven duplicates and
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
    normal_scores: Tuple[float, ...]
    counterfactual_margins: Tuple[float, ...]
    rule_known_fractions: Mapping[str, float]

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


def _rule_signature(rule: RuleSpec) -> Tuple[str, Tuple[str, ...], str, Tuple[str, ...]]:
    return (
        rule.canonical_predicate,
        tuple(rule.entity_types),
        _parameter_signature(rule.parameters),
        tuple(rule.dependencies),
    )


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
    for rule in rule_set:
        if rule.canonical_predicate in {"count_eq", "count_in"}:
            if not _integer_parameter(rule.parameters.get("expected")):
                return False
            if not _integer_parameter(rule.parameters.get("tolerance", 0)):
                return False
            required_arity = 2 if rule.canonical_predicate == "count_in" else 1
            if len(rule.entity_types) != required_arity:
                return False
            if int(float(rule.parameters.get("tolerance", 0))) > int(
                float(rule.parameters["expected"])
            ):
                # A tolerance wider than the expected count makes absence
                # indistinguishable from satisfaction and is inadmissible.
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
        elif predicate in {"above", "left_of", "aligned_vertical", "disjoint"}:
            if len(rule.entity_types) != 2:
                return False
            try:
                threshold = float(rule.parameters.get("threshold", 0.0))
            except (TypeError, ValueError):
                return False
            if not np.isfinite(threshold) or threshold < 0.0:
                return False
    return True


def executable(rule_set: Sequence[RuleSpec], registry: RuleRegistry) -> bool:
    return all(registry.resolve(rule.canonical_predicate) is not None for rule in rule_set)


def contains_redundant_rules(rule_set: Sequence[RuleSpec]) -> bool:
    """Reject exact logical duplicates, not merely rules sharing an object."""
    signatures = [_rule_signature(rule) for rule in rule_set]
    return len(signatures) != len(set(signatures))


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
    unavailable = sorted({
        rule.canonical_predicate for rule in rule_set
        if registry.resolve(rule.canonical_predicate) is None
    })
    reasons.extend(f"UNSUPPORTED_PREDICATE:{name}" for name in unavailable)
    if contains_redundant_rules(rule_set):
        reasons.append("PROVEN_EXACT_REDUNDANCY")
    if structurally_valid(rule_set) and contains_unconditional_contradiction(rule_set):
        reasons.append("UNCONDITIONAL_SEMANTIC_CONTRADICTION")
    return tuple(reasons)


def _execute(
    rule_set: Sequence[RuleSpec],
    context: ExecutionContext,
    registry: RuleRegistry,
) -> Tuple[ViolationResult, ...]:
    return tuple(
        registry.resolve(rule.canonical_predicate).executor.execute(rule, context)
        for rule in rule_set
    )


def _aggregate(results: Sequence[ViolationResult]) -> Tuple[float, bool]:
    known = [row for row in results if row.status != RuleStatus.UNKNOWN]
    if not known:
        return 0.0, False
    return float(max(row.violation_score for row in known)), True


def evaluate_rule_set(
    rule_set: Sequence[RuleSpec],
    normal_facts: Sequence[ExecutionContext],
    counterfactuals: Sequence[CounterfactualPair],
    *,
    registry: Optional[RuleRegistry] = None,
    minimum_known_fraction: float = 0.8,
    counterfactual_margin: float = 0.5,
) -> Optional[RuleSetEvaluation]:
    """Evaluate the two losses, returning ``None`` for any hard violation."""
    registry = registry or default_rule_registry()
    rules = tuple(rule_set)
    if hard_constraint_reasons(rules, registry):
        return None
    if not normal_facts or not counterfactuals:
        return None
    if any(not pair.validated for pair in counterfactuals):
        return None

    normal_rows = [_execute(rules, context, registry) for context in normal_facts]
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
    normal_scores = tuple(score for score, _ in normal_aggregates)
    normal_loss = float(np.mean(normal_scores))

    margins = []
    known_pairs = 0
    for pair in counterfactuals:
        before_score, before_known = _aggregate(_execute(rules, pair.normal, registry))
        after_score, after_known = _aggregate(_execute(rules, pair.counterfactual, registry))
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
    )


def pareto_front(
    evaluations: Iterable[RuleSetEvaluation],
) -> Tuple[RuleSetEvaluation, ...]:
    """Return deterministic non-dominated solutions for minimization."""
    unique: Dict[Tuple[Tuple[str, str], ...], RuleSetEvaluation] = {}
    for row in evaluations:
        key = tuple((rule.rule_id, _parameter_signature(rule.parameters)) for rule in row.rules)
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
            tuple(rule.rule_id for rule in row.rules),
        ),
    ))


Genome = Tuple[Optional[int], ...]


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


def _admissible_candidates(
    candidates: Sequence[RuleSpec], registry: RuleRegistry
) -> Tuple[RuleSpec, ...]:
    result = []
    seen = set()
    for rule in candidates:
        if registry.resolve(rule.canonical_predicate) is None:
            continue
        # Candidate-level schema check uses its proposed tolerance.
        if not structurally_valid((rule,)):
            continue
        signature = _rule_signature(rule)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(rule)
    return tuple(result)


class EvolutionaryRuleSearch:
    """Deterministic small-population Pareto GA over add/drop/tolerance edits."""

    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        registry: Optional[RuleRegistry] = None,
    ) -> None:
        self.config = config or SearchConfig()
        self.registry = registry or default_rule_registry()

    def _random_genome(self, count: int, rng: random.Random) -> Genome:
        choices = (None,) + tuple(int(v) for v in self.config.tolerance_grid)
        values = tuple(rng.choice(choices) for _ in range(count))
        if all(value is None for value in values):
            index = rng.randrange(count)
            values = tuple(
                int(self.config.tolerance_grid[0]) if i == index else value
                for i, value in enumerate(values)
            )
        return values

    def _mutate(self, genome: Genome, rng: random.Random) -> Genome:
        choices = (None,) + tuple(int(v) for v in self.config.tolerance_grid)
        values = list(genome)
        for index in range(len(values)):
            if rng.random() < self.config.mutation_probability:
                alternatives = tuple(value for value in choices if value != values[index])
                values[index] = rng.choice(alternatives)
        if all(value is None for value in values):
            values[rng.randrange(len(values))] = int(self.config.tolerance_grid[0])
        return tuple(values)

    def search(
        self,
        candidates: Sequence[RuleSpec],
        normal_facts: Sequence[ExecutionContext],
        counterfactuals: Sequence[CounterfactualPair],
    ) -> SearchOutcome:
        admitted = _admissible_candidates(tuple(candidates), self.registry)
        if not admitted:
            raise ValueError("no executable, structurally valid rule candidate")
        if not self.config.tolerance_grid:
            raise ValueError("tolerance_grid cannot be empty")
        rng = random.Random(self.config.seed)
        population = []
        base_tolerance = int(self.config.tolerance_grid[0])
        # Seed every singleton and the full proposal, then fill randomly.
        for index in range(len(admitted)):
            population.append(tuple(
                base_tolerance if row == index else None
                for row in range(len(admitted))
            ))
        population.append(tuple(base_tolerance for _ in admitted))
        while len(population) < self.config.population_size:
            population.append(self._random_genome(len(admitted), rng))

        cache: Dict[Genome, Optional[RuleSetEvaluation]] = {}
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
                    self._random_genome(len(admitted), rng)
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
                next_population.append(self._mutate(child, rng))
            population = next_population
        if not archive:
            raise ValueError("evolution produced no feasible rule set")
        selected = min(
            archive,
            key=lambda row: (
                row.scalar(self.config.lambda_cf), row.normal_loss,
                row.counterfactual_loss, len(row.rules),
                tuple(rule.rule_id for rule in row.rules),
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
    choices = (None,) + tuple(int(value) for value in config.tolerance_grid)
    state_count = len(choices) ** len(admitted) - 1
    if state_count <= 0:
        raise ValueError("no admitted candidates")
    if state_count > int(maximum_states):
        raise ValueError(f"exhaustive state count {state_count} exceeds safety limit")
    evaluations = []
    for genome in itertools.product(choices, repeat=len(admitted)):
        if all(value is None for value in genome):
            continue
        row = evaluate_rule_set(
            _materialize(admitted, genome), normal_facts, counterfactuals,
            registry=registry,
            minimum_known_fraction=config.minimum_known_fraction,
            counterfactual_margin=config.counterfactual_margin,
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
            tuple(rule.rule_id for rule in row.rules),
        ),
    )
    return SearchOutcome(
        selected=selected, pareto_front=front, history=tuple(),
        evaluated_genomes=state_count, method="exhaustive_add_drop_tolerance_oracle",
    )


__all__ = [
    "CounterfactualPair", "RuleSetEvaluation", "SearchConfig", "SearchOutcome",
    "EvolutionaryRuleSearch", "evaluate_rule_set", "exhaustive_rule_search",
    "pareto_front", "structurally_valid", "executable",
    "contains_redundant_rules", "contains_unconditional_contradiction",
    "hard_constraint_reasons",
]
