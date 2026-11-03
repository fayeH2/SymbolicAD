"""Deterministic regressions for rule identity, dependencies, and redundancy."""

import pytest

from symbolicAD.rules.rule_registry import default_rule_registry
from symbolicAD.rules.rule_schema import RuleSpec
from symbolicAD.rules.search.symbolicdet_evolution import (
    RuleSetEvaluation,
    SearchOutcome,
    _admissible_candidates,
    contains_redundant_rules,
    pareto_front,
    rule_set_phenotype,
    structurally_valid,
)
from symbolicAD.rules.verify_breakfast_box import compare_search


def _evaluation(rule: RuleSpec) -> RuleSetEvaluation:
    return RuleSetEvaluation(
        rules=(rule,),
        normal_loss=0.1,
        counterfactual_loss=0.2,
        normal_known_fraction=1.0,
        counterfactual_known_fraction=1.0,
        normal_scores=(0.1,),
        counterfactual_margins=(0.3,),
        rule_known_fractions={rule.rule_id: 1.0},
        rule_counterfactual_paired_fractions={rule.rule_id: 1.0},
    )


def _outcome(row: RuleSetEvaluation) -> SearchOutcome:
    return SearchOutcome(
        selected=row,
        pareto_front=(row,),
        history=tuple(),
        evaluated_genomes=1,
    )


def test_phenotype_is_order_independent_and_canonicalizes_self_alignment():
    self_pair = RuleSpec(
        "alignment", "aligned_vertical", ("orange", "orange"),
        {"threshold": 0.1}, dependencies=("z", "a", "a"),
    )
    unary = RuleSpec(
        "alignment", "aligned_vertical", ("orange",),
        {"threshold": 0.1}, dependencies=("a", "z"),
    )
    presence = RuleSpec(
        "presence", "exists", ("orange",), {"minimum_count": 1},
    )

    pair_phenotype = rule_set_phenotype((self_pair, presence))
    unary_phenotype = rule_set_phenotype((presence, unary))

    assert pair_phenotype == unary_phenotype
    alignment = next(row for row in pair_phenotype if row[0] == "alignment")
    assert alignment[2] == ("orange",)
    assert alignment[4] == ("a", "z")


def test_pareto_and_comparison_keys_use_the_complete_phenotype():
    # The historical partial key (rule_id, parameters) collided for these.
    presence = RuleSpec("same_id", "exists", ("part",), {})
    alignment = RuleSpec("same_id", "aligned_vertical", ("part",), {})
    presence_row = _evaluation(presence)
    alignment_row = _evaluation(alignment)

    assert len(pareto_front((presence_row, alignment_row))) == 2
    comparison = compare_search(
        _outcome(presence_row), _outcome(alignment_row)
    )
    assert comparison["selected_rule_ids_match"] is True
    assert comparison["selected_phenotype_match"] is False
    assert comparison["front_intersection"] == 0
    assert comparison["front_recall"] == 0.0


def test_candidate_rule_ids_must_be_globally_unique():
    first = RuleSpec("duplicate", "exists", ("a",), {"minimum_count": 1})
    second = RuleSpec("duplicate", "exists", ("b",), {"minimum_count": 1})

    with pytest.raises(ValueError, match="duplicate candidate rule_id"):
        _admissible_candidates(
            (first, second), default_rule_registry()
        )


def test_dependencies_are_rejected_until_conditional_execution_is_implemented():
    canonical = RuleSpec(
        "base_a", "exists", ("part",), {"minimum_count": 1},
    )
    alias = RuleSpec(
        "base_z", "exists", ("part",), {"minimum_count": 1},
    )
    dependent = RuleSpec(
        "dependent", "exists", ("other",), {"minimum_count": 1},
        dependencies=("base_z",),
    )
    registry = default_rule_registry()
    admitted = _admissible_candidates(
        (canonical, alias, dependent), registry
    )
    assert [rule.rule_id for rule in admitted] == ["base_a"]


def test_self_and_mutual_dependency_cycles_are_rejected_with_dependents():
    good = RuleSpec("good", "exists", ("good",), {"minimum_count": 1})
    self_cycle = RuleSpec(
        "self", "exists", ("self",), {"minimum_count": 1},
        dependencies=("self",),
    )
    left = RuleSpec(
        "left", "exists", ("left",), {"minimum_count": 1},
        dependencies=("right",),
    )
    right = RuleSpec(
        "right", "exists", ("right",), {"minimum_count": 1},
        dependencies=("left",),
    )
    downstream = RuleSpec(
        "downstream", "exists", ("downstream",), {"minimum_count": 1},
        dependencies=("left",),
    )

    assert not structurally_valid((self_cycle,))
    assert not structurally_valid((left, right))
    admitted = _admissible_candidates(
        (downstream, right, good, self_cycle, left), default_rule_registry()
    )
    assert [rule.rule_id for rule in admitted] == ["good"]


def test_only_proven_same_scope_interval_implications_are_redundant():
    strict_count = RuleSpec(
        "strict", "count_in", ("fruit", "left"),
        {"expected": 3, "tolerance": 0},
    )
    loose_count = RuleSpec(
        "loose", "count_in", ("fruit", "left"),
        {"expected": 3, "tolerance": 1, "minimum_confidence": 0.5},
    )
    other_scope = RuleSpec(
        "other_scope", "count_in", ("fruit", "right"),
        {"expected": 3, "tolerance": 1},
    )
    other_evidence_gate = RuleSpec(
        "other_gate", "count_in", ("fruit", "left"),
        {"expected": 3, "tolerance": 1, "minimum_confidence": 0.9},
    )
    strict_exists = RuleSpec(
        "exists_two", "exists", ("almond", "right"),
        {"minimum_count": 2},
    )
    loose_exists = RuleSpec(
        "exists_one", "exists", ("almond", "right"),
        {"minimum_count": 1},
    )

    assert contains_redundant_rules((strict_count, loose_count))
    assert contains_redundant_rules((strict_exists, loose_exists))
    assert not contains_redundant_rules((strict_count, other_scope))
    assert not contains_redundant_rules((strict_count, other_evidence_gate))
    # count and exists use different three-valued evidence contracts, so no
    # cross-executor implication is claimed here.
    same_scope_exists = RuleSpec(
        "fruit_exists", "exists", ("fruit", "left"),
        {"minimum_count": 1},
    )
    assert not contains_redundant_rules((strict_count, same_scope_exists))


def test_unary_and_repeated_self_alignment_are_exact_semantic_aliases():
    unary = RuleSpec(
        "unary", "aligned_vertical", ("orange",), {"threshold": 0.1},
    )
    repeated = RuleSpec(
        "repeated", "aligned_vertical", ("orange", "orange"),
        {"threshold": 0.1},
    )
    assert contains_redundant_rules((unary, repeated))
