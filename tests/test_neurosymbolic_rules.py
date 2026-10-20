"""Tests for explicit neuro-symbolic rule combinations."""

import numpy as np
import pytest

from neurosymbolic_fsad.fuzzy_logic import fuzzy_xor
from neurosymbolic_fsad.neurosymbolic_rules import (
    NeuroSymbolicRule,
    RuleCombination,
    NeuroSymbolicEngine,
    execute_neuro_symbolic,
    create_xor_combinations,
    create_and_combinations,
    create_implication_chain,
)


def test_fuzzy_xor_scalar():
    assert fuzzy_xor(0.0, 0.0) == pytest.approx(0.0)
    assert fuzzy_xor(1.0, 0.0) == pytest.approx(1.0)
    assert fuzzy_xor(0.0, 1.0) == pytest.approx(1.0)
    assert fuzzy_xor(1.0, 1.0) == pytest.approx(0.0)
    assert fuzzy_xor(0.5, 0.5) == pytest.approx(0.5)
    assert fuzzy_xor(0.8, 0.2) == pytest.approx(0.68)


def test_fuzzy_xor_array():
    left = np.asarray([0.0, 0.5, 1.0])
    right = np.asarray([1.0, 0.5, 0.0])
    expected = np.asarray([1.0, 0.5, 1.0])
    np.testing.assert_allclose(fuzzy_xor(left, right), expected)


def test_fuzzy_xor_broadcast():
    left = np.asarray([[0.0], [0.5], [1.0]])
    right = np.asarray([0.0, 1.0])
    expected = np.asarray([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    np.testing.assert_allclose(fuzzy_xor(left, right), expected)


def test_neurosymbolic_engine_xor():
    rules = [
        NeuroSymbolicRule("rule1", "First rule", 0.9),
        NeuroSymbolicRule("rule2", "Second rule", 0.3),
        NeuroSymbolicRule("rule3", "Third rule", 0.8),
    ]
    combinations = [
        RuleCombination("xor_12", "XOR", ["rule1", "rule2"], "rule1 XOR rule2"),
    ]
    engine = NeuroSymbolicEngine(rules, combinations)
    result = engine.evaluate(
        {"rule1": 0.9, "rule2": 0.3, "rule3": 0.8},
        neural_anomaly=0.4,
        grounding_confidence=0.9,
    )
    assert "final_score" in result
    assert "symbolic_score" in result
    assert "violated_rules" in result
    assert "trace" in result
    assert "rule2" in result["violated_rules"]


def test_neurosymbolic_engine_and():
    rules = [
        NeuroSymbolicRule("rule1", "First rule", 0.2),
        NeuroSymbolicRule("rule2", "Second rule", 0.3),
    ]
    combinations = [
        RuleCombination("and_12", "AND", ["rule1", "rule2"], "rule1 AND rule2"),
    ]
    engine = NeuroSymbolicEngine(rules, combinations)
    result = engine.evaluate(
        {"rule1": 0.2, "rule2": 0.3},
        neural_anomaly=0.5,
        grounding_confidence=1.0,
    )
    assert result["final_score"] > 0.5


def test_neurosymbolic_engine_or():
    rules = [
        NeuroSymbolicRule("rule1", "First rule", 0.9),
        NeuroSymbolicRule("rule2", "Second rule", 0.8),
    ]
    combinations = [
        RuleCombination("or_12", "OR", ["rule1", "rule2"], "rule1 OR rule2"),
    ]
    engine = NeuroSymbolicEngine(rules, combinations)
    result = engine.evaluate(
        {"rule1": 0.9, "rule2": 0.8},
        neural_anomaly=0.1,
        grounding_confidence=1.0,
    )
    assert "final_score" in result


def test_neurosymbolic_engine_implies():
    rules = [
        NeuroSymbolicRule("rule1", "First rule", 0.2),
        NeuroSymbolicRule("rule2", "Second rule", 0.8),
    ]
    combinations = [
        RuleCombination("implies_12", "IMPLIES", ["rule1", "rule2"], "rule1 -> rule2"),
    ]
    engine = NeuroSymbolicEngine(rules, combinations)
    result = engine.evaluate(
        {"rule1": 0.2, "rule2": 0.8},
        neural_anomaly=0.3,
        grounding_confidence=1.0,
    )
    assert "final_score" in result


def test_create_xor_combinations():
    rule_ids = ["r1", "r2", "r3"]
    combinations = create_xor_combinations(rule_ids)
    assert len(combinations) == 3
    assert combinations[0].rule_ids == ["r1", "r2"]
    assert combinations[1].rule_ids == ["r1", "r3"]
    assert combinations[2].rule_ids == ["r2", "r3"]
    assert all(c.operator == "XOR" for c in combinations)


def test_create_and_combinations():
    rule_ids = ["r1", "r2", "r3"]
    combinations = create_and_combinations(rule_ids)
    assert len(combinations) == 3
    assert combinations[0].rule_ids == ["r1", "r2"]
    assert combinations[1].rule_ids == ["r1", "r3"]
    assert combinations[2].rule_ids == ["r2", "r3"]
    assert all(c.operator == "AND" for c in combinations)


def test_create_implication_chain():
    rule_ids = ["r1", "r2", "r3"]
    combinations = create_implication_chain(rule_ids)
    assert len(combinations) == 2
    assert combinations[0].rule_ids == ["r1", "r2"]
    assert combinations[1].rule_ids == ["r2", "r3"]
    assert all(c.operator == "IMPLIES" for c in combinations)


def test_execute_neuro_symbolic_convenience():
    rules = [
        NeuroSymbolicRule("lead_count", "Lead count normal", 0.95),
        NeuroSymbolicRule("lead_geometry", "Lead geometry normal", 0.4),
        NeuroSymbolicRule("lead_symmetry", "Lead symmetry normal", 0.85),
    ]
    combinations = create_xor_combinations(["lead_count", "lead_geometry", "lead_symmetry"])
    result = execute_neuro_symbolic(
        rules,
        combinations,
        neural_anomaly=0.6,
        grounding_confidence=0.9,
    )
    assert isinstance(result.final_score, float)
    assert isinstance(result.symbolic_score, float)
    assert isinstance(result.combination_scores, dict)
    assert len(result.combination_scores) == 3


def test_neurosymbolic_with_spatial_maps():
    rules = [
        NeuroSymbolicRule("rule1", "First rule", 0.8, spatial_map=np.ones((16, 16)) * 0.8),
        NeuroSymbolicRule("rule2", "Second rule", 0.3, spatial_map=np.ones((16, 16)) * 0.3),
    ]
    combinations = [
        RuleCombination("xor_12", "XOR", ["rule1", "rule2"], "rule1 XOR rule2"),
    ]
    engine = NeuroSymbolicEngine(rules, combinations)
    rule_maps = {r.rule_id: r.spatial_map for r in rules if r.spatial_map is not None}
    result = engine.evaluate(
        {"rule1": 0.8, "rule2": 0.3},
        neural_anomaly=0.5,
        grounding_confidence=1.0,
        rule_maps=rule_maps,
    )
    assert result["spatial_map"] is not None
    assert result["spatial_map"].shape == (16, 16)


def test_multiple_combinations():
    rules = [
        NeuroSymbolicRule("r1", "Rule 1", 0.9),
        NeuroSymbolicRule("r2", "Rule 2", 0.4),
        NeuroSymbolicRule("r3", "Rule 3", 0.3),
    ]
    combinations = [
        RuleCombination("xor_12", "XOR", ["r1", "r2"]),
        RuleCombination("and_23", "AND", ["r2", "r3"]),
        RuleCombination("or_all", "OR", ["r1", "r2", "r3"]),
    ]
    engine = NeuroSymbolicEngine(rules, combinations)
    result = engine.evaluate(
        {"r1": 0.9, "r2": 0.4, "r3": 0.3},
        neural_anomaly=0.5,
        grounding_confidence=1.0,
    )
    assert "final_score" in result
    assert result["trace"][-1]["op"] == "OR"
    assert result["trace"][-1]["node_id"] == "final.neuro_symbolic_anomaly"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])