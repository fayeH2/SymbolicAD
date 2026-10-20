import numpy as np
import pytest

from neurosymbolic_fsad.fuzzy_logic import (
    FuzzyLogicProgram,
    LogicValidationError,
    execute_logic_ast,
    fuzzy_and,
    fuzzy_implies,
    fuzzy_not,
    fuzzy_or,
    validate_logic_ast,
)


def test_product_fuzzy_scalar_operators_are_bounded_and_exact():
    assert fuzzy_not(0.2) == pytest.approx(0.8)
    assert fuzzy_and(0.5, 0.4) == pytest.approx(0.2)
    assert fuzzy_or(0.5, 0.4) == pytest.approx(0.7)
    assert fuzzy_implies(0.8, 0.25) == pytest.approx(0.4)
    assert fuzzy_and(-1.0, 2.0) == pytest.approx(0.0)
    assert fuzzy_or(-1.0, 2.0) == pytest.approx(1.0)


def test_array_operators_broadcast_and_keep_small_or_values_stable():
    left = np.asarray([0.0, 0.5, 1.0])
    np.testing.assert_allclose(fuzzy_and(left, 0.2), [0.0, 0.1, 0.2])
    np.testing.assert_allclose(fuzzy_or(left, 0.2), [0.2, 0.6, 1.0])
    tiny = fuzzy_or(1e-16, 2e-16)
    assert tiny == pytest.approx(3e-16, rel=1e-12)
    assert np.all(np.isfinite(fuzzy_implies(left, np.asarray([[0.1], [0.9]]))))


def test_ast_executes_every_active_rule_once_and_traces_every_node():
    ast = {
        "id": "final_anomaly",
        "op": "OR",
        "args": [
            {"id": "neural_not", "op": "NOT", "arg": {"atom": "neural"}},
            {
                "id": "symbolic_violation",
                "op": "NOT",
                "arg": {
                    "id": "normal_rules",
                    "op": "AND",
                    "args": [
                        {"atom": "count"},
                        {
                            "op": "IMPLIES",
                            "args": [{"atom": "grounding"}, {"atom": "geometry"}],
                        },
                    ],
                },
            },
        ],
    }
    neural = np.asarray([0.9, 0.2])
    grounding = np.asarray([1.0, 0.5])
    count = np.asarray([0.8, 0.7])
    geometry = np.asarray([0.75, 0.4])
    program = FuzzyLogicProgram(ast, ["count", "geometry"])
    output, trace = program.evaluate({
        "neural": neural,
        "grounding": grounding,
        "count": count,
        "geometry": geometry,
    })

    implied = fuzzy_implies(grounding, geometry)
    expected = fuzzy_or(fuzzy_not(neural), fuzzy_not(fuzzy_and(count, implied)))
    np.testing.assert_allclose(output, expected)
    assert program.validation.active_atom_counts == {"count": 1, "geometry": 1}
    assert program.validation.special_atom_counts == {"grounding": 1, "neural": 1}
    assert len(trace) == program.validation.node_count == 9
    assert trace[-1]["node_id"] == "final_anomaly"
    assert trace[-1]["op"] == "OR"
    assert [entry["atom"] for entry in trace if entry["op"] == "ATOM"] == [
        "neural", "count", "grounding", "geometry",
    ]
    assert all("output" in entry for entry in trace)


@pytest.mark.parametrize(
    "ast, active, message",
    [
        ({"atom": "r1"}, ["r1", "r2"], "missing active atoms"),
        (
            {"op": "AND", "args": [{"atom": "r1"}, {"atom": "r1"}]},
            ["r1"],
            "duplicated active atoms",
        ),
        ({"atom": "inactive"}, ["r1"], "Inactive or unknown atom"),
        (
            {"op": "OR", "args": [{"atom": "neural"}, {"atom": "neural"}]},
            [],
            "duplicated special atoms",
        ),
    ],
)
def test_reachability_rejects_missing_duplicate_and_inactive_atoms(ast, active, message):
    with pytest.raises(LogicValidationError, match=message):
        validate_logic_ast(ast, active)


def test_special_neural_and_grounding_atoms_are_allowed_but_not_required_active():
    ast = {
        "op": "AND",
        "args": [{"atom": "r1"}, {"atom": "neural"}, {"atom": "grounding"}],
    }
    report = validate_logic_ast(ast, ["r1"])
    assert report.execution_atoms == ("r1", "neural", "grounding")


def test_execution_rejects_missing_or_nonfinite_truths():
    ast = {"op": "AND", "args": [{"atom": "r1"}, {"atom": "neural"}]}
    with pytest.raises(ValueError, match="Missing truth value for atom: neural"):
        execute_logic_ast(ast, {"r1": 0.5}, active_atoms=["r1"])
    with pytest.raises(ValueError, match="must contain only finite values"):
        execute_logic_ast(
            ast, {"r1": np.nan, "neural": 0.5}, active_atoms=["r1"]
        )


def test_invalid_arity_duplicate_ids_and_shared_nodes_are_rejected():
    with pytest.raises(LogicValidationError, match="NOT requires exactly one"):
        validate_logic_ast({"op": "NOT", "args": []}, [])
    with pytest.raises(LogicValidationError, match="Duplicate AST node id"):
        validate_logic_ast(
            {
                "op": "AND",
                "args": [
                    {"id": "same", "atom": "r1"},
                    {"id": "same", "atom": "r2"},
                ],
            },
            ["r1", "r2"],
        )
    leaf = {"atom": "r1"}
    with pytest.raises(LogicValidationError, match="cannot be shared or cyclic"):
        validate_logic_ast({"op": "OR", "args": [leaf, leaf]}, ["r1"])
