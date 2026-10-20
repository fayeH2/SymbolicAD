import numpy as np
import pytest

from neurosymbolic_fsad.fuzzy_logic import LogicValidationError
from neurosymbolic_fsad.symbolicdet_evolution import (
    EvolutionConfig,
    SymbolicEvolutionSearch,
    anomaly_ast,
    atom_names,
    counterfactual_search_data,
    evaluate_ast_matrix,
    normalize_normality_ast,
    support_truth_matrix,
    validate_normality_ast,
)


def test_normality_ast_executes_each_selected_atom_once():
    normal = {
        "op": "AND",
        "args": [
            {"op": "ATOM", "name": "count"},
            {"op": "ATOM", "name": "geometry"},
        ],
    }
    assert validate_normality_ast(normal, ["count", "geometry"]) == (
        "count", "geometry"
    )
    values = {
        "count": np.asarray([1.0, 0.0, 1.0]),
        "geometry": np.asarray([1.0, 1.0, 0.0]),
    }
    np.testing.assert_allclose(
        evaluate_ast_matrix(anomaly_ast(normal), values), [0.0, 1.0, 1.0]
    )


def test_duplicate_or_unknown_atoms_fail_closed():
    duplicate = {
        "op": "OR",
        "args": [
            {"op": "ATOM", "name": "count"},
            {"op": "ATOM", "name": "count"},
        ],
    }
    with pytest.raises(LogicValidationError):
        validate_normality_ast(duplicate, ["count"])
    with pytest.raises(LogicValidationError):
        validate_normality_ast({"op": "ATOM", "name": "query_label"}, ["count"])


def test_qwen_not_args_spelling_is_normalized_strictly():
    emitted = {
        "op": "NOT",
        "args": [{"op": "ATOM", "name": "count"}],
    }
    assert normalize_normality_ast(emitted) == {
        "op": "NOT",
        "arg": {"op": "ATOM", "name": "count"},
    }
    with pytest.raises(LogicValidationError):
        normalize_normality_ast({"op": "ATOM", "name": "count", "extra": 1})


def test_support_truths_and_counterfactuals_are_support_only():
    audits = [
        {
            "id": "count",
            "leave_one_out_values": [0.0, 0.2, 0.6],
            "upper": 0.2,
            "scale": 0.2,
        },
        {
            "id": "geometry",
            "leave_one_out_values": [0.1, 0.1, 0.1],
            "upper": 0.1,
            "scale": 0.2,
        },
    ]
    truths = support_truth_matrix(audits, ["count", "geometry"])
    assert truths.shape == (3, 2)
    assert truths[2, 0] < 1.0
    X, y, provenance = counterfactual_search_data(
        truths, ["count", "geometry"], seed=7,
        variants_per_atom=2, pair_variants=1,
    )
    assert X.shape[0] == len(y) == len(provenance)
    assert np.sum(y == 0) == 3
    assert np.sum(y == 1) == 15
    assert all("violated_atoms" in row for row in provenance)


def test_evolution_is_deterministic_and_records_llm_injection():
    normal = np.ones((8, 2), dtype=float)
    X, y, _ = counterfactual_search_data(
        normal, ["count", "geometry"], seed=11,
        variants_per_atom=2, pair_variants=1,
    )
    calls = []

    def advisor(generation, top):
        calls.append((generation, len(top)))
        return ([{
            "op": "AND",
            "args": [
                {"op": "ATOM", "name": "count"},
                {"op": "ATOM", "name": "geometry"},
            ],
        }], {"raw_output": "unit-test", "query_or_test_information_supplied": False})

    config = EvolutionConfig(
        population_size=20,
        generations=7,
        elite_count=3,
        early_stop_patience=20,
        llm_feedback_interval=2,
        random_seed=19,
    )
    first = SymbolicEvolutionSearch(
        ["count", "geometry"], config, llm_advisor=advisor
    ).run(X, y)
    second = SymbolicEvolutionSearch(
        ["count", "geometry"], config, llm_advisor=None
    ).run(X, y)
    assert calls
    assert first["llm_feedback"]
    assert set(first["best_active_atoms"]).issubset({"count", "geometry"})
    validate_normality_ast(first["best_normality_ast"], ["count", "geometry"])
    validate_normality_ast(second["best_normality_ast"], ["count", "geometry"])
    assert first["generation_history"][0]["best_fitness"] == pytest.approx(
        second["generation_history"][0]["best_fitness"]
    )
