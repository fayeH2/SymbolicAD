import numpy as np

from neurosymbolic_fsad.fuzzy_logic import FuzzyLogicProgram
from neurosymbolic_fsad.transistor_logic import (
    FINAL_FUSION_AST,
    _atom_llm_vote,
    _llm_vote_rates,
    _logic_ast,
    execute_grounded_neural_symbolic,
    preprocess_mask,
    preprocess_rgb,
)


def test_llm_family_vote_unions_rotation_specific_canonical_candidates():
    bank = {
        "expected_support_count": 4,
        "candidates": [
            {
                "rule": {"predicate": "ordered_spacing"},
                "occurrence_rate_among_expected_supports": 0.5,
                "sample_ids": ["s0", "s1"],
            },
            {
                "rule": {"predicate": "mirror_symmetry"},
                "occurrence_rate_among_expected_supports": 0.5,
                "sample_ids": ["s2", "s3"],
            },
        ],
    }
    rates = _llm_vote_rates(bank)
    assert rates["ordered_spacing"] == 0.5
    assert rates["mirror_symmetry"] == 0.5
    assert _atom_llm_vote("lead_symmetry", rates, bank) == 1.0


def test_empty_selected_rule_set_explicitly_abstains():
    ast = _logic_ast([])
    program = FuzzyLogicProgram(
        ast, (), special_atoms=("grounding", "symbolic_abstain")
    )
    value, trace = program.evaluate({"symbolic_abstain": 0.0})
    assert value == 0.0
    assert trace == [{
        "node_id": "symbolic.abstain_zero",
        "op": "ATOM",
        "atom": "symbolic_abstain",
        "input_node_ids": [],
        "output": 0.0,
    }]


def test_final_fusion_executes_neural_grounding_and_symbolic_and_or():
    value, trace = execute_grounded_neural_symbolic(0.2, 0.8)
    assert np.isclose(value, 0.328)
    assert [row["op"] for row in trace] == ["ATOM", "ATOM", "ATOM", "AND", "OR"]
    assert trace[-1]["node_id"] == "final.neuro_symbolic_anomaly"
    assert FINAL_FUSION_AST["op"] == "OR"


def test_final_fusion_runs_pixelwise_with_same_ast():
    baseline = np.asarray([[0.0, 0.2], [0.6, 1.0]], dtype=np.float32)
    symbolic = np.asarray([[1.0, 0.8], [0.5, 1.0]], dtype=np.float32)
    result, _ = execute_grounded_neural_symbolic(baseline, symbolic)
    expected = baseline + baseline * symbolic - baseline * baseline * symbolic
    np.testing.assert_allclose(result, expected, atol=1e-7)


def test_rule_preprocessing_aligns_rgb_and_mask_frames():
    rgb = np.zeros((512, 640, 3), dtype=np.uint8)
    mask = np.zeros((512, 640), dtype=np.uint8)
    mask[100:200, 200:300] = 255
    assert preprocess_rgb(rgb).shape == (256, 256, 3)
    prepared_mask = preprocess_mask(mask)
    assert prepared_mask.shape == (256, 256)
    assert prepared_mask.dtype == bool
    assert prepared_mask.any()
