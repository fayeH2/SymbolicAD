import numpy as np
import pytest

from neurosymbolic_fsad.fuzzy_logic import LogicValidationError
from neurosymbolic_fsad.ovd_grounder import (
    OpenVocabularySymbolicEngine,
    _relational_signature,
    derive_prompt_candidates,
)


def _rows(centers, sizes):
    result = []
    for center, size in zip(centers, sizes):
        width, height = size
        result.append({
            "center": np.asarray(center, dtype=np.float32),
            "size": np.asarray(size, dtype=np.float32),
            "aspect": float(np.log(width / height)),
            "box": [0.0, 0.0, 1.0, 1.0],
        })
    return result


def test_relational_signature_is_similarity_invariant():
    centers = np.asarray([[0.1, 0.2], [0.4, 0.2], [0.9, 0.2]], dtype=np.float32)
    sizes = np.asarray([[0.10, 0.20], [0.12, 0.20], [0.08, 0.20]], dtype=np.float32)
    angle = np.deg2rad(37.0)
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ], dtype=np.float32)
    transformed_centers = (centers @ rotation.T) * 2.7 + np.asarray([3.0, -1.0])
    transformed_sizes = sizes * 2.7
    before = _relational_signature(_rows(centers, sizes))
    after = _relational_signature(_rows(transformed_centers, transformed_sizes))
    np.testing.assert_allclose(before["gaps"], after["gaps"], atol=1e-5)
    np.testing.assert_allclose(
        before["relative_log_area"], after["relative_log_area"], atol=1e-5
    )
    np.testing.assert_allclose(
        before["relative_aspect"], after["relative_aspect"], atol=1e-5
    )
    assert abs(before["line_deviation"] - after["line_deviation"]) < 1e-5


def test_relational_signature_detects_changed_spacing():
    sizes = [[0.1, 0.2]] * 3
    normal = _relational_signature(_rows([[0.1, 0.2], [0.4, 0.2], [0.9, 0.2]], sizes))
    changed = _relational_signature(_rows([[0.1, 0.2], [0.75, 0.2], [0.9, 0.2]], sizes))
    assert float(np.mean(np.abs(normal["gaps"] - changed["gaps"]))) > 0.3


def test_prompt_candidates_are_generic_and_deduplicated():
    candidates = derive_prompt_candidates({
        "id": "transistor_pins",
        "visual_description": "three metallic curved pins",
        "detector_prompts": ["pin", "pin"],
    })
    assert candidates[0] == "pin"
    assert len(candidates) == len(set(candidates))
    assert "transistor pins" in candidates


def test_saved_non_monotone_program_fails_closed():
    class Detector:
        nms_iou = 0.5

    atoms = ["ovd_count_eq", "ovd_arrangement_eq"]
    library = {
        "grounder": {
            "prompt": "part",
            "threshold": 0.2,
            "expected_count": 2,
            "arrangement": "row",
            "prototype": {
                "gaps": [1.0],
                "line_deviation": 0.0,
                "relative_log_area": [0.0, 0.0],
                "relative_aspect": [0.0, 0.0],
                "absolute_center": [[0.25, 0.5], [0.75, 0.5]],
                "absolute_size": [[0.2, 0.2], [0.2, 0.2]],
            },
        },
        "rules": [{"id": name} for name in atoms],
        "logic": {
            "normality_ast": {
                "op": "XOR",
                "args": [{"op": "ATOM", "name": name} for name in atoms],
            },
            "anomaly_ast": {"op": "ATOM", "name": atoms[0]},
        },
    }
    with pytest.raises(LogicValidationError, match="positive and monotone"):
        OpenVocabularySymbolicEngine(library, Detector())
