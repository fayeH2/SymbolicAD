import numpy as np
import torch
from types import SimpleNamespace

from neurosymbolic_fsad.entity_bridge import EntityTokenBridge, _roi_token
from neurosymbolic_fsad.orthogonal_symbolic import GenericLogicInference
from neurosymbolic_fsad.inpformer_adapter import _project_feature_grids


def test_project_feature_grids_accepts_inp_bchw_layout():
    source = [torch.randn(2, 12, 7, 7), torch.randn(2, 12, 7, 7)]
    target = [value + 0.05 * torch.randn_like(value) for value in source]
    output = _project_feature_grids(source, target, projection_dim=4)
    assert tuple(output.shape) == (2, 2, 8, 7, 7)
    assert torch.isfinite(output).all()


def test_roi_token_is_normalized_and_spatially_sensitive():
    grids = np.zeros((2, 4, 8, 8), dtype=np.float32)
    grids[:, :, :4, :4] = 1.0
    first = _roi_token(grids, [0.0, 0.0, 0.5, 0.5])
    second = _roi_token(grids, [0.5, 0.5, 1.0, 1.0])
    assert np.isclose(np.linalg.norm(first), 1.0)
    assert not np.allclose(first, second)


def test_non_count_rule_does_not_require_local_inp_residual_to_trigger():
    size = 4
    localizer = np.zeros((size, size), dtype=np.float32)
    localizer[:2, :2] = 1.0
    symbolic = GenericLogicInference(
        symbolic_anomaly=1.0,
        symbolic_map=localizer,
        facts={"layout_error": 1.0},
        atoms={"layout": {
            "violation": 1.0,
            "fact": "layout_error",
        }},
        localization_maps={"layout": localizer},
        trace=[],
        diagnostics={
            "detections": [
                {"box": [0.0, 0.0, 0.5, 1.0]},
                {"box": [0.5, 0.0, 1.0, 1.0]},
            ],
        },
    )

    class Program:
        @staticmethod
        def evaluate(truths):
            return 1.0 - truths["layout"], []

    engine = SimpleNamespace(
        evaluate=lambda image: symbolic,
        program=Program(),
        grounder=SimpleNamespace(
            expected_count=2,
            prototype={
                "absolute_center": [[0.25, 0.5], [0.75, 0.5]],
                "absolute_size": [[0.5, 1.0], [0.5, 1.0]],
            },
        ),
    )
    state = {
        "slot_token_prototypes": [[1.0, 0.0], [1.0, 0.0]],
        "slot_distance_thresholds": [0.1, 0.1],
        "slot_distance_scales": [0.1, 0.1],
        "support_score_upper": 0.5,
        "entity_set_distance_threshold": 0.1,
        "entity_set_distance_scale": 0.1,
        "rule_reliability": {"layout": 1.0},
        "rule_local_score_upper": {"layout": 0.5},
        "map_size": size,
        "rescue_gain": 0.1,
        "suppression_gain": 0.0,
    }
    bridge = EntityTokenBridge(engine, state)
    baseline = np.full((size, size), 0.1, dtype=np.float32)
    feature_grids = np.ones((1, 1, 2, 2), dtype=np.float32)

    result = bridge.infer(None, feature_grids, baseline, baseline_score=0.1)

    assert result.diagnostics["corrected_rules"]["layout"][
        "entity_verification_confidence"
    ] == 1.0
    assert result.diagnostics["corrected_rules"]["layout"][
        "anomaly_confirmation_confirmed"
    ] is False
    assert result.diagnostics["corrected_rules"]["layout"][
        "corrected_violation"
    ] == 1.0
    assert float(result.anomaly_map[:2, :2].mean()) > 0.1
