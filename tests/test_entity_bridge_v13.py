"""Entity Verification、INP residual 与单调融合之间的边界回归。"""

import numpy as np
import torch

from symbolicAD.entity_bridge import EntityBridge, _roi_token
from symbolicAD.inpformer_adapter import _project_feature_grids
from symbolicAD.reasoning.constraint_executor import ConstraintExecution
from symbolicAD.reasoning.violation_result import RuleStatus


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


def _constraint(status, score, anomaly_map):
    return ConstraintExecution(
        status=status, violation_score=score, evidence_confidence=1.0,
        anomaly_map=anomaly_map, rule_results=tuple(),
        satisfied_rule_ids=tuple(), violated_rule_ids=tuple(),
        unknown_rule_ids=tuple(),
    )


def test_low_inp_residual_does_not_erase_reliable_logic_violation():
    visual = np.full((8, 8), 0.01, dtype=np.float32)
    logic_map = np.zeros_like(visual)
    logic_map[2:5, 2:5] = 0.9
    result = EntityBridge().fuse(
        visual, 0.01,
        _constraint(RuleStatus.VIOLATED, 0.9, logic_map),
        confirmation_threshold=0.5,
    )
    assert result.anomaly_confirmation.confirmed is False
    assert result.logic_status == RuleStatus.VIOLATED
    assert result.logic_score == 0.9
    assert float(result.fused_map[2:5, 2:5].mean()) > 0.9


def test_unknown_logic_leaves_visual_baseline_exactly_unchanged():
    visual = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    malformed_nonzero_map = np.ones_like(visual)
    result = EntityBridge().fuse(
        visual, 0.7,
        _constraint(RuleStatus.UNKNOWN, 0.0, malformed_nonzero_map),
    )
    np.testing.assert_array_equal(result.fused_map, visual)
    assert result.logic_score == 0.0
    assert result.diagnostics["unknown_contribution"] == 0.0
