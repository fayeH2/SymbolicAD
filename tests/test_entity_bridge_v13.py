import numpy as np
import torch

from neurosymbolic_fsad.entity_bridge import _roi_token
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
