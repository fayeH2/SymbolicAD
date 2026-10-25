"""Map-frame regressions for LOCO localization evaluation."""

import numpy as np
import pytest

from symbolicAD.evaluation.spro import (
    MapFrame,
    ScoreMapGeometry,
    _strict_counts_above,
    fuse_in_original_frame,
    project_score_map_to_original,
)


def test_center_crop_inverse_zeros_unobserved_border_on_non_square_image():
    restored = project_score_map_to_original(
        np.ones((2, 2), dtype=np.float32), (80, 100),
        ScoreMapGeometry(MapFrame.MODEL_CENTER_CROP, input_size=8, crop_size=4),
    )
    assert restored.shape == (80, 100)
    assert restored[40, 50] == pytest.approx(1.0)
    assert restored[0, 0] == 0.0
    assert restored[-1, -1] == 0.0


def test_resized_map_without_frame_metadata_fails_closed():
    with pytest.raises(ValueError, match="geometry is required"):
        project_score_map_to_original(np.ones((16, 16)), (80, 100))


def test_full_image_and_center_crop_frames_are_not_interchangeable():
    values = np.ones((4, 4), dtype=np.float32)
    full = project_score_map_to_original(
        values, (80, 100), ScoreMapGeometry(MapFrame.FULL_IMAGE_NORMALIZED)
    )
    crop = project_score_map_to_original(
        values, (80, 100),
        ScoreMapGeometry(MapFrame.MODEL_CENTER_CROP, input_size=8, crop_size=4),
    )
    assert full[0, 0] == 1.0
    assert crop[0, 0] == 0.0
    assert not np.array_equal(full, crop)


def test_original_frame_fusion_projects_before_probability_or():
    baseline = np.ones((2, 2), dtype=np.float32) * 0.5
    logic = np.ones((2, 2), dtype=np.float32) * 0.4
    fused = fuse_in_original_frame(
        baseline,
        ScoreMapGeometry(MapFrame.MODEL_CENTER_CROP, input_size=8, crop_size=4),
        logic,
        ScoreMapGeometry(MapFrame.FULL_IMAGE_NORMALIZED),
        (80, 100),
    )
    assert fused[40, 50] == pytest.approx(0.7, abs=1e-5)
    # The baseline has no field of view at the corner, so only logic remains.
    assert fused[0, 0] == pytest.approx(0.4, abs=1e-5)


def test_strict_counts_above_preserves_equal_threshold_semantics():
    values = np.asarray([0.0, 1.0, 1.5, 2.0, 4.0])
    thresholds = np.asarray([1.0, 2.0, 3.0])
    assert _strict_counts_above(values, thresholds).tolist() == [3, 1, 1]
