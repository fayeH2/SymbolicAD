import numpy as np

from symbolicAD.config import load_config
from symbolicAD.grounding.relation_grounder import calibrate_atom_results
from symbolicAD.grounding.support_memory import SupportMemory
from symbolicAD.localization.fusion import normalize_logic_map
from symbolicAD.models.baseline_adapter import (
    _apply_whitening,
    _fit_whitening,
    _local_smooth,
)
from symbolicAD.rules.executor import AtomicResult


def test_whitening_helpers_preserve_shapes():
    rng = np.random.default_rng(1)
    bank = rng.normal(size=(64, 16)).astype(np.float32)
    whitening = _fit_whitening(bank, max_components=8, epsilon=1e-3)
    transformed = _apply_whitening(bank, whitening)
    assert transformed.shape == (64, 8)
    features = rng.normal(size=(2, 16, 4, 4)).astype(np.float32)
    mapped = _apply_whitening(features, whitening, channel_axis=1)
    assert mapped.shape == (2, 8, 4, 4)


def test_local_smooth_keeps_shape_and_reduces_noise():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(1, 4, 8, 8)).astype(np.float32)
    smoothed = _local_smooth(features, radius=1, weight=0.5)
    assert smoothed.shape == features.shape
    assert smoothed.std() < features.std()


def test_support_calibration_keeps_high_normal_truth():
    config = load_config()
    memory = SupportMemory(category="demo", grid_size=(2, 2))
    memory.atom_stats = {
        "A": {"q05": 0.8, "q50": 0.9, "q95": 1.0},
        "B": {"q05": 0.0, "q50": 1.0, "q95": 1.0},
    }
    results = {
        "A": AtomicResult("A", "Exist(X)", "Exist", 0.95, 0.05, np.zeros((2, 2)), True),
        "B": AtomicResult("B", "CountEq(X,1)", "CountEq", 1.0, 0.0, np.zeros((2, 2)), True),
    }
    calibrated = calibrate_atom_results(results, memory, config)
    assert calibrated["A"].truth == 1.0
    assert calibrated["B"].truth == 1.0
    results["A"].truth = 0.5
    calibrated = calibrate_atom_results(results, memory, config)
    assert calibrated["A"].truth == 0.0
    assert calibrated["A"].evidence["raw_truth"] == 0.5


def test_normalize_logic_map_uses_support_noise_floor():
    values = np.asarray([[0.0, 0.1, 0.3]], dtype=np.float32)
    normalized = normalize_logic_map(values, {"q_low": 0.1, "q_high": 0.3})
    assert np.allclose(normalized, [[0.0, 0.0, 1.0]])


def test_default_protocol_is_strict_support_only():
    config = load_config()
    assert config.model.feature_source != "auto"
