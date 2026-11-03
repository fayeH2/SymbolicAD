import numpy as np

from symbolicAD.grounding.instance_extractor import extract_instances
from symbolicAD.localization.fusion import (
    final_image_score,
    fuse_maps,
    logic_image_score,
    normalize_map,
    topk_mean,
)


def test_instance_extraction_two_blobs():
    component_map = np.zeros((10, 10), dtype=np.float32)
    component_map[1:3, 1:3] = 0.9
    component_map[6:8, 6:8] = 0.8
    features = np.random.default_rng(0).normal(size=(4, 10, 10)).astype(np.float32)
    instances = extract_instances(
        component_map, features, concept="A", threshold=0.5, min_area=2
    )
    assert len(instances) == 2
    assert all(instance.mask.shape == (10, 10) for instance in instances)
    assert all(len(instance.feature) == 4 for instance in instances)


def test_normalize_map_is_support_calibrated():
    values = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    normalized = normalize_map(values, q_low=0.0, q_high=2.0)
    assert np.allclose(normalized, [0.0, 0.0, 0.5, 1.0])


def test_fusion_and_scores():
    base = np.full((4, 4), 0.2, dtype=np.float32)
    logic = np.full((4, 4), 0.6, dtype=np.float32)
    fused = fuse_maps(base, logic, lambda_logic=0.5)
    assert fused.max() <= 1.0 and fused.min() >= base.min()
    assert abs(logic_image_score(logic, root_truth=0.4) - 0.6) < 1e-5
    assert abs(final_image_score(0.2, 0.6, beta=0.5) - 0.4) < 1e-5
    assert topk_mean(np.asarray([[0.1, 0.2, 0.9]], dtype=np.float32)).shape == (1,)
