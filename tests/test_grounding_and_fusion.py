import numpy as np

from symbolicAD.grounding.instance_extractor import component_spatial_prior, extract_instances
from symbolicAD.grounding.support_memory import _text_seed_prototypes
from symbolicAD.localization.responsibility import build_logic_heatmap
from symbolicAD.localization.fusion import (
    final_image_score,
    fuse_maps,
    logic_image_score,
    normalize_map,
    topk_mean,
)
from symbolicAD.config import load_config
from symbolicAD.models.baseline_adapter import CLIPFewShotBaseline
from symbolicAD.rules.executor import AtomicResult, LogicResult
from symbolicAD.rules.schema import RuleKB


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


def test_component_spatial_prior_removes_black_background_and_applies_position():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[10:95, 30:70] = 180
    bottle = component_spatial_prior(
        image, (10, 10), position="center", role="Container"
    )
    upper = component_spatial_prior(
        image, (10, 10), position="upper_center", role="Label"
    )
    assert bottle[:, :2].sum() == 0
    assert bottle[:, 8:].sum() == 0
    assert upper[:2].sum() == 0
    assert upper[7:].sum() == 0
    assert 0 < upper.sum() < bottle.sum()


def test_text_seed_prototypes_use_top_raw_cosine_patches():
    patches = np.zeros((1, 2, 2, 2), dtype=np.float32)
    patches[0, :, 0, 0] = [1.0, 0.0]
    patches[0, :, 1, 1] = [0.0, 1.0]
    text_maps = np.asarray([[[0.31, 0.10], [0.20, 0.42]]], dtype=np.float32)
    prototypes = _text_seed_prototypes(patches, text_maps, max_prototypes=1)
    assert len(prototypes) == 1
    assert np.allclose(prototypes[0], [0.0, 1.0])


def test_text_seed_prototypes_respect_component_candidate_mask():
    patches = np.zeros((1, 2, 1, 2), dtype=np.float32)
    patches[0, :, 0, 0] = [1.0, 0.0]
    patches[0, :, 0, 1] = [0.0, 1.0]
    text_maps = np.asarray([[[0.99, 0.40]]], dtype=np.float32)
    candidate_masks = np.asarray([[[0.0, 1.0]]], dtype=np.float32)
    prototypes = _text_seed_prototypes(
        patches,
        text_maps,
        max_prototypes=1,
        candidate_masks=candidate_masks,
    )
    assert np.allclose(prototypes[0], [0.0, 1.0])


def test_one_shot_baseline_calibration_excludes_self_patch():
    baseline = CLIPFewShotBaseline.__new__(CLIPFewShotBaseline)
    baseline.knn = 1
    baseline.topk_percent = 25.0
    baseline._stats_cache = {}
    baseline.support_patches = np.asarray(
        [[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=np.float32
    )
    maps = baseline._leave_one_out_maps()
    assert maps.shape == (1, 1, 2)
    assert np.allclose(maps, 1.0)
    stats = baseline.support_statistics()
    assert stats["image_q_low"] == stats["image_q_high"] == 1.0


def test_position_aware_baseline_detects_patch_swap_hidden_by_global_bank():
    baseline = CLIPFewShotBaseline.__new__(CLIPFewShotBaseline)
    baseline.knn = 1
    baseline.spatial_radius = 0
    baseline.spatial_weight = 1.0
    support = np.asarray(
        [[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=np.float32
    )
    query = support[..., ::-1].copy()
    bank = support.transpose(0, 2, 3, 1).reshape(-1, 2)
    global_map = baseline._global_maps(query, bank)
    spatial_map = baseline._raw_maps(query, bank, spatial_support=support)
    assert np.allclose(global_map, 0.0)
    assert np.allclose(spatial_map, 1.0)


def test_normalize_map_is_support_calibrated():
    values = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    normalized = normalize_map(values, q_low=0.0, q_high=2.0)
    assert np.allclose(normalized, [0.0, 0.0, 0.5, 1.0])


def test_fusion_and_scores():
    base = np.full((4, 4), 0.2, dtype=np.float32)
    logic = np.full((4, 4), 0.6, dtype=np.float32)
    fused = fuse_maps(base, logic, lambda_logic=0.5)
    assert fused.max() <= 1.0 and fused.min() >= base.min()
    assert np.array_equal(fuse_maps(base, logic, lambda_logic=0.0), base)
    assert abs(logic_image_score(logic, root_truth=0.4) - 0.6) < 1e-5
    assert abs(final_image_score(0.2, 0.6, beta=0.5) - 0.4) < 1e-5
    assert abs(
        final_image_score(0.2, 0.6, beta=0.5, baseline_scale=0.1) - 0.23
    ) < 1e-5
    assert topk_mean(np.asarray([[0.1, 0.2, 0.9]], dtype=np.float32)).shape == (1,)


def test_logic_heatmap_only_projects_formula_active_failures():
    rule_kb = RuleKB.from_dict(
        {
            "category": "demo",
            "components": [{"name": "A", "prompts": ["a"]}],
            "atoms": [
                {"id": "R1", "expression": "Exist(A)"},
                {"id": "R2", "expression": "Exist(A)"},
            ],
            "formula": {"op": "OR", "children": ["R1", "R2"]},
        }
    )
    mask = np.ones((2, 2), dtype=np.float32)
    atom_results = {
        "R1": AtomicResult("R1", "Exist(A)", "Exist", 0.9, 0.1, mask, True),
        "R2": AtomicResult("R2", "Exist(A)", "Exist", 0.1, 0.9, mask, True),
    }
    logic_result = LogicResult(
        root_truth=0.9,
        logic_score=0.1,
        failed_atoms=[],
        atom_truths={"R1": 0.9, "R2": 0.1},
        active_failed_atom_ids=[],
    )
    heatmap, atom_maps = build_logic_heatmap(
        atom_results,
        load_config(),
        logic_result=logic_result,
    )
    assert np.allclose(heatmap, 0.0)
    assert all(np.allclose(value, 0.0) for value in atom_maps.values())
