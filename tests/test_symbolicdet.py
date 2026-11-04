import numpy as np
from PIL import Image, ImageDraw

from symbolicAD.config import SymbolicDetConfig
from symbolicAD.symbolicdet import SegmentFirstSymbolicDet


def _two_component_image(*, swap=False, missing_left=False):
    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)
    if not missing_left:
        left = (220, 30, 30) if not swap else (30, 180, 40)
        draw.rectangle((15, 25, 32, 70), fill=left)
    right = (30, 180, 40) if not swap else (220, 30, 30)
    draw.rectangle((60, 25, 77, 70), fill=right)
    return image


def _clip_features(*, swap=False, missing_left=False):
    features = np.zeros((4, 12, 12), dtype=np.float32)
    features[0] = 1.0  # background prototype
    if not missing_left:
        features[:, 3:9, 2:4] = 0.0
        features[2 if swap else 1, 3:9, 2:4] = 1.0
    features[:, 3:9, 8:10] = 0.0
    features[1 if swap else 2, 3:9, 8:10] = 1.0
    return features


def _model():
    config = SymbolicDetConfig(visual_prototypes=3, expert_top_k=2)
    supports = [_two_component_image(), _two_component_image()]
    patches = np.stack([_clip_features(), _clip_features()], axis=0)
    return SegmentFirstSymbolicDet(config).fit(supports, patches)


def test_dynamic_expressions_are_support_only_and_quiet_on_support():
    model = _model()
    output = model.infer(_two_component_image(), _clip_features())
    payload = model.to_dict()
    assert payload["protocol"] == "compiled_from_normal_support_clip_features"
    assert payload["feature_source"] == "clip_projected_patch_features"
    assert payload["support_count"] == 2
    assert len(payload["expressions"]) > 10
    assert output.score == 0.0
    assert float(output.anomaly_map.max()) == 0.0


def test_colour_swap_activates_relation_expert_and_localises_components():
    model = _model()
    output = model.infer(_two_component_image(swap=True), _clip_features(swap=True))
    assert output.score > 0.8
    assert output.expert_weights["relation"] > 0.0
    assert any(item["family"] == "relation" for item in output.violations)
    object_region = output.anomaly_map[3:9, 2:10]
    background = output.anomaly_map[:2, :2]
    assert float(object_region.mean()) > float(background.mean()) + 0.2


def test_missing_component_projects_violation_to_expected_location():
    model = _model()
    output = model.infer(
        _two_component_image(missing_left=True),
        _clip_features(missing_left=True),
    )
    left_expected = output.anomaly_map[3:9, 2:4]
    corner = output.anomaly_map[:2, :2]
    assert output.score > 0.6
    assert float(np.mean(left_expected)) > float(np.mean(corner)) + 0.2


def test_static_program_round_trip_needs_no_support_or_llm(tmp_path):
    model = _model()
    path = model.save(tmp_path / "symbolicdet.npz")
    restored = SegmentFirstSymbolicDet.load(path, model.config)
    output = restored.infer(
        _two_component_image(swap=True), _clip_features(swap=True)
    )
    assert output.score > 0.8
    assert restored.to_dict()["support_count"] == 2
