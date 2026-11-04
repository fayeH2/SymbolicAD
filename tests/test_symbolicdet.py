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


def _model():
    config = SymbolicDetConfig(analysis_size=96, color_clusters=2, expert_top_k=2)
    supports = [_two_component_image(), _two_component_image()]
    return SegmentFirstSymbolicDet(config).fit(supports)


def test_dynamic_expressions_are_support_only_and_quiet_on_support():
    model = _model()
    output = model.infer(_two_component_image())
    payload = model.to_dict()
    assert payload["protocol"] == "normal_support_only"
    assert payload["support_count"] == 2
    assert len(payload["expressions"]) > 10
    assert output.score == 0.0
    assert float(output.anomaly_map.max()) == 0.0


def test_colour_swap_activates_relation_expert_and_localises_components():
    model = _model()
    output = model.infer(_two_component_image(swap=True))
    assert output.score > 0.8
    assert output.expert_weights["relation"] > 0.0
    assert any(item["family"] == "relation" for item in output.violations)
    object_region = output.anomaly_map[24:72, 14:79]
    background = output.anomaly_map[:12, :12]
    assert float(object_region.mean()) > float(background.mean()) + 0.2


def test_missing_component_projects_violation_to_expected_location():
    model = _model()
    output = model.infer(_two_component_image(missing_left=True))
    left_expected = output.anomaly_map[25:71, 15:33]
    corner = output.anomaly_map[:12, :12]
    assert output.score > 0.6
    assert float(np.mean(left_expected)) > float(np.mean(corner)) + 0.2
