"""Pixel counterfactuals fail closed at region/image boundaries."""

import numpy as np
from types import SimpleNamespace

from symbolicAD.rules.facts.interventions import (
    relocate_entity_to_region,
    separate_entity_horizontally,
)
from symbolicAD.rules.facts.pipeline import plan_rule_matched_count_interventions
from symbolicAD.rules.rule_schema import RuleSpec


def _record(boxes):
    return {
        "image_size": [100, 100],
        "detections": [
            {"entity_type": "part", "box_xyxy": list(box)} for box in boxes
        ],
    }


def test_horizontal_separation_skips_patch_larger_than_region():
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    record = _record(((0, 0, 100, 100), (25, 25, 35, 35)))
    result = separate_entity_horizontally(
        image, record, "part", (20, 20, 50, 50),
    )
    assert result is None


def test_relocation_skips_patch_larger_than_target_region():
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    record = _record(((5, 5, 45, 45),))
    result = relocate_entity_to_region(
        image, record, "part", (0, 0, 50, 50), (70, 70, 90, 90),
    )
    assert result is None


def test_rule_matched_count_edit_records_target_but_does_not_flip_truth():
    class Builder:
        @staticmethod
        def build_from_record(record):
            return SimpleNamespace(metadata={
                "container_box": (0.0, 0.0, 100.0, 100.0),
                "divider_x": 50.0,
            })

    image = np.full((100, 100, 3), 220, dtype=np.uint8)
    image[20:30, 20:30] = 50
    record = _record(((20, 20, 30, 30),))
    count = RuleSpec(
        "count_part", "count_eq", ("part",), {"expected": 1},
    )
    relation = RuleSpec(
        "align_part", "aligned_vertical", ("part", "part"),
        {"threshold": 0.1},
    )

    plan = plan_rule_matched_count_interventions(
        Builder(), "support.png", record, (count, relation),
    )

    assert [name for name, _ in plan] == ["erase_for_count_part_part"]
    edited = plan[0][1](image)
    assert edited is not None
    assert edited.edit.kind == "erase_entity"
    assert edited.edit.extra["target_rule_id"] == "count_part"
    assert edited.edit.extra["target_predicate"] == "count_eq"
    assert "status" not in edited.edit.extra
    assert "violation" not in edited.edit.extra
