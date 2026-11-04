import numpy as np
from PIL import Image

from symbolicAD.rules.qwen_rule_generator import (
    _extract_json,
    _support_image_signature,
    build_rule_kb_from_description,
)


def test_qwen_builder_repairs_placeholders_and_duplicates():
    description = {
        "category": "demo",
        "components": [
            {
                "name": "Bottle",
                "role": "container",
                "part_of": "Parent",
                "position": "center",
                "prompts": ["a bottle"],
                "count": 1,
            },
            {
                "name": "FruitLabel",
                "role": "label",
                "part_of": "Bottle",
                "position": "upper center",
                "prompts": ["a fruit label"],
                "count": 1,
            },
            {
                "name": "TextLabel",
                "role": "label",
                "part_of": "Bottle",
                "position": "lower center",
                "prompts": ["a text label"],
                "count": 1,
            },
        ],
        "relations": [
            {"type": "Inside", "a": "Bottle", "b": "FruitLabel"},
            {"type": "Above", "a": "FruitLabel", "b": "TextLabel"},
        ],
        "attribute_variants": [
            {
                "group": "fruit",
                "conditions": [
                    {"component": "FruitLabel", "attribute": "fruit", "value": "orange"},
                    {"component": "FruitLabel", "attribute": "color", "value": "yellow"},
                ],
            },
            {
                "group": "fruit",
                "conditions": [
                    {"component": "FruitLabel", "attribute": "fruit", "value": "orange"},
                    {"component": "FruitLabel", "attribute": "color", "value": "yellow"},
                ],
            },
        ],
    }
    rule_kb = build_rule_kb_from_description(description, "demo")
    names = [component.name for component in rule_kb.components]
    assert "Parent" not in names
    assert names == ["Bottle", "FruitLabel", "TextLabel"]
    expressions = [atom.expression for atom in rule_kb.atoms]
    assert "Inside(Bottle,FruitLabel)" not in expressions
    assert "Above(FruitLabel,TextLabel)" in expressions
    assert "PositionMatch(FruitLabel,upper_center)" in expressions
    attribute_atoms = [atom for atom in rule_kb.atoms if atom.kind == "AttributeMatch"]
    assert len(attribute_atoms) == 2  # duplicate variant removed


def test_support_image_signature_tracks_content_and_order():
    black = Image.fromarray(np.zeros((4, 8, 3), dtype=np.uint8))
    white = Image.fromarray(np.full((4, 8, 3), 255, dtype=np.uint8))
    assert _support_image_signature([black]) != _support_image_signature([white])
    assert _support_image_signature([black, white]) != _support_image_signature(
        [white, black]
    )


def test_qwen_builder_bounds_variants_and_does_not_invent_empty_parent():
    description = {
        "components": [
            {
                "name": "Bottle",
                "part_of": "",
                "position": "center",
                "prompts": ["a bottle"],
                "count": 1,
            },
            {
                "name": "Label",
                "part_of": "Empty",
                "position": "center",
                "prompts": ["a label"],
                "count": 1,
            },
        ],
        "attribute_variants": [
            {
                "group": f"variant_{index}",
                "conditions": [
                    {
                        "component": "Label",
                        "attribute": "fruit",
                        "value": value,
                    }
                ],
            }
            for index, value in enumerate(["cherry", "banana", "orange"])
        ],
    }
    rule_kb = build_rule_kb_from_description(
        description, "demo", max_variants=2
    )
    assert [item.name for item in rule_kb.components] == ["Bottle", "Label"]
    assert rule_kb.components[0].part_of == ""
    assert rule_kb.components[1].part_of == ""
    values = [
        atom.attribute_value
        for atom in rule_kb.atoms
        if atom.kind == "AttributeMatch"
    ]
    assert values == ["cherry", "banana"]


def test_qwen_json_repair_fixes_mismatched_array_closer():
    parsed = _extract_json(
        '{"components":[],"attribute_variants":[{"group":"v"}}}'
    )
    assert parsed["components"] == []
    assert parsed["attribute_variants"] == [{"group": "v"}]
