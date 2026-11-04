from symbolicAD.rules.qwen_rule_generator import build_rule_kb_from_description


def test_qwen_builder_repairs_placeholders_and_duplicates():
    description = {
        "category": "demo",
        "components": [
            {
                "name": "Bottle",
                "role": "container",
                "part_of": "Parent",
                "prompts": ["a bottle"],
                "count": 1,
            },
            {
                "name": "FruitLabel",
                "role": "label",
                "part_of": "Bottle",
                "prompts": ["a fruit label"],
                "count": 1,
            },
            {
                "name": "TextLabel",
                "role": "label",
                "part_of": "Bottle",
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
    attribute_atoms = [atom for atom in rule_kb.atoms if atom.kind == "AttributeMatch"]
    assert len(attribute_atoms) == 2  # duplicate variant removed
