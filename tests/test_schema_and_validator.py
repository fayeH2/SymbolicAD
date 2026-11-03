import numpy as np

from symbolicAD.rules.schema import (
    LogicNode,
    RelationAtom,
    RuleKB,
    parse_expression,
)
from symbolicAD.rules.validator import validate_rule_kb


def test_parse_expression_atoms():
    assert parse_expression("Exist(Screw)") == ("Exist", ["Screw"], {})
    assert parse_expression("CountEq(Screw, 4)") == ("CountEq", ["Screw", 4], {})
    kind, args, params = parse_expression("Inside(Screw, Slot)")
    assert kind == "SpatialRelation"
    assert args == ["Screw", "Slot"]
    assert params == {"relation": "Inside"}


def test_roundtrip_rule_kb(tmp_path):
    payload = {
        "category": "demo",
        "components": [
            {"name": "Screw", "prompts": ["a screw"]},
            {"name": "Slot", "prompts": ["a slot"]},
        ],
        "atoms": [
            {
                "id": "R1",
                "expression": "CountEq(Screw,4)",
                "normal_prompt": "four screws",
                "violation_prompt": "wrong number of screws",
            },
            {"id": "R2", "expression": "Inside(Screw,Slot)"},
        ],
        "formula": {"op": "AND", "children": ["R1", "R2"]},
    }
    rule_kb = RuleKB.from_dict(payload)
    path = tmp_path / "demo.json"
    rule_kb.save(path)
    loaded = RuleKB.load(path)
    assert loaded.category == "demo"
    assert isinstance(loaded.formula, LogicNode)
    assert loaded.formula.atom_ids() == ["R1", "R2"]
    assert isinstance(loaded.atoms[0], RelationAtom)


def test_validator_disables_unknown_component():
    payload = {
        "category": "demo",
        "components": [{"name": "Screw", "prompts": ["a screw"]}],
        "atoms": [
            {"id": "R1", "expression": "Exist(Screw)"},
            {"id": "R2", "expression": "Exist(Missing)"},
        ],
        "formula": {"op": "AND", "children": ["R1", "R2"]},
    }
    rule_kb, report = validate_rule_kb(RuleKB.from_dict(payload))
    assert rule_kb.atom("R2").weight == 0.0
    assert "R2" in report.dropped_atoms
    assert rule_kb.atom("R1").weight == 1.0


def test_validator_adds_default_prompts():
    payload = {
        "category": "demo",
        "components": [{"name": "Screw"}],
        "atoms": [{"id": "R1", "expression": "Exist(Screw)"}],
        "formula": {"op": "AND", "children": ["R1"]},
    }
    rule_kb, report = validate_rule_kb(RuleKB.from_dict(payload))
    assert rule_kb.components[0].all_prompts()
    assert any("prompts" in issue for issue in report.issues)
