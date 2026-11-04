from types import SimpleNamespace

import numpy as np

from symbolicAD.config import load_config
from symbolicAD.grounding.relation_grounder import evaluate_atoms
from symbolicAD.grounding.scene_graph import ComponentInstance, SceneGraph
from symbolicAD.grounding.support_memory import (
    ComponentMemory,
    SupportMemory,
    load_support_memory,
    save_support_memory,
)
from symbolicAD.grounding.instance_extractor import named_color_lab
from symbolicAD.rules.compiler import compile_rule_kb
from symbolicAD.rules.schema import RuleKB, parse_expression
from symbolicAD.rules.validator import validate_against_support


def _memory(expected, std):
    return SimpleNamespace(
        expected_count=expected,
        count_std=std,
        exist_q05=0.0,
        exist_q50=1.0,
    )


def _support(**components):
    return SimpleNamespace(components=components)


def test_parse_extended_atoms():
    assert parse_expression("CountInRange(Label,1,2)") == (
        "CountInRange",
        ["Label", 1, 2],
        {},
    )
    kind, args, _ = parse_expression("AttributeMatch(FruitLabel,fruit,orange)")
    assert kind == "AttributeMatch"
    assert args == ["FruitLabel", "fruit", "orange"]


def test_compiler_dedup_implied_and_tree():
    payload = {
        "category": "demo",
        "components": [
            {"name": "Screw", "prompts": ["a screw"]},
            {"name": "Slot", "prompts": ["a slot"]},
        ],
        "atoms": [
            {"id": "R1", "expression": "Exist(Screw)"},
            {"id": "R2", "expression": "CountEq(Screw,1)"},
            {"id": "R3", "expression": "CountEq(Slot,1)"},
            {"id": "R4", "expression": "OneToOne(Screw,Slot)"},
            {"id": "R5", "expression": "Inside(Screw,Slot)"},
            {"id": "R6", "expression": "Inside(Screw, Slot)"},
        ],
        "formula": {"op": "AND", "children": ["R1", "R2", "R3", "R4", "R5", "R6"]},
    }
    config = load_config()
    support = _support(Screw=_memory(1.0, 0.0), Slot=_memory(1.0, 0.0))
    compiled, report = compile_rule_kb(RuleKB.from_dict(payload), support, config)

    expressions = {atom.expression for atom in compiled.atoms}
    assert "Inside(Screw, Slot)" not in expressions or "Inside(Screw,Slot)" not in expressions
    assert not any(atom.kind == "Exist" for atom in compiled.atoms)
    assert not any(atom.kind == "OneToOne" for atom in compiled.atoms)
    assert len(compiled.atoms) == 3
    assert compiled.formula.op == "AND"
    assert report.tree_depth >= 2
    assert len(set(report.groups)) >= 2


def test_compiler_repairs_count_to_range_and_attribute_or():
    payload = {
        "category": "demo",
        "components": [
            {"name": "Label", "prompts": ["a label"]},
            {"name": "Liquid", "prompts": ["liquid"]},
        ],
        "atoms": [
            {"id": "R1", "expression": "CountEq(Label,1)"},
            {
                "id": "R2",
                "expression": "AttributeMatch(Label,fruit,orange)",
                "params": {"group": "orange"},
            },
            {
                "id": "R3",
                "expression": "AttributeMatch(Liquid,color,yellow)",
                "params": {"group": "orange"},
            },
            {
                "id": "R4",
                "expression": "AttributeMatch(Label,fruit,banana)",
                "params": {"group": "banana"},
            },
            {
                "id": "R5",
                "expression": "AttributeMatch(Liquid,color,white)",
                "params": {"group": "banana"},
            },
        ],
        "formula": {"op": "AND", "children": ["R1", "R2", "R3", "R4", "R5"]},
    }
    config = load_config()
    config.compiler.repair_counts = True
    support = _support(Label=_memory(4.0, 0.8), Liquid=_memory(1.0, 0.0))
    compiled, report = compile_rule_kb(RuleKB.from_dict(payload), support, config)
    count_atoms = [atom for atom in compiled.atoms if atom.kind == "CountInRange"]
    assert count_atoms
    assert report.repaired_counts
    tree = compiled.formula.to_dict()
    assert "OR" in str(tree)


def _instance(name, mask, confidence=0.9):
    ys, xs = np.nonzero(mask)
    return ComponentInstance(
        instance_id=name,
        concept=name.split("_")[0],
        confidence=confidence,
        feature=np.ones(4, dtype=np.float32) / 2.0,
        mask=mask.astype(np.float32),
        bbox=(0.0, 0.0, 1.0, 1.0),
        center=(0.5, 0.5),
        area=float(mask.mean()),
    )


def test_count_in_range_atom():
    config = load_config()
    rule_kb = RuleKB.from_dict(
        {
            "category": "demo",
            "components": [{"name": "Screw", "prompts": ["a screw"]}],
            "atoms": [{"id": "R1", "expression": "CountInRange(Screw,1,2)"}],
            "formula": {"op": "AND", "children": ["R1"]},
        }
    )
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    scene = SceneGraph(
        component_maps={"Screw": np.ones((4, 4), dtype=np.float32)},
        instances={
            "Screw": [
                _instance("Screw_00", mask),
                _instance("Screw_01", mask),
            ]
        },
        patch_features=np.ones((4, 4, 4), dtype=np.float32),
    )
    results = evaluate_atoms(scene, rule_kb, None, config)
    assert results["R1"].truth == 1.0
    assert results["R1"].evidence["count"] == 2.0


def test_support_validation_can_disable_unreliable_count_grounding():
    rule_kb = RuleKB.from_dict(
        {
            "category": "demo",
            "components": [{"name": "Label", "prompts": ["a label"]}],
            "atoms": [{"id": "R1", "expression": "CountEq(Label,1)"}],
            "formula": {"op": "AND", "children": ["R1"]},
        }
    )
    support = _support(Label=_memory(2.5, 0.2))
    report = validate_against_support(
        rule_kb, support, disable_on_mismatch=True
    )
    assert report.dropped_atoms == ["R1"]
    assert rule_kb.atom("R1").weight == 0.0


def test_position_and_color_atoms_localize_failed_component():
    config = load_config()
    rule_kb = RuleKB.from_dict(
        {
            "category": "demo",
            "components": [{"name": "Label", "prompts": ["a red label"]}],
            "atoms": [
                {"id": "P", "expression": "PositionMatch(Label,upper_center)"},
                {"id": "C", "expression": "AttributeMatch(Label,color,red)"},
            ],
            "formula": {"op": "AND", "children": ["P", "C"]},
        }
    )
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    instance = _instance("Label_00", mask)
    instance.color_feature = named_color_lab("blue")
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[0:2, 1:3] = 1.0
    support = SupportMemory(
        category="demo",
        grid_size=(4, 4),
        components={"Label": ComponentMemory(name="Label", expected_mask=expected)},
    )
    scene = SceneGraph(
        component_maps={"Label": mask.astype(np.float32)},
        instances={"Label": [instance]},
        patch_features=np.ones((4, 4, 4), dtype=np.float32),
    )
    results = evaluate_atoms(scene, rule_kb, support, config)
    assert results["P"].truth < config.logic.fail_threshold
    assert results["P"].responsibility_map.sum() > 0
    assert results["C"].truth < config.logic.fail_threshold
    assert results["C"].evidence["grounding"] == "lab_color"


def test_attribute_visual_prototype_survives_static_memory_roundtrip(tmp_path):
    prototype = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    memory = SupportMemory(
        category="demo",
        grid_size=(2, 2),
        feature_dim=4,
        attribute_prototypes={"A1": prototype},
        metadata={"support_memory_version": "test"},
    )
    path = save_support_memory(memory, tmp_path / "support_memory.npz")
    restored = load_support_memory(path)
    np.testing.assert_allclose(restored.attribute_prototypes["A1"], prototype)
