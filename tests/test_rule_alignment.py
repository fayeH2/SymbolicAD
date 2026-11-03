"""Regression tests for ontology-constrained generation and entity alignment."""

from __future__ import annotations

import numpy as np

from symbolicAD.perception.component_instances import ComponentInstance
from symbolicAD.perception.scene_graph import SceneGraph
from symbolicAD.reasoning.predicate_executor import (
    AlignedVerticalExecutor, CountEqExecutor, ExecutionContext,
)
from symbolicAD.reasoning.violation_result import RuleStatus
from symbolicAD.rules.alignment import (
    align_atomic_entities, build_targeted_detector_vocabulary,
    build_validated_rule_bank, canonicalize_rules_with_bindings,
)
from symbolicAD.rules.facts.open_vocab import Detection
from symbolicAD.rules.facts.scene_builder import select_container
from symbolicAD.rules.generation.ontology import DEFAULT_BREAKFAST_ONTOLOGY
from symbolicAD.rules.generation.ontology_prompt import (
    build_rule_generation_prompt, predicate_contracts,
)
from symbolicAD.rules.rule_registry import default_rule_registry
from symbolicAD.rules.rule_schema import RuleSpec


def _instance(entity, instance_id, x, *, confidence=0.9, shape=(64, 64)):
    mask = np.zeros(shape, dtype=bool)
    mask[8:20, x:x + 6] = True
    return ComponentInstance.from_mask(
        instance_id, entity, mask, confidence=confidence, source="test",
    )


def _context(
    *instances, calibrated=True, entities=("orange", "apple"),
    incomplete=(),
):
    types = {index: row.entity_type for index, row in enumerate(instances)}
    scene = SceneGraph(
        components=list(instances), component_types=types,
        grounding_confidence=0.9, observation_complete=True,
    )
    incomplete = set(incomplete)
    return ExecutionContext(scene, map_size=64, metadata={
        "count_observation_complete": {
            name: calibrated and name not in incomplete for name in entities
        },
        "count_observation_confidence": {
            name: (1.0 if calibrated and name not in incomplete else 0.0)
            for name in entities
        },
    })


def _detection(entity, score, box, phrase=None):
    return Detection(entity, phrase or entity, score, box)


def test_ontology_canonicalizes_aliases_but_not_unknown_names():
    ontology = DEFAULT_BREAKFAST_ONTOLOGY
    assert ontology.canonicalize("almonds").canonical_id == "almond"
    assert ontology.canonicalize("breakfast_box").canonical_id == "bento_box"
    assert ontology.canonicalize("fruit").canonical_id == "fruit"
    apple = ontology.canonicalize("apple")
    assert apple.canonical_id is None
    assert apple.status == "UNRESOLVED"
    assert ontology.canonicalize("dried_fruit").canonical_id is None


def test_prompt_is_derived_from_ontology_and_registered_predicates():
    prompt = build_rule_generation_prompt("breakfast_box", 2)
    assert '"canonical_id": "almond"' in prompt
    assert '"canonical_id": "granola"' in prompt
    registered = set(default_rule_registry().predicates())
    contracts = predicate_contracts()
    assert contracts
    assert {row["typed_predicate"] for row in contracts} <= registered
    assert '"typed_predicate": "count_eq"' in prompt
    assert '"typed_predicate": "left_of"' in prompt


def test_alignment_status_for_alias_unknown_and_ambiguous_region():
    record = {
        "detections": [
            _detection("orange", 0.9, (10, 10, 20, 20)).__dict__,
            _detection("granola", 0.8, (30, 10, 50, 30), "granola cereal").__dict__,
            _detection("granola", 0.5, (30, 35, 50, 55), "cereal muesli").__dict__,
            _detection("almond", 0.7, (40, 40, 45, 45)).__dict__,
            _detection("bento_box", 0.9, (0, 0, 60, 60), "bento").__dict__,
        ],
    }
    rules = (
        RuleSpec("r1", "count_eq", ("orange",), {"expected": 1}),
        RuleSpec("r2", "count_eq", ("apple",), {"expected": 1}),
        RuleSpec("r3", "count_eq", ("granola",), {"expected": 1}),
        RuleSpec("r4", "count_eq", ("almonds",), {"expected": 1}),
    )
    bindings = align_atomic_entities(
        rules, {"support": record}, DEFAULT_BREAKFAST_ONTOLOGY,
    )
    assert bindings["orange"].status == "ALIGNED"
    assert bindings["apple"].status == "UNRESOLVED"
    assert bindings["granola"].status == "AMBIGUOUS"
    assert bindings["almonds"].status == "ALIGNED_WITH_CANONICALIZATION"
    assert bindings["almonds"].canonical_id == "almond"

    canonical, decisions = canonicalize_rules_with_bindings(rules, bindings)
    assert [rule.rule_id for rule in canonical] == ["r1", "r4"]
    assert canonical[1].entity_types == ("almond",)
    assert canonical[1].support_evidence["original_entity_types"] == ["almonds"]
    quarantined = {
        row.rule_id: row for row in decisions if row.status == "QUARANTINED"
    }
    assert set(quarantined) == {"r2", "r3"}


def test_targeted_detector_vocabulary_uses_only_rule_entities_and_container():
    rules = (
        RuleSpec("r1", "count_eq", ("almonds",), {"expected": 1}),
        RuleSpec("r2", "count_eq", ("apple",), {"expected": 1}),
    )
    vocabulary = build_targeted_detector_vocabulary(
        rules, DEFAULT_BREAKFAST_ONTOLOGY,
    )
    assert "almond" in vocabulary
    assert "bento_box" in vocabulary
    assert "apple" not in vocabulary
    assert "peach" not in vocabulary


def test_validated_bank_accepts_only_fully_aligned_rules():
    record = {
        "detections": [
            _detection("orange", 0.9, (10, 10, 20, 20)).__dict__,
            _detection("almond", 0.7, (40, 40, 45, 45)).__dict__,
            _detection("bento_box", 0.9, (0, 0, 60, 60), "bento").__dict__,
        ],
    }
    rules = (
        RuleSpec("r1", "count_eq", ("orange",), {"expected": 1}),
        RuleSpec("r2", "count_eq", ("apple",), {"expected": 1}),
        RuleSpec("r3", "count_eq", ("almonds",), {"expected": 1}),
    )
    bank = build_validated_rule_bank(
        rules, {"support": record}, category="breakfast_box",
    )
    assert [rule.rule_id for rule in bank.accepted_rules] == ["r1", "r3"]
    assert bank.accepted_rules[1].entity_types == ("almond",)
    quarantined = {row["rule_id"] for row in bank.quarantined_rules}
    assert "r2" in quarantined


def test_count_eq_fails_closed_without_entity_calibration():
    executor = CountEqExecutor()
    uncalibrated = _context(
        _instance("orange", "o1", 4), calibrated=False,
    )
    result = executor.execute(
        RuleSpec("apple", "count_eq", ("apple",), {"expected": 1}),
        uncalibrated,
    )
    assert result.status == RuleStatus.UNKNOWN
    assert result.violation_score == 0.0

    calibrated = _context(
        _instance("orange", "o1", 4), _instance("orange", "o2", 20),
        calibrated=True, entities=("orange", "apple"),
        incomplete=("apple",),
    )
    assert executor.execute(
        RuleSpec("orange", "count_eq", ("orange",), {"expected": 2}),
        calibrated,
    ).status == RuleStatus.SATISFIED
    assert executor.execute(
        RuleSpec("apple", "count_eq", ("apple",), {"expected": 1}),
        calibrated,
    ).status == RuleStatus.UNKNOWN


def test_aligned_vertical_requires_both_entity_types():
    context = _context(
        _instance("orange", "o1", 4), _instance("orange", "o2", 20),
        calibrated=True, entities=("orange", "peach"),
    )
    result = AlignedVerticalExecutor().execute(
        RuleSpec("align", "aligned_vertical", ("orange", "peach"), {"threshold": 0.1}),
        context,
    )
    assert result.status == RuleStatus.UNKNOWN
    assert result.details["unknown_reason"] == "no_grounded_instances"
    assert result.details["missing_entity_types"] == ["peach"]


def test_select_container_rejects_tiny_detector_box():
    detections = [
        _detection("bento_box", 0.9, (648, 1178, 952, 1219), "bento"),
    ]
    box, confidence, phrase, source = select_container(
        detections,
        foreground_box=(0.0275, 0.0, 0.9719, 0.98125),
        foreground_confidence=1.0,
        image_size=(1600, 1280),
        container_entity="bento_box",
    )
    assert source == "bright_foreground_silhouette_container_mismatch"
    assert box == (44.0, 0.0, 1555.04, 1256.0)
    assert phrase is None
    assert confidence == 1.0

    accepted = select_container(
        [_detection("bento_box", 0.9, (44, 0, 1555, 1256), "bento")],
        foreground_box=(0.0275, 0.0, 0.9719, 0.98125),
        foreground_confidence=1.0,
        image_size=(1600, 1280),
        container_entity="bento_box",
    )
    assert accepted[3] == "groundingdino_bento_box"
