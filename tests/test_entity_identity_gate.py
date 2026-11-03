"""The entity gate rejects hallucinated names without semantic rewriting."""

import pytest

from symbolicAD.rules.rule_schema import RuleSpec
from symbolicAD.rules.rule_validator import audit_rule_entity_identity


def _record(*entities):
    return {
        "detections": [
            {"entity_type": name, "score": 0.9, "box_xyxy": [0, 0, 1, 1]}
            for name in entities
        ]
    }


SUPPORTS = (
    _record("peach", "orange"),
    _record("peach", "orange"),
    _record("peach", "orange"),
    _record("peach"),
)


def _audit(entity, **kwargs):
    return audit_rule_entity_identity(
        RuleSpec("rule", "count_eq", (entity,), {"expected": 1}),
        SUPPORTS,
        vocabulary_entities=("peach", "orange"),
        supertype_members={"fruit": ("peach", "orange")},
        derived_entities=("left_compartment", "right_compartment"),
        **kwargs,
    )


def test_unknown_name_is_rejected_and_never_aliased_to_observed_peach():
    result = _audit("apple")

    assert not result.accepted
    assert result.reason_codes == (
        "ENTITY_NOT_IN_PERCEPTION_ONTOLOGY:apple",
    )
    assert result.entity_evidence["apple"]["ontology_status"] == "UNKNOWN_ENTITY"
    assert "peach" not in result.entity_evidence


def test_exact_entity_requires_declared_support_presence_fraction():
    assert _audit("orange", minimum_presence_fraction=0.75).accepted
    rejected = _audit("orange", minimum_presence_fraction=1.0)

    assert not rejected.accepted
    assert rejected.entity_evidence["orange"]["support_presence_fraction"] == 0.75
    assert rejected.reason_codes == (
        "ENTITY_NOT_OBSERVED_ON_SUPPORT:orange",
    )


def test_declared_supertype_and_derived_region_are_audited_without_aliases():
    assert _audit("fruit").accepted
    assert _audit("left_compartment").accepted
    assert (
        _audit("left_compartment")
        .entity_evidence["left_compartment"]["observation_source"]
        == "deterministic_scene_geometry"
    )


def test_invalid_presence_threshold_fails_closed():
    with pytest.raises(ValueError, match="must be in"):
        _audit("orange", minimum_presence_fraction=1.1)
