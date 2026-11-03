"""Regression tests for relation executors whose bugs can fabricate witnesses."""

import numpy as np

from symbolicAD.perception.component_instances import ComponentInstance
from symbolicAD.perception.scene_graph import SceneGraph
from symbolicAD.reasoning.predicate_executor import (
    AlignedVerticalExecutor,
    ExecutionContext,
)
from symbolicAD.reasoning.violation_result import RuleStatus
from symbolicAD.rules.rule_schema import RuleSpec


def _orange(instance_id: str, x_start: int) -> ComponentInstance:
    mask = np.zeros((32, 32), dtype=bool)
    mask[6:12, x_start:x_start + 4] = True
    return ComponentInstance.from_mask(
        instance_id, "orange", mask, confidence=0.95, source="test"
    )


def _context(*instances: ComponentInstance) -> ExecutionContext:
    return ExecutionContext(
        SceneGraph(
            components=list(instances),
            component_types={index: "orange" for index in range(len(instances))},
            grounding_confidence=0.95,
            observation_complete=True,
        ),
        map_size=32,
    )


def test_aligned_vertical_self_pair_requires_two_distinct_instances():
    rule = RuleSpec(
        "orange_column", "aligned_vertical", ("orange", "orange"),
        {"threshold": 0.2},
    )
    result = AlignedVerticalExecutor().execute(rule, _context(_orange("o1", 8)))
    assert result.status == RuleStatus.UNKNOWN
    assert result.details["unknown_reason"] == "need_two_instances"
    assert result.witness_instances == []
    assert not result.anomaly_map.any()


def test_aligned_vertical_self_pair_has_unique_witnesses_on_violation():
    rule = RuleSpec(
        "orange_column", "aligned_vertical", ("orange", "orange"),
        {"threshold": 0.1},
    )
    result = AlignedVerticalExecutor().execute(
        rule, _context(_orange("o1", 2), _orange("o2", 24))
    )
    assert result.status == RuleStatus.VIOLATED
    assert result.witness_instances == ["o1", "o2"]
    assert len(result.witness_instances) == len(set(result.witness_instances))
    assert result.anomaly_map.max() > 0

