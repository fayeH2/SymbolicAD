"""Regression tests for strict three-valued rule-set aggregation.

These fixtures deliberately use a tiny fake predicate so the tests exercise
the rule-set/search contracts without depending on detector behaviour.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pytest

from symbolicAD.evaluation.loco_rule_evaluation import _execute_set
from symbolicAD.perception.scene_graph import SceneGraph
from symbolicAD.reasoning.predicate_executor import ExecutionContext
from symbolicAD.reasoning.violation_result import RuleStatus, ViolationResult
from symbolicAD.rules.rule_registry import RuleCapability, RuleRegistry
from symbolicAD.rules.rule_schema import RuleSpec
from symbolicAD.rules.search.symbolicdet_evolution import (
    CounterfactualPair,
    _aggregate,
    evaluate_rule_set,
)


class _MetadataExecutor:
    """Return the rule outcome declared by an ``ExecutionContext`` fixture."""

    def execute(
        self, rule: RuleSpec, context: ExecutionContext
    ) -> ViolationResult:
        payload: Mapping[str, Any] = context.metadata["rule_outcomes"][rule.rule_id]
        result = ViolationResult(
            rule_id=rule.rule_id,
            status=payload["status"],
            violation_score=float(payload.get("score", 0.0)),
            evidence_confidence=1.0,
            anomaly_map=np.full(
                (context.map_size, context.map_size),
                float(payload.get("map_value", 0.0)),
                dtype=np.float32,
            ),
            attribution_uncertainty=0.0,
        )
        if payload.get("corrupt_map_after_validation", False):
            # Simulate a custom executor (or later mutation) bypassing the
            # ViolationResult constructor invariant.  The set evaluator must
            # still gate maps by status rather than trusting their contents.
            result.anomaly_map.fill(float(payload.get("map_value", 1.0)))
        return result


def _registry() -> RuleRegistry:
    registry = RuleRegistry()
    registry.register(RuleCapability(
        predicate="test_status",
        executor=_MetadataExecutor(),
        required_observations=(),
        grounding_strategy="test_fixture",
    ))
    return registry


def _rules() -> tuple[RuleSpec, RuleSpec]:
    # Distinct entity scopes keep the two rules logically non-redundant.
    return (
        RuleSpec("r1", "test_status", ("entity_a",)),
        RuleSpec("r2", "test_status", ("entity_b",)),
    )


def _context(
    outcomes: Mapping[str, Mapping[str, Any]], *, map_size: int = 8
) -> ExecutionContext:
    return ExecutionContext(
        scene=SceneGraph(components=[], grounding_confidence=1.0),
        map_size=map_size,
        metadata={"rule_outcomes": dict(outcomes)},
    )


def _result(
    rule_id: str, status: RuleStatus, score: float = 0.0
) -> ViolationResult:
    return ViolationResult(
        rule_id=rule_id,
        status=status,
        violation_score=score,
        evidence_confidence=1.0,
        anomaly_map=np.ones((4, 4), dtype=np.float32),
        attribution_uncertainty=0.0,
    )


def test_rule_set_aggregate_uses_strong_kleene_and() -> None:
    satisfied = _result("satisfied", RuleStatus.SATISFIED)
    unknown = _result("unknown", RuleStatus.UNKNOWN)
    violated = _result("violated", RuleStatus.VIOLATED, score=0.7)

    assert _aggregate((satisfied, unknown)) == (0.0, False)
    score, decided = _aggregate((violated, unknown))
    assert decided is True
    assert score == pytest.approx(0.7)


def test_counterfactual_margin_rejects_cross_rule_known_set_swap() -> None:
    rules = _rules()
    normal = _context({
        "r1": {"status": RuleStatus.SATISFIED},
        "r2": {"status": RuleStatus.SATISFIED},
    })
    before = _context({
        "r1": {"status": RuleStatus.VIOLATED, "score": 0.2},
        "r2": {"status": RuleStatus.UNKNOWN},
    })
    after = _context({
        "r1": {"status": RuleStatus.UNKNOWN},
        "r2": {"status": RuleStatus.VIOLATED, "score": 0.9},
    })
    pair = CounterfactualPair(
        pair_id="known_set_swap",
        normal=before,
        counterfactual=after,
        intervention="test_swap",
        validated=True,
    )

    evaluation = evaluate_rule_set(
        rules,
        (normal,),
        (pair,),
        registry=_registry(),
        minimum_known_fraction=1.0,
        counterfactual_margin=0.5,
    )
    assert evaluation is None


@pytest.mark.parametrize(
    "status", (RuleStatus.SATISFIED, RuleStatus.UNKNOWN)
)
def test_nonviolated_result_cannot_carry_anomaly_map(
    status: RuleStatus,
) -> None:
    result = ViolationResult(
        rule_id="nonviolated",
        status=status,
        violation_score=0.8,
        evidence_confidence=1.0,
        anomaly_map=np.ones((6, 6), dtype=np.float32),
        attribution_uncertainty=0.0,
    )

    assert result.violation_score == 0.0
    assert result.anomaly_map is not None
    assert not np.any(result.anomaly_map)


def test_loco_set_executor_abstains_and_blocks_unknown_map_pollution() -> None:
    rules = _rules()
    context = _context({
        "r1": {
            "status": RuleStatus.SATISFIED,
            "map_value": 1.0,
            "corrupt_map_after_validation": True,
        },
        "r2": {
            "status": RuleStatus.UNKNOWN,
            "map_value": 1.0,
            "corrupt_map_after_validation": True,
        },
    })

    score, logic_map, known_fraction, rule_maps, status = _execute_set(
        rules, context, _registry()
    )

    assert status == RuleStatus.UNKNOWN
    assert np.isnan(score)
    assert known_fraction == pytest.approx(0.5)
    assert not np.any(logic_map)
    assert all(not np.any(values) for values in rule_maps.values())


def test_zero_minimum_known_fraction_is_rejected() -> None:
    rules = _rules()
    unknown = _context({
        "r1": {"status": RuleStatus.UNKNOWN},
        "r2": {"status": RuleStatus.UNKNOWN},
    })
    pair = CounterfactualPair(
        pair_id="all_unknown",
        normal=unknown,
        counterfactual=unknown,
        intervention="none",
        validated=True,
    )

    with pytest.raises(ValueError, match="minimum_known_fraction"):
        evaluate_rule_set(
            rules,
            (unknown,),
            (pair,),
            registry=_registry(),
            minimum_known_fraction=0.0,
        )


@pytest.mark.parametrize("field", ("score", "confidence", "uncertainty", "map"))
def test_violation_result_rejects_nonfinite_outputs(field: str) -> None:
    kwargs = {
        "rule_id": "bad",
        "status": RuleStatus.VIOLATED,
        "violation_score": 0.5,
        "evidence_confidence": 0.8,
        "anomaly_map": np.zeros((2, 2), dtype=np.float32),
        "attribution_uncertainty": 0.2,
    }
    if field == "score":
        kwargs["violation_score"] = float("nan")
    elif field == "confidence":
        kwargs["evidence_confidence"] = float("inf")
    elif field == "uncertainty":
        kwargs["attribution_uncertainty"] = float("-inf")
    else:
        kwargs["anomaly_map"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ViolationResult(**kwargs)
