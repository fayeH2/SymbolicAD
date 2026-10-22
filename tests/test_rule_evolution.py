"""Regression tests for scoped counting and the two-objective search contract."""

import numpy as np

from symbolicAD.perception.component_instances import ComponentInstance
from symbolicAD.perception.scene_graph import SceneGraph
from symbolicAD.reasoning.predicate_executor import ExecutionContext
from symbolicAD.reasoning.violation_result import RuleStatus
from symbolicAD.rules.rule_registry import default_rule_registry
from symbolicAD.rules.rule_schema import RuleSpec
from symbolicAD.rules.search.symbolicdet_evolution import (
    CounterfactualPair,
    EvolutionaryRuleSearch,
    SearchConfig,
    contains_redundant_rules,
    exhaustive_rule_search,
)


def _rectangle(shape, x0, y0, x1, y1):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _breakfast_context(*, fruit_on_left=True, calibrated=True):
    shape = (32, 32)
    positions = [(5, 8, 9, 12), (5, 18, 9, 22), (10, 13, 14, 17)]
    if not fruit_on_left:
        positions = [(23, 8, 27, 12), (23, 18, 27, 22), (18, 13, 22, 17)]
    kinds = ("orange", "orange", "peach")
    components = []
    component_types = {}
    for index, (kind, box) in enumerate(zip(kinds, positions)):
        component_types[len(components)] = kind
        components.append(ComponentInstance.from_mask(
            f"{kind}_{index}", kind, _rectangle(shape, *box),
            confidence=0.95, source="test_detector",
        ))
    for kind, box in (
        ("left_compartment", (0, 0, 16, 32)),
        ("right_compartment", (16, 0, 32, 32)),
    ):
        component_types[len(components)] = kind
        components.append(ComponentInstance.from_mask(
            f"{kind}_0", kind, _rectangle(shape, *box),
            confidence=1.0, source="fixed_split",
        ))
    scene = SceneGraph(
        components=components,
        component_types=component_types,
        type_hierarchy={"fruit": ("peach", "orange")},
        grounding_confidence=0.95,
    )
    confidence = 0.95 if calibrated else 0.0
    complete = calibrated
    return ExecutionContext(scene, map_size=32, metadata={
        "count_observation_complete": {
            "fruit": complete, "peach": complete, "orange": complete,
        },
        "count_observation_confidence": {
            "fruit": confidence, "peach": confidence, "orange": confidence,
        },
    })


def _rule(rule_id, entity, expected, predicate="count_in"):
    entities = (entity, "left_compartment") if predicate == "count_in" else (entity,)
    return RuleSpec(
        rule_id, predicate, entities,
        parameters={"expected": expected, "tolerance": 0},
        family="cardinality", source="qwen_proposal",
    )


def test_count_in_uses_explicit_hierarchy_and_localizes_misplaced_instances():
    executor = default_rule_registry().resolve("count_in").executor
    rule = _rule("fruit_left", "fruit", 3)
    normal = executor.execute(rule, _breakfast_context())
    assert normal.status == RuleStatus.SATISFIED
    assert normal.details["actual"] == 3

    swapped = executor.execute(rule, _breakfast_context(fruit_on_left=False))
    assert swapped.status == RuleStatus.VIOLATED
    assert swapped.violation_score == 1.0
    assert len(swapped.witness_instances) == 3
    assert swapped.anomaly_map[:, 16:].max() == 1.0
    assert swapped.anomaly_map[:, :16].max() == 0.0


def test_count_in_fails_closed_without_entity_specific_calibration():
    executor = default_rule_registry().resolve("count_in").executor
    result = executor.execute(
        _rule("peach_left", "peach", 1),
        _breakfast_context(calibrated=False),
    )
    assert result.status == RuleStatus.UNKNOWN
    assert result.violation_score == 0.0
    assert not result.anomaly_map.any()


def test_pareto_evolution_matches_small_exhaustive_oracle():
    candidates = (
        _rule("fruit_left", "fruit", 3),
        _rule("peach_left", "peach", 1),
        _rule("orange_left", "orange", 2),
        # It remains in the proposal ledger but is hard-filtered: no executor.
        RuleSpec("orange_aligned", "aligned_vertical", ("orange",), {}),
    )
    normals = tuple(_breakfast_context() for _ in range(6))
    counterfactuals = tuple(
        CounterfactualPair(
            f"mirror_{index}", _breakfast_context(),
            _breakfast_context(fruit_on_left=False), "horizontal_mirror",
            validated=True,
        )
        for index in range(4)
    )
    config = SearchConfig(
        population_size=16, generations=12, tolerance_grid=(0, 1),
        minimum_known_fraction=1.0, counterfactual_margin=0.5, seed=7,
    )
    evolved = EvolutionaryRuleSearch(config).search(
        candidates, normals, counterfactuals
    )
    oracle = exhaustive_rule_search(
        candidates, normals, counterfactuals, config=config
    )
    assert evolved.selected.objectives == oracle.selected.objectives == (0.0, 0.0)
    assert all(rule.predicate == "count_in" for rule in evolved.selected.rules)
    assert all(rule.parameters["tolerance"] == 0 for rule in evolved.selected.rules)


def test_redundancy_check_is_exact_not_shared_object_heuristic():
    peach = _rule("peach_left", "peach", 1)
    orange = _rule("orange_left", "orange", 2)
    alias = _rule("peach_left_alias", "peach", 1)
    assert not contains_redundant_rules((peach, orange))
    assert contains_redundant_rules((peach, alias))
