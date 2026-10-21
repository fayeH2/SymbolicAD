import numpy as np

from neurosymbolic_fsad.grounding.instance_grounder import InstanceGrounder
from neurosymbolic_fsad.perception.component_instances import ComponentInstance
from neurosymbolic_fsad.perception.scene_graph import SceneGraph
from neurosymbolic_fsad.reasoning.constraint_executor import (
    ConstraintExecutor, ConstraintOperator, combine_statuses,
)
from neurosymbolic_fsad.reasoning.predicate_executor import ExecutionContext
from neurosymbolic_fsad.reasoning.violation_result import RuleStatus
from neurosymbolic_fsad.rules.rule_compiler import (
    CompileDisposition, RuleCompiler, SupportRuleInducer,
)
from neurosymbolic_fsad.rules.rule_optimizer import GreedyRuleSetOptimizer
from neurosymbolic_fsad.rules.rule_schema import RuleSpec


def _mask(index, count, shape=(64, 64)):
    result = np.zeros(shape, dtype=bool)
    top = 3 + index * max(3, (shape[0] - 8) // max(count, 1))
    result[top:top + 2, 8:34] = True
    return result


def _context(rows, *, map_size=64, confidence=1.0):
    instances = []
    for index, attributes in enumerate(rows):
        instances.append(ComponentInstance.from_mask(
            f"wire_{index}", "wire", _mask(index, len(rows)),
            confidence=confidence, attributes=dict(attributes), source="test",
        ))
    scene = SceneGraph(
        components=instances,
        component_types={index: "wire" for index in range(len(instances))},
        grounding_confidence=confidence,
    )
    return ExecutionContext(scene, map_size=map_size)


def _length_only_support():
    counts = [1, 2, 3, 4]
    contexts = []
    for scene_index, count in enumerate(counts):
        contexts.append(_context([
            {
                "length": 0.4 + 0.002 * ((index + scene_index) % 3 - 1),
                "aspect_ratio": [0.25, 0.65, 1.2, 3.4][(index + scene_index) % 4],
                "color_h": [0.05, 0.25, 0.55, 0.85][(index + 2 * scene_index) % 4],
            }
            for index in range(count)
        ]))
    return contexts


def test_length_only_support_induces_length_without_proxy_mapping():
    support = _length_only_support()
    induced = SupportRuleInducer().induce(support)
    assert induced.structure_signature == ("length_range",)
    decisions = {row.predicate: row for row in induced.decisions}
    assert decisions["length_range"].reason_code == "STABLE_ON_SUPPORT"
    assert decisions["count_eq"].reason_code == "UNSTABLE_ON_SUPPORT"
    assert decisions["color_range"].reason_code == "UNSTABLE_ON_SUPPORT"
    assert decisions["aspect_range"].reason_code == "UNSTABLE_ON_SUPPORT"

    compiled = RuleCompiler().compile(induced.proposed_rules, support)
    assert compiled.substitutions == ()
    assert [row.disposition for row in compiled.records] == [
        CompileDisposition.EXECUTABLE
    ]
    assert compiled.executable_rules[0].predicate == "length_range"

    query = _context([{
        "length": 0.20, "aspect_ratio": 0.65, "color_h": 0.25,
    }])
    result = ConstraintExecutor().execute(compiled, query)
    assert result.status == RuleStatus.VIOLATED
    assert result.violated_rule_ids == ("auto_wire_length_range",)
    assert result.rule_results[0].witness_instances == ["wire_0"]
    assert result.anomaly_map.max() > 0


def test_same_pipeline_learns_different_rule_structure_for_count_only_category():
    support = [
        _context([
            {"length": 0.2 + 0.1 * scene, "aspect_ratio": 0.5 + scene,
             "color_h": 0.1 + 0.2 * scene},
            {"length": 0.7 - 0.1 * scene, "aspect_ratio": 3.5 - scene,
             "color_h": 0.9 - 0.2 * scene},
        ])
        for scene in range(4)
    ]
    induced = SupportRuleInducer().induce(support)
    assert induced.structure_signature == ("count_eq",)
    assert induced.structure_signature != SupportRuleInducer().induce(
        _length_only_support()
    ).structure_signature


def test_unsupported_predicate_remains_visible_and_is_not_substituted():
    rule = RuleSpec(
        "curvature", "curvature_profile", ("wire",),
        parameters={"lower": 0.0, "upper": 1.0},
    )
    compiled = RuleCompiler().compile([rule], support_contexts=[])
    record = compiled.records[0]
    assert record.disposition == CompileDisposition.UNSUPPORTED
    assert record.reason_code == "NO_REGISTERED_EXECUTOR"
    assert compiled.executable_rules == ()
    assert compiled.substitutions == ()


def test_missing_attribute_and_low_identity_are_unknown_not_satisfied():
    rule = RuleSpec(
        "length", "length_range", ("wire",),
        parameters={"lower": 0.35, "upper": 0.45},
    )
    compiled = RuleCompiler().compile([rule], [], validate_support=False)
    no_length = _context([{"aspect_ratio": 1.0}], confidence=1.0)
    no_length.scene.components[0].attributes.pop("length")
    result = ConstraintExecutor().execute(compiled, no_length)
    assert result.status == RuleStatus.UNKNOWN
    assert result.violation_score == 0.0
    assert not result.anomaly_map.any()
    assert result.satisfied_rule_ids == ()

    low_identity = _context([{"length": 0.2}], confidence=0.2)
    result = ConstraintExecutor().execute(compiled, low_identity)
    assert result.status == RuleStatus.UNKNOWN
    assert result.violation_score == 0.0


def test_three_valued_normality_operators():
    S, V, U = RuleStatus.SATISFIED, RuleStatus.VIOLATED, RuleStatus.UNKNOWN
    assert combine_statuses([S, S], ConstraintOperator.AND) == S
    assert combine_statuses([S, U], ConstraintOperator.AND) == U
    assert combine_statuses([V, U], ConstraintOperator.AND) == V
    assert combine_statuses([V, V], ConstraintOperator.OR) == V
    assert combine_statuses([V, U], ConstraintOperator.OR) == U
    assert combine_statuses([S, U], ConstraintOperator.OR) == S


def test_three_grounding_levels_share_executor_and_keep_gt_out_of_auto_inference():
    normal = np.zeros((64, 64), dtype=np.int32)
    normal[30:34, 16:48] = 1
    short = np.zeros((64, 64), dtype=np.int32)
    short[30:34, 16:32] = 1
    expected_rule = RuleSpec(
        "manual_length", "length_range", ("wire",),
        parameters={
            "lower": 0.45, "upper": 0.55,
            "reference_endpoint_a": [16 / 63, 31.5 / 63],
            "reference_endpoint_b": [47 / 63, 31.5 / 63],
        },
        source="manual", grounding_strategy="instance_mask_endpoint_difference",
    )
    compiler = RuleCompiler()
    compiled_manual = compiler.compile([expected_rule], [], validate_support=False)
    grounder = InstanceGrounder()

    manual = grounder.from_manual_mask(short, entity_type="wire", diagnostic_only=True)
    manual_result = ConstraintExecutor().execute(
        compiled_manual, ExecutionContext(manual.scene, map_size=64)
    )
    assert manual.uses_ground_truth_during_inference is True
    assert manual_result.status == RuleStatus.VIOLATED
    assert manual_result.anomaly_map[:, 32:49].max() > 0

    automatic = grounder.from_binary_observation(
        short > 0, entity_type="wire", confidence=0.9
    )
    automatic_result = ConstraintExecutor().execute(
        compiled_manual, ExecutionContext(automatic.scene, map_size=64)
    )
    assert automatic.uses_ground_truth_during_inference is False
    assert automatic_result.status == RuleStatus.VIOLATED

    support_contexts = []
    for _ in range(4):
        grounded = grounder.from_binary_observation(
            normal > 0, entity_type="wire", confidence=0.9
        )
        support_contexts.append(ExecutionContext(grounded.scene, map_size=64))
    induced = SupportRuleInducer().induce(support_contexts)
    # This fixture also has stable count/aspect/color-absence; select the
    # automatically induced length rule to isolate rule-induction grounding.
    auto_length = [
        rule for rule in induced.proposed_rules if rule.predicate == "length_range"
    ]
    compiled_auto = compiler.compile(auto_length, support_contexts)
    auto_result = ConstraintExecutor().execute(
        compiled_auto, ExecutionContext(automatic.scene, map_size=64)
    )
    assert auto_result.status == RuleStatus.VIOLATED
    assert auto_result.anomaly_map.max() > 0


def test_greedy_optimizer_closes_dependencies_and_deduplicates_responses():
    rules = [
        RuleSpec("length", "length_range", ("wire",),
                 {"lower": 0.3, "upper": 0.5}, family="shape"),
        RuleSpec("length_alias", "length_range", ("wire",),
                 {"lower": 0.3, "upper": 0.5}, family="shape"),
        RuleSpec("aspect", "aspect_range", ("wire",),
                 {"lower": 0.2, "upper": 4.0}, family="alignment"),
        RuleSpec("dependent", "color_range", ("wire",),
                 {"lower": 0.0, "upper": 1.0}, family="spacing",
                 dependencies=("aspect",)),
    ]
    compiled = RuleCompiler().compile(rules, [], validate_support=False)
    report = GreedyRuleSetOptimizer().select(
        compiled, budget=3,
        diagnostic_responses={
            "length": [1, 1, 0, 0, 0],
            "length_alias": [1, 1, 0, 0, 0],
            "aspect": [0, 0, 1, 0, 0],
            "dependent": [0, 0, 0, 1, 1],
        },
    )
    assert set(report.selected_ids) == {"length", "aspect", "dependent"}
    assert "length_alias" not in report.selected_ids
    assert report.diagnostic_intervention_coverage == 1.0
    assert report.real_anomaly_coverage is None
