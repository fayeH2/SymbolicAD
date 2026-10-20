import json
from pathlib import Path

import numpy as np
from PIL import Image

from neurosymbolic_fsad.features import FEATURE_CATALOG, extract_features
from neurosymbolic_fsad.data import collect_test_samples, select_inpformer_support
from neurosymbolic_fsad.qwen_rules import extract_json_object, validate_proposal
from neurosymbolic_fsad.rule_engine import StaticRuleEngine, calibrate_rule_library
from neurosymbolic_fsad.reliable_rules import (
    RELIABLE_FEATURE_CATALOG, ReliableRuleEngine, mine_reliable_rules,
)
from neurosymbolic_fsad.qwen_relations import validate_relational_proposal
from neurosymbolic_fsad.cable_relations import (
    CableRelationalRuleEngine, CableSceneGrounder, _copy_disk, _prototype_to_detected,
    _rule_fact,
    compile_relational_rules,
)
from neurosymbolic_fsad.qwen_generic_relations import (
    consensus_generic_relational_proposals,
    validate_generic_relational_proposal,
)
from neurosymbolic_fsad.generic_relations import GenericRelationalRuleEngine
from scripts.validate_generic_relational_suite import _bootstrap
from neurosymbolic_fsad.class_specific_rules import (
    PCB1Grounder, _template_distance, build_class_specific_library,
)
from neurosymbolic_fsad.training_utils import augmentation, global_cosine_hm_adaptive


def test_isolated_training_utils_need_no_upstream_evaluation_dependencies():
    torch = __import__("torch")
    image = torch.rand(3, 32, 32)
    views = augmentation(image)
    assert views.shape == (22, 3, 32, 32)
    source = [torch.rand(2, 4, 8, requires_grad=False)]
    target = [torch.rand(2, 4, 8, requires_grad=True)]
    loss = global_cosine_hm_adaptive(source, target)
    loss.backward()
    assert target[0].grad is not None
    assert torch.isfinite(target[0].grad).all()


def test_features_are_complete_and_finite():
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    image[24:72, 40:88] = (190, 80, 40)
    score_map = np.full((64, 64), 0.02, dtype=np.float32)
    score_map[28:36, 28:36] = 0.08
    facts = extract_features(image, score_map)
    assert set(facts) == set(FEATURE_CATALOG)
    assert all(np.isfinite(value) for value in facts.values())


def test_rule_engine_scores_out_of_envelope_higher():
    supports = [
        {"map_top1_mean": 0.10, "foreground_cx": 0.49},
        {"map_top1_mean": 0.11, "foreground_cx": 0.51},
        {"map_top1_mean": 0.09, "foreground_cx": 0.50},
        {"map_top1_mean": 0.10, "foreground_cx": 0.50},
    ]
    proposals = [
        {"feature": "map_top1_mean", "direction": "low", "importance": 1, "confidence": 1},
        {"feature": "foreground_cx", "direction": "stable", "importance": 1, "confidence": 1},
    ]
    library = calibrate_rule_library(
        proposals, supports, class_name="toy", dataset="synthetic",
        support_paths=[str(index) for index in range(4)],
    )
    engine = StaticRuleEngine(library)
    normal_score, _ = engine.score({"map_top1_mean": 0.10, "foreground_cx": 0.50})
    anomaly_score, evidence = engine.score({"map_top1_mean": 0.50, "foreground_cx": 0.90})
    assert normal_score == 0.0
    assert anomaly_score > 0.5
    assert evidence[0]["violation"] > 0


def test_catalog_tolerance_is_a_floor_not_standard_error():
    supports = [{"foreground_cx": value} for value in (0.49, 0.50, 0.50, 0.51)]
    library = calibrate_rule_library(
        [{"feature": "foreground_cx", "direction": "stable"}], supports,
        class_name="toy", dataset="synthetic", support_paths=["0", "1", "2", "3"],
    )
    rule = library["rules"][0]
    assert rule["soft_scale"] >= FEATURE_CATALOG["foreground_cx"]["tolerance"]
    assert rule["lower"] <= min(row["foreground_cx"] for row in supports) - 0.06
    assert rule["upper"] >= max(row["foreground_cx"] for row in supports) + 0.06


def test_empirical_retriever_can_reject_and_select_without_type_quota():
    def row(cx, nuisance=0.0):
        payload = {name: nuisance for name in RELIABLE_FEATURE_CATALOG}
        payload["foreground_cx"] = cx
        return payload

    normal_rows, pseudo_rows, normal_groups, pseudo_groups = [], [], [], []
    for group in range(4):
        for delta in (-0.01, 0.0, 0.01):
            normal_rows.append(row(0.5 + delta))
            normal_groups.append(group)
        for delta in (0.0, 0.02, 0.04):
            pseudo_rows.append(row(0.82 + delta))
            pseudo_groups.append(group)
    rules, audit = mine_reliable_rules(
        normal_rows, pseudo_rows,
        np.zeros(len(normal_rows)), np.zeros(len(pseudo_rows)),
        normal_groups, pseudo_groups, max_rules=3,
    )
    assert any(rule["feature"] == "foreground_cx" for rule in rules)
    assert all(rule["feature"] not in {"map_mean", "map_p95", "map_top1_mean"} for rule in rules)
    assert any(item.get("rejection", "").startswith("absolute map magnitude") for item in audit)


def test_v2_abstention_returns_exact_raw_baseline():
    library = {
        "schema_version": "2.0",
        "shape_context": {},
        "rules": [],
        "fusion": {
            "enabled": False,
            "baseline_calibration": {"center": 0.1, "scale": 1e-6},
            "baseline_weight": 1.0,
            "rule_weights": [],
            "bias": 0.0,
        },
    }
    engine = ReliableRuleEngine(library)
    decision, rule_score, evidence = engine.decision_score(123.456, {})
    assert decision == 123.456
    assert rule_score == 0.0
    assert evidence == []


def test_qwen_json_parser_and_whitelist():
    text = "<think>omitted</think>\n```json\n" + json.dumps({
        "normality_summary": "centered object",
        "rules": [
            {"feature": "foreground_cx", "direction": "stable", "importance": 0.9,
             "confidence": 0.8, "reason": "centered"},
            {"feature": "map_top1_mean", "direction": "low", "importance": 0.8,
             "confidence": 0.9, "reason": "quiet normal map"},
            {"feature": "edge_strength", "direction": "stable", "importance": 0.5,
             "confidence": 0.6, "reason": "stable outline"},
            {"feature": "map_hot_component", "direction": "low", "importance": 0.7,
             "confidence": 0.8, "reason": "compact normal response"},
            {"feature": "__import__('os').system('bad')", "direction": "low"},
        ],
    }) + "\n```"
    parsed = validate_proposal(extract_json_object(text))
    assert len(parsed["rules"]) == 4
    assert all(rule["feature"] in FEATURE_CATALOG for rule in parsed["rules"])


def test_official_mvtec_support_matches_checkpoint_log():
    root = "/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTec"
    if not __import__("pathlib").Path(root).is_dir():
        return
    paths = select_inpformer_support(root, "mvtec", "screw", 4, seed=1)
    assert [path.name for path in paths] == ["283.png", "119.png", "176.png", "118.png"]


def test_nested_test_layout_and_ground_truth_directory_are_handled(tmp_path):
    class_root = tmp_path / "toy"
    paths = [
        class_root / "test_public" / "good" / "normal.png",
        class_root / "test_public" / "logical_anomalies" / "Relation" / "bad.png",
        class_root / "test_public" / "ground_truth" / "bad" / "bad_mask.png",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    samples = collect_test_samples(class_root, test_subdir="test_public")
    assert [row["label"] for row in samples] == [0, 1]
    assert [row["defect_type"] for row in samples] == [
        "good", "logical_anomalies/Relation",
    ]
    assert all("ground_truth" not in str(row["path"]) for row in samples)


def test_relational_proposal_rejects_code_and_canonicalizes_colors():
    payload = {
        "normality_summary": "three colored conductors",
        "entities": [
            {"id": "conductor", "expected_count": 3, "visual_description": "round"},
            {"id": "__import__('os')", "expected_count": 1},
        ],
        "rules": [
            {"id": "count", "predicate": "count_eq", "subject": "conductor",
             "target": 3},
            {"id": "colors", "predicate": "attribute_multiset_eq",
             "subject": "conductor", "attribute": "color_identity",
             "values": ["yellow-green", "blue", "brown"]},
            {"id": "bad", "predicate": "python_eval", "subject": "conductor",
             "target": "__import__('os').system('bad')"},
        ],
    }
    parsed = validate_relational_proposal(payload)
    assert [entity["id"] for entity in parsed["entities"]] == ["conductor"]
    assert [rule["predicate"] for rule in parsed["rules"]] == [
        "count_eq", "attribute_multiset_eq",
    ]
    assert parsed["rules"][1]["values"] == ["blue", "brown", "yellow_green"]


def test_relational_compiler_uses_counterfactual_cv_and_deduplicates_fact():
    proposal = {
        "rules": [
            {"id": "distinct", "predicate": "distinct_attribute_count_eq",
             "subject": "conductor", "attribute": "color_identity", "target": 3,
             "importance": 1.0, "confidence": 1.0},
            {"id": "multiset", "predicate": "attribute_multiset_eq",
             "subject": "conductor", "attribute": "color_identity",
             "values": ["blue", "brown", "yellow_green"],
             "importance": 0.9, "confidence": 0.9},
        ],
    }
    normal_rows, pseudo_rows, normal_groups, pseudo_groups = [], [], [], []
    for group in range(4):
        normal_rows.extend([{"color_duplicate_count": 0.0}] * 3)
        normal_groups.extend([group] * 3)
        pseudo_rows.extend([{"color_duplicate_count": 1.0}] * 3)
        pseudo_groups.extend([group] * 3)
    rules, audit = compile_relational_rules(
        proposal, normal_rows, pseudo_rows, ["identity"] * len(pseudo_rows),
        normal_groups, pseudo_groups, allowed_families=("identity",),
    )
    assert len(rules) == 1
    assert rules[0]["fact"] == "color_duplicate_count"
    assert rules[0]["weight"] == 1.0
    assert sum(item["accepted"] for item in audit) == 2


def test_relational_runtime_abstains_to_exact_raw_baseline():
    library = {
        "schema_version": "3.0-relational",
        "grounder": {
            "kind": "cable_circle_chromaticity_v1",
            "image_size": 256,
            "slots": [[0.5, 0.32], [0.34, 0.6], [0.66, 0.6]],
            "slot_radii": [0.16, 0.16, 0.16],
            "role_color_prototypes": [[0.4, 0.4, 0.2], [0.2, 0.3, 0.5], [0.4, 0.3, 0.3]],
            "role_color_exemplars": [
                [[0.4, 0.4, 0.2]], [[0.2, 0.3, 0.5]], [[0.4, 0.3, 0.3]],
            ],
            "semantic_roles": {"yellow_green": 0, "blue": 1, "brown": 2},
            "core_reference": 0.5,
        },
        "rules": [],
        "fusion": {"enabled": False, "alpha": 100.0},
    }
    engine = CableRelationalRuleEngine(library)
    assert engine.decision_score(123.456, 1.0) == 123.456


def test_relation_assignment_is_inverted_for_identity_positions():
    # Detected rows 0,1,2 were assigned to prototypes 2,0,1.  Therefore
    # prototype rows 0,1,2 live at detected rows 1,2,0 respectively.
    assert _prototype_to_detected((2, 0, 1)).tolist() == [1, 2, 0]


def test_counterfactual_swap_reads_both_sources_from_original():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[12:29, 12:29] = (255, 0, 0)
    image[36:53, 36:53] = (0, 0, 255)
    first = np.asarray((20, 20, 8), dtype=float)
    second = np.asarray((44, 44, 8), dtype=float)
    swapped = _copy_disk(image, first, second, source_image=image)
    swapped = _copy_disk(swapped, second, first, source_image=image)
    assert swapped[20, 20, 2] > swapped[20, 20, 0]
    assert swapped[44, 44, 0] > swapped[44, 44, 2]


def test_relational_dsl_rejects_wrong_hard_coded_arguments():
    invalid = [
        {"predicate": "count_eq", "subject": "conductor", "target": 1},
        {"predicate": "all_inside", "subject": "conductor", "object": "copper_core"},
        {"predicate": "one_to_one_inside", "subject": "conductor", "object": "copper_core"},
        {"predicate": "distinct_attribute_count_eq", "subject": "conductor",
         "attribute": "color_identity", "target": 2},
        {"predicate": "attribute_multiset_eq", "subject": "conductor",
         "attribute": "color_identity", "values": ["blue", "brown"]},
    ]
    assert all(_rule_fact(rule)[0] is None for rule in invalid)
    fact, family = _rule_fact({
        "predicate": "triangular_layout", "subject": "conductor",
    })
    assert (fact, family) == ("triangle_degeneracy", "geometry")
    left_fact, _ = _rule_fact({
        "predicate": "relative_position", "subject": "blue", "object": "brown",
        "relation": "left_of",
    })
    inverse_fact, _ = _rule_fact({
        "predicate": "relative_position", "subject": "brown", "object": "blue",
        "relation": "right_of",
    })
    assert left_fact == inverse_fact


def test_crossfit_rows_override_optimistic_full_grounder_rows():
    proposal = {"rules": [{
        "id": "identity", "predicate": "distinct_attribute_count_eq",
        "subject": "conductor", "object": None, "attribute": "color_identity",
        "relation": None, "target": 3, "values": [],
        "importance": 1.0, "confidence": 1.0,
    }]}
    normal = [{"color_duplicate_count": 0.0}] * 8
    pseudo = [{"color_duplicate_count": 1.0}] * 8
    folds = [{
        "train_normal_rows": [{"color_duplicate_count": 0.0}] * 3,
        # A held representation failure must be exposed by strict cross-fit.
        "held_normal_rows": [{"color_duplicate_count": 1.0}] * 2,
        "held_pseudo_rows": [{"color_duplicate_count": 0.0}] * 2,
        "held_pseudo_families": ["identity", "identity"],
    }]
    rules, audit = compile_relational_rules(
        proposal, normal, pseudo, ["identity"] * 8,
        [0] * 4 + [1] * 4, [0] * 4 + [1] * 4,
        crossfit_folds=folds,
    )
    assert rules == []
    assert audit[0]["cv_protocol"] == "leave_one_support_out_refit_grounder"
    assert audit[0]["accepted"] is False


def test_outer_anchor_can_abstain_without_using_conductor_count():
    state = {
        "kind": "cable_circle_chromaticity_v1",
        "image_size": 64,
        "slots": [[0.5, 0.32], [0.34, 0.6], [0.66, 0.6]],
        "slot_radii": [0.16, 0.16, 0.16],
        "role_color_prototypes": [[0.4, 0.4, 0.2], [0.2, 0.3, 0.5], [0.4, 0.3, 0.3]],
        "role_color_exemplars": [
            [[0.4, 0.4, 0.2]], [[0.2, 0.3, 0.5]], [[0.4, 0.3, 0.3]],
        ],
        "semantic_roles": {"yellow_green": 0, "blue": 1, "brown": 2},
        "core_reference": 0.5,
        "hue_cost_weight": 0.5,
        "anchor_strength_floor": 0.05,
    }
    facts, _ = CableSceneGrounder(state).extract(np.zeros((64, 64, 3), dtype=np.uint8))
    assert facts["anchor_valid"] == 0.0
    assert facts["conductor_count_error"] > 0.0


def test_generic_proposal_whitelists_relations_and_rejects_texture_entities():
    valid = validate_generic_relational_proposal({
        "normality_summary": "four pins in a row",
        "primary_entity": {
            "id": "connector_pin", "expected_count": 4,
            "arrangement": "row", "visual_description": "four metal pins",
        },
        "rules": [
            {"id": "spacing", "predicate": "ordered_spacing"},
            {"id": "bad", "predicate": "python_eval"},
        ],
    })
    assert valid["primary_entity"]["expected_count"] == 4
    assert [row["predicate"] for row in valid["rules"]] == ["ordered_spacing"]
    assert valid["rules"][0]["family"] == "spatial_configuration"
    rejected = validate_generic_relational_proposal({
        "primary_entity": {
            "id": "mesh_cell", "expected_count": 4,
            "arrangement": "grid", "visual_description": "repeating texture cells",
        },
        "rules": [{"predicate": "count_eq"}],
    })
    assert rejected["primary_entity"] is None
    assert rejected["rules"] == []


def test_generic_consensus_does_not_force_count_or_duplicate_language_rules():
    proposals = []
    for index in range(4):
        proposals.append({
            "normality_summary": "three visible parts",
            "primary_entity": {
                "id": f"synonym_{index}", "expected_count": 3,
                "arrangement": "row", "visual_description": "three metal pins",
            },
            "rules": [
                {"id": f"size_{index}", "predicate": "similar_size"},
                {"id": f"space_{index}", "predicate": "ordered_spacing"},
            ],
        })
    result = consensus_generic_relational_proposals(proposals)
    assert result["primary_entity"]["expected_count"] == 3
    assert {row["predicate"] for row in result["rules"]} == {
        "similar_size", "ordered_spacing",
    }
    assert "count_eq" not in {row["predicate"] for row in result["rules"]}
    assert result["predicate_support_votes"] == {
        "similar_size": 4, "ordered_spacing": 4,
    }


def test_generic_runtime_and_zero_delta_bootstrap_abstain_exactly():
    engine = GenericRelationalRuleEngine({"grounder": None, "rules": []})
    score, evidence, facts, diagnostics = engine.evaluate(
        np.zeros((32, 32, 3), dtype=np.uint8)
    )
    assert score == 0.0
    assert evidence == []
    assert facts == {}
    assert diagnostics["abstained"] is True
    labels = np.asarray([0, 0, 1, 1])
    baseline = np.asarray([0.1, 0.2, 0.8, 0.9])
    result = _bootstrap(labels, baseline, baseline.copy(), repeats=2000, seed=1)
    assert result["delta_i_auroc"]["ci95"] == [0.0, 0.0]
    assert result["delta_i_aupr"]["mean"] == 0.0


def test_class_specific_template_distance_uses_frozen_nearest_support():
    references = [[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]
    assert _template_distance(np.asarray([0.0, 0.5, 1.0]), references) == 0.0
    assert np.isclose(
        _template_distance(np.asarray([0.0, 0.5, 0.5]), references), 1.0 / 6.0
    )


def test_pcb1_category_program_crossfits_four_normal_pin_rows():
    root = __import__("pathlib").Path(
        "/home/hjh/Desktop/0Datasets/anomaly/industrial/VisA/pcb1"
    )
    if not root.is_dir():
        return
    supports = select_inpformer_support(root.parent, "visa", "pcb1", 4, seed=1)
    library = build_class_specific_library("visa", "pcb1", supports)
    assert library["runtime_requires_llm"] is False
    assert library["grounder"]["kind"] == PCB1Grounder.kind
    assert all(fold["facts"]["pin_count_error"] == 0.0 for fold in library["crossfit"])
