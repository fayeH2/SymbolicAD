import numpy as np

from neurosymbolic_fsad.evidence_consistency import (
    EvidenceConsistentBridge,
    localized_neural_evidence,
    top_fraction_mean,
)
from neurosymbolic_fsad.fuzzy_logic import FuzzyLogicProgram
from neurosymbolic_fsad.transistor_logic import LogicInference


class _FakeEngine:
    active_ids = ["position_rule"]
    rules = [{
        "id": "position_rule", "fact": "position_error",
        "predicate": "relative_position", "upper": 0.2, "scale": 0.1,
    }]
    program = FuzzyLogicProgram(
        {"op": "NOT", "arg": {"op": "ATOM", "name": "position_rule"}},
        active_ids,
        special_atoms=("grounding", "symbolic_abstain"),
    )


def _logic(violation=0.95):
    localization = np.zeros((10, 10), dtype=np.float32)
    localization[2:5, 2:5] = 1.0
    return LogicInference(
        symbolic_anomaly=violation,
        symbolic_map=violation * localization,
        facts={"position_error": 0.5},
        atoms={"position_rule": {
            "predicate": "relative_position", "fact": "position_error",
            "fact_value": 0.5, "violation": violation,
        }},
        scalar_trace=[], diagnostics={}, explanation="",
        rule_maps={"position_rule": 1.0 - violation * localization},
        localization_maps={"position_rule": localization},
    )


def _bridge():
    return EvidenceConsistentBridge(_FakeEngine(), {
        "parameters": {"localized_top_fraction": 0.1},
        "pixel_confirmation": {"lower": 0.1, "upper": 0.5},
        "rules": {"position_rule": {
            "accepted": True,
            "calibrated_upper": 0.2,
            "calibrated_scale": 0.1,
            "localized_evidence_threshold": 0.2,
            "localized_evidence_scale": 0.1,
        }},
    })


def test_top_fraction_mean_matches_expected_tail():
    values = np.arange(100, dtype=np.float32).reshape(10, 10) / 100.0
    assert np.isclose(top_fraction_mean(values, 0.01), 0.99)
    assert np.isclose(top_fraction_mean(values, 0.10), np.mean(np.arange(90, 100) / 100.0))


def test_localized_evidence_ignores_anomaly_outside_predicate_region():
    anomaly = np.zeros((10, 10), dtype=np.float32)
    localization = np.zeros_like(anomaly)
    localization[2:5, 2:5] = 1.0
    anomaly[7:9, 7:9] = 1.0
    assert localized_neural_evidence(anomaly, localization) == 0.0
    anomaly[3, 3] = 0.8
    assert localized_neural_evidence(anomaly, localization) > 0.7


def test_rule_cannot_change_map_without_colocated_neural_evidence():
    baseline = np.zeros((10, 10), dtype=np.float32)
    baseline[7:9, 7:9] = 0.9
    decision = _bridge().infer("unused", baseline, logic_result=_logic())
    assert decision.rule_evidence["position_rule"]["status"] == (
        "violated_but_not_neurally_confirmed"
    )
    assert np.allclose(decision.fused_map, baseline)


def test_colocated_neural_evidence_executes_and_fuses_rule():
    baseline = np.zeros((10, 10), dtype=np.float32)
    baseline[2:5, 2:5] = 0.8
    decision = _bridge().infer("unused", baseline, logic_result=_logic())
    assert decision.rule_evidence["position_rule"]["status"] == (
        "neural_symbolic_confirmed"
    )
    assert decision.confirmed_symbolic_anomaly > 0.0
    assert np.max(decision.fused_map[2:5, 2:5]) > 0.8
    assert np.allclose(decision.fused_map[7:9, 7:9], 0.0)
