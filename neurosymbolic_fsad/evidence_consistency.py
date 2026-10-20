"""Evidence-consistent bridge between executable rules and INP-Former.

The rule grounder and INP-Former do not share a semantic feature space.  This
module therefore never treats an LLM rule score as if it were an INP feature.
Instead it asks a narrower, testable question for every rule: does INP-Former
show unusually strong anomaly evidence inside the pixels localized by that
rule?  A rule may affect the fused heatmap only when the answer is yes.

All thresholds are calibrated on normal images.  Query labels and masks are
not accepted by this API, which makes accidental test-label tuning harder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .fuzzy_logic import fuzzy_or
from .transistor_logic import LogicInference, TransistorLogicEngine


def top_fraction_mean(array: np.ndarray, fraction: float = 0.01) -> float:
    """Use the same top-1% pooling as the frozen INP-Former adapter."""
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    if not values.size:
        return 0.0
    count = max(1, int(values.size * float(fraction)))
    return float(np.partition(values, values.size - count)[-count:].mean())


def localized_neural_evidence(
    anomaly_map: np.ndarray,
    localization_map: np.ndarray,
    *,
    top_fraction: float = 0.10,
    minimum_localizer: float = 0.05,
) -> float:
    """Robust top-tail INP evidence inside one predicate's spatial support."""
    anomaly = np.clip(np.asarray(anomaly_map, dtype=np.float64), 0.0, 1.0)
    localization = np.clip(np.asarray(localization_map, dtype=np.float64), 0.0, 1.0)
    if anomaly.shape != localization.shape:
        raise ValueError(
            f"anomaly/localization shapes differ: {anomaly.shape} vs {localization.shape}"
        )
    selected = localization > float(minimum_localizer)
    if not np.any(selected):
        return 0.0
    values = anomaly[selected]
    weights = localization[selected]
    count = max(1, int(math.ceil(values.size * float(top_fraction))))
    # Rank by jointly high neural and symbolic spatial support, then report the
    # weighted neural magnitude.  This is an agreement statistic, not a label.
    ranking = values * weights
    indices = np.argpartition(ranking, ranking.size - count)[-count:]
    return float(np.sum(values[indices] * weights[indices]) /
                 max(np.sum(weights[indices]), 1e-12))


def _higher_quantile(values: Sequence[float], quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        raise ValueError("normal calibration values cannot be empty")
    try:
        return float(np.quantile(array, quantile, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(array, quantile, interpolation="higher"))


def _robust_scale(values: Sequence[float], floor: float = 1e-4) -> float:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return max(1.4826 * mad, float(np.std(array)) * 0.25, float(floor))


@dataclass
class BridgeInference:
    raw_symbolic_anomaly: float
    raw_symbolic_map: np.ndarray
    confirmed_symbolic_anomaly: float
    confirmed_symbolic_map: np.ndarray
    fused_score: float
    fused_map: np.ndarray
    rule_evidence: Dict[str, Dict[str, Any]]
    explanation: str


class EvidenceConsistentBridge:
    """Calibrated symbolic/INP spatial-consistency gate.

    The symbolic program remains an explicit fuzzy-logic AST.  The only change
    is its atom truth: ``1 - (rule_violation AND neural_confirmation)``.  Rules
    that are satisfied, unstable on normal validation, or spatially unsupported
    by INP-Former become a neutral truth value of one and cannot alter the
    baseline output.
    """

    schema_version = "10.0-evidence-consistent-neuro-symbolic-bridge"

    def __init__(
        self,
        engine: TransistorLogicEngine,
        calibration: Mapping[str, Any],
    ):
        self.engine = engine
        self.calibration = dict(calibration)
        self.rule_calibration = {
            str(key): dict(value)
            for key, value in self.calibration["rules"].items()
        }

    @classmethod
    def calibrate(
        cls,
        engine: TransistorLogicEngine,
        normal_images: Sequence[Any],
        normal_maps: Sequence[np.ndarray],
        *,
        normal_identifiers: Sequence[str] | None = None,
        protocol: str,
        fact_quantile: float = 0.95,
        evidence_quantile: float = 0.95,
        pixel_lower_quantile: float = 0.95,
        pixel_upper_quantile: float = 0.999,
        max_threshold_shift_scales: float = 3.0,
        maximum_validation_fpr: float = 0.10,
        top_fraction: float = 0.10,
    ) -> "EvidenceConsistentBridge":
        if len(normal_images) != len(normal_maps) or not normal_images:
            raise ValueError("normal image/map calibration sets must be non-empty and aligned")
        if normal_identifiers is None:
            normal_identifiers = [f"normal-validation://{i}" for i in range(len(normal_images))]
        if len(normal_identifiers) != len(normal_images):
            raise ValueError("normal identifier count differs from calibration count")

        fact_values = {name: [] for name in engine.active_ids}
        evidence_values = {name: [] for name in engine.active_ids}
        raw_violations = {name: [] for name in engine.active_ids}
        for image, anomaly_map in zip(normal_images, normal_maps):
            result = engine.evaluate(image)
            for name in engine.active_ids:
                atom = result.atoms[name]
                fact_values[name].append(float(atom["fact_value"]))
                raw_violations[name].append(float(atom["violation"]))
                evidence_values[name].append(localized_neural_evidence(
                    anomaly_map, result.localization_maps[name],
                    top_fraction=top_fraction,
                ))

        rule_rows: Dict[str, Dict[str, Any]] = {}
        for rule in engine.rules:
            name = str(rule["id"])
            values = np.asarray(fact_values[name], dtype=np.float64)
            original_upper = float(rule["upper"])
            original_scale = max(float(rule["scale"]), 1e-8)
            calibrated_upper = max(
                original_upper, _higher_quantile(values, fact_quantile)
            )
            calibrated_scale = max(original_scale, _robust_scale(values))
            violations = np.tanh(
                np.maximum(0.0, values - calibrated_upper) / calibrated_scale
            )
            validation_fpr = float(np.mean(violations > 1e-8))
            threshold_shift_scales = (
                calibrated_upper - original_upper
            ) / original_scale
            accepted = bool(
                validation_fpr <= maximum_validation_fpr
                and threshold_shift_scales <= max_threshold_shift_scales
            )

            evidence = np.asarray(evidence_values[name], dtype=np.float64)
            evidence_threshold = _higher_quantile(evidence, evidence_quantile)
            evidence_high = _higher_quantile(
                evidence, min(0.999, max(evidence_quantile, 0.99))
            )
            evidence_scale = max(
                evidence_high - evidence_threshold,
                _robust_scale(evidence, floor=0.002),
            )
            rule_rows[name] = {
                "accepted": accepted,
                "rejection_reason": (
                    None if accepted else
                    "normal-validation instability or excessive threshold shift"
                ),
                "original_upper": original_upper,
                "calibrated_upper": calibrated_upper,
                "original_scale": original_scale,
                "calibrated_scale": calibrated_scale,
                "threshold_shift_scales": float(threshold_shift_scales),
                "raw_validation_trigger_rate": float(
                    np.mean(np.asarray(raw_violations[name]) > 1e-8)
                ),
                "calibrated_validation_fpr": validation_fpr,
                "localized_evidence_threshold": evidence_threshold,
                "localized_evidence_scale": evidence_scale,
                "localized_evidence_median": float(np.median(evidence)),
                "localized_evidence_max": float(np.max(evidence)),
                "normal_fact_values": values.tolist(),
                "normal_localized_evidence": evidence.tolist(),
            }

        flat_pixels = np.concatenate([
            np.asarray(item, dtype=np.float64).reshape(-1) for item in normal_maps
        ])
        pixel_lower = _higher_quantile(flat_pixels, pixel_lower_quantile)
        pixel_upper = _higher_quantile(flat_pixels, pixel_upper_quantile)
        calibration = {
            "schema_version": cls.schema_version,
            "protocol": str(protocol),
            "normal_validation_count": int(len(normal_images)),
            "normal_validation_identifiers": list(normal_identifiers),
            "query_images_or_labels_used": False,
            "parameters": {
                "fact_quantile": float(fact_quantile),
                "evidence_quantile": float(evidence_quantile),
                "pixel_lower_quantile": float(pixel_lower_quantile),
                "pixel_upper_quantile": float(pixel_upper_quantile),
                "max_threshold_shift_scales": float(max_threshold_shift_scales),
                "maximum_validation_fpr": float(maximum_validation_fpr),
                "localized_top_fraction": float(top_fraction),
            },
            "pixel_confirmation": {
                "lower": pixel_lower,
                "upper": max(pixel_upper, pixel_lower + 1e-6),
            },
            "rules": rule_rows,
        }
        return cls(engine, calibration)

    def infer(
        self,
        image: Any,
        baseline_map: np.ndarray,
        *,
        logic_result: LogicInference | None = None,
    ) -> BridgeInference:
        result = logic_result or self.engine.evaluate(image)
        baseline = np.clip(np.asarray(baseline_map, dtype=np.float64), 0.0, 1.0)
        pixel = self.calibration["pixel_confirmation"]
        neural_confirmation_map = np.clip(
            (baseline - float(pixel["lower"])) /
            max(float(pixel["upper"]) - float(pixel["lower"]), 1e-8),
            0.0, 1.0,
        )

        scalar_truths: Dict[str, Any] = {"grounding": 1.0, "symbolic_abstain": 0.0}
        map_truths: Dict[str, Any] = {
            "grounding": np.ones_like(baseline),
            "symbolic_abstain": np.zeros_like(baseline),
        }
        evidence_rows: Dict[str, Dict[str, Any]] = {}
        top_fraction = float(self.calibration["parameters"]["localized_top_fraction"])
        for rule in self.engine.rules:
            name = str(rule["id"])
            state = self.rule_calibration[name]
            atom = result.atoms[name]
            fact_value = float(atom["fact_value"])
            violation = float(math.tanh(
                max(0.0, fact_value - float(state["calibrated_upper"])) /
                max(float(state["calibrated_scale"]), 1e-8)
            ))
            localized_evidence = localized_neural_evidence(
                baseline, result.localization_maps[name], top_fraction=top_fraction
            )
            excess = max(
                0.0, localized_evidence - float(state["localized_evidence_threshold"])
            )
            gate = float(math.tanh(
                excess / max(float(state["localized_evidence_scale"]), 1e-8)
            )) if bool(state["accepted"]) else 0.0
            supported_violation = violation * gate
            supported_map = np.clip(
                supported_violation
                * np.asarray(result.localization_maps[name], dtype=np.float64)
                * neural_confirmation_map,
                0.0, 1.0,
            )
            scalar_truths[name] = 1.0 - supported_violation
            map_truths[name] = 1.0 - supported_map
            if not state["accepted"]:
                status = "abstained_unstable_on_normal_validation"
            elif violation <= 1e-8:
                status = "rule_satisfied"
            elif gate <= 1e-8:
                status = "violated_but_not_neurally_confirmed"
            else:
                status = "neural_symbolic_confirmed"
            evidence_rows[name] = {
                "predicate": atom["predicate"],
                "fact": atom["fact"],
                "fact_value": fact_value,
                "normal_upper": float(state["calibrated_upper"]),
                "rule_violation": violation,
                "localized_inp_evidence": localized_evidence,
                "normal_inp_evidence_threshold": float(
                    state["localized_evidence_threshold"]
                ),
                "neural_confirmation_gate": gate,
                "supported_violation": supported_violation,
                "status": status,
            }

        confirmed_symbolic, _ = self.engine.program.evaluate(scalar_truths)
        confirmed_map, _ = self.engine.program.evaluate(map_truths)
        confirmed_map = np.clip(np.asarray(confirmed_map, dtype=np.float64), 0.0, 1.0)
        fused_map = np.asarray(fuzzy_or(baseline, confirmed_map), dtype=np.float32)
        fused_score = top_fraction_mean(fused_map, 0.01)

        violations = [
            (name, row) for name, row in evidence_rows.items()
            if row["rule_violation"] > 1e-8
        ]
        if not violations:
            explanation = "All executable normality rules are satisfied."
        else:
            clauses = []
            for name, row in violations:
                if row["status"] == "neural_symbolic_confirmed":
                    suffix = "and INP-Former confirms anomaly evidence in the same region"
                elif row["status"] == "violated_but_not_neurally_confirmed":
                    suffix = "but INP-Former does not confirm co-located evidence; fusion abstains"
                else:
                    suffix = "but the rule is unstable on normal validation; fusion abstains"
                clauses.append(
                    f"{name}: {row['fact']}={row['fact_value']:.4f} exceeds "
                    f"normal upper {row['normal_upper']:.4f}, {suffix}"
                )
            explanation = "; ".join(clauses)

        return BridgeInference(
            raw_symbolic_anomaly=float(result.symbolic_anomaly),
            raw_symbolic_map=np.asarray(result.symbolic_map, dtype=np.float32),
            confirmed_symbolic_anomaly=float(confirmed_symbolic),
            confirmed_symbolic_map=confirmed_map.astype(np.float32),
            fused_score=float(fused_score),
            fused_map=fused_map,
            rule_evidence=evidence_rows,
            explanation=explanation,
        )


__all__ = [
    "BridgeInference", "EvidenceConsistentBridge", "localized_neural_evidence",
    "top_fraction_mean",
]
