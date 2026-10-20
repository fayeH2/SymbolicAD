"""Calibration and safe execution of static normality rules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .features import FEATURE_CATALOG


VALID_DIRECTIONS = {"stable", "low", "high"}


def _finite_values(rows: Sequence[Mapping[str, float]], feature: str) -> np.ndarray:
    values = np.asarray([row[feature] for row in rows if feature in row], dtype=float)
    return values[np.isfinite(values)]


def _calibrate_one(
    proposal: Mapping[str, Any], rows: Sequence[Mapping[str, float]],
) -> Dict[str, Any]:
    feature = str(proposal["feature"])
    if feature not in FEATURE_CATALOG:
        raise ValueError(f"Unknown rule feature: {feature}")
    values = _finite_values(rows, feature)
    if len(values) != len(rows) or not len(values):
        raise ValueError(f"Feature {feature} is missing from support facts")

    metadata = FEATURE_CATALOG[feature]
    direction = str(proposal.get("direction", metadata["default_direction"])).lower()
    if direction not in VALID_DIRECTIONS:
        direction = str(metadata["default_direction"])
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    robust_sigma = 1.4826 * mad
    observed_range = float(values.max() - values.min())
    base_tolerance = float(metadata["tolerance"])

    # Few-shot calibration must not mistake a single example for zero variance.
    # The catalog tolerance is therefore always a lower bound on the envelope.
    margin = max(
        # FEATURE_CATALOG.tolerance is an absolute safety floor, not a
        # standard error.  Dividing it by sqrt(n) made a 4-shot envelope twice
        # as narrow and caused avoidable normal false positives.
        base_tolerance,
        2.5 * robust_sigma,
        0.5 * observed_range,
    )
    if len(values) == 1:
        reliability = 0.45
    elif len(values) == 2:
        reliability = 0.58
    elif len(values) <= 4:
        reliability = 0.72
    elif len(values) <= 8:
        reliability = 0.84
    else:
        reliability = 0.92

    lower = upper = None
    if direction == "stable":
        lower = float(values.min() - margin)
        upper = float(values.max() + margin)
    elif direction == "low":
        upper = float(values.max() + margin)
    else:
        lower = float(values.min() - margin)

    importance = float(np.clip(proposal.get("importance", 0.7), 0.05, 1.0))
    confidence = float(np.clip(proposal.get("confidence", 0.7), 0.05, 1.0))
    return {
        "id": str(proposal.get("id", f"normal_{feature}")),
        "feature": feature,
        "direction": direction,
        "description": str(proposal.get("reason", metadata["description"])),
        "lower": lower,
        "upper": upper,
        "center": center,
        "soft_scale": float(max(margin, 1e-8)),
        "support_min": float(values.min()),
        "support_max": float(values.max()),
        "support_median": center,
        "support_mad": mad,
        "weight": importance * confidence * reliability,
        "llm_importance": importance,
        "llm_confidence": confidence,
        "few_shot_reliability": reliability,
    }


def calibrate_rule_library(
    proposals: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, float]],
    *,
    class_name: str,
    dataset: str,
    support_paths: Sequence[str],
    normality_summary: str = "",
    llm_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Turn LLM feature proposals into a completely static JSON rule library."""
    if not support_rows:
        raise ValueError("At least one normal support is required")
    seen = set()
    rules = []
    for proposal in proposals:
        feature = str(proposal.get("feature", ""))
        if feature in seen or feature not in FEATURE_CATALOG:
            continue
        seen.add(feature)
        rules.append(_calibrate_one(proposal, support_rows))
    if not rules:
        raise ValueError("No valid rule proposals remained after whitelist validation")

    base_values = _finite_values(support_rows, "map_top1_mean")
    base_center = float(np.median(base_values)) if len(base_values) else 0.0
    base_mad = float(np.median(np.abs(base_values - base_center))) if len(base_values) else 0.0
    base_scale = max(
        1.4826 * base_mad,
        float(FEATURE_CATALOG["map_top1_mean"]["tolerance"]),
        float(np.ptp(base_values)) if len(base_values) else 0.0,
    )
    return {
        "schema_version": "1.0",
        "dataset": dataset,
        "class_name": class_name,
        "normality_summary": normality_summary,
        "support_only_calibration": True,
        "support_count": len(support_rows),
        "support_paths": list(support_paths),
        "baseline_calibration": {"center": base_center, "scale": base_scale},
        "aggregation": {"weighted_mean": 0.7, "max_violation": 0.3},
        "rules": rules,
        "llm": dict(llm_metadata or {}),
    }


def default_rule_proposals(limit: int = 8) -> List[Dict[str, Any]]:
    """Non-LLM ablation, also useful as a fail-fast smoke-test baseline."""
    names = [
        "map_top1_mean", "map_p99", "map_peak_ratio", "map_hot_component",
        "map_lr_imbalance", "map_tb_imbalance", "edge_strength",
        "foreground_area", "foreground_cx", "foreground_cy",
    ][:limit]
    return [
        {
            "feature": name,
            "direction": FEATURE_CATALOG[name]["default_direction"],
            "importance": 0.7,
            "confidence": 0.6,
            "reason": "fixed no-LLM ablation rule",
        }
        for name in names
    ]


class StaticRuleEngine:
    """Evaluate a validated rule library; no dynamic code execution is used."""

    def __init__(self, library: Mapping[str, Any]):
        self.library = dict(library)
        if self.library.get("schema_version") != "1.0":
            raise ValueError("Unsupported rule-library schema")
        self.rules = list(self.library.get("rules", []))
        if not self.rules:
            raise ValueError("Rule library is empty")
        for rule in self.rules:
            if rule.get("feature") not in FEATURE_CATALOG:
                raise ValueError(f"Unknown feature in library: {rule.get('feature')}")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "StaticRuleEngine":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.library, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        return path

    @staticmethod
    def _violation(rule: Mapping[str, Any], value: float) -> float:
        scale = max(float(rule["soft_scale"]), 1e-8)
        lower, upper = rule.get("lower"), rule.get("upper")
        distance = 0.0
        if lower is not None and value < float(lower):
            distance = float(lower) - value
        elif upper is not None and value > float(upper):
            distance = value - float(upper)
        # Smoothly saturate: a violation one calibration margin outside the
        # envelope contributes tanh(1)=0.76, rather than an unstable hard jump.
        return float(math.tanh(max(distance, 0.0) / scale))

    def score(self, facts: Mapping[str, float]) -> Tuple[float, List[Dict[str, Any]]]:
        evidence = []
        weighted_sum = 0.0
        weight_total = 0.0
        maximum = 0.0
        for rule in self.rules:
            feature = str(rule["feature"])
            if feature not in facts or not np.isfinite(facts[feature]):
                raise ValueError(f"Missing/non-finite inference fact: {feature}")
            value = float(facts[feature])
            violation = self._violation(rule, value)
            weight = max(float(rule.get("weight", 1.0)), 0.0)
            weighted_sum += weight * violation
            weight_total += weight
            maximum = max(maximum, violation)
            evidence.append({
                "rule_id": rule["id"], "feature": feature, "value": value,
                "lower": rule.get("lower"), "upper": rule.get("upper"),
                "violation": violation, "weight": weight,
            })
        mean = weighted_sum / max(weight_total, 1e-12)
        aggregation = self.library.get("aggregation", {})
        mean_weight = float(aggregation.get("weighted_mean", 0.7))
        max_weight = float(aggregation.get("max_violation", 0.3))
        score = (mean_weight * mean + max_weight * maximum) / max(
            mean_weight + max_weight, 1e-12
        )
        evidence.sort(key=lambda item: item["weight"] * item["violation"], reverse=True)
        return float(np.clip(score, 0.0, 1.0)), evidence

    def normalize_baseline(self, raw_score: float) -> float:
        calibration = self.library.get("baseline_calibration", {})
        center = float(calibration.get("center", 0.0))
        scale = max(float(calibration.get("scale", 1.0)), 1e-8)
        # center maps to 0.119; center + 2*scale maps to 0.5.
        z_value = np.clip((float(raw_score) - center) / scale - 2.0, -30.0, 30.0)
        return float(1.0 / (1.0 + np.exp(-z_value)))

    def fused_score(
        self, raw_baseline_score: float, rule_score: float, alpha: float = 0.25,
    ) -> float:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        baseline = self.normalize_baseline(raw_baseline_score)
        return float((1.0 - alpha) * baseline + alpha * rule_score)
