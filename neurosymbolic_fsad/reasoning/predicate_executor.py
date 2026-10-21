"""Predicate executors with explicit UNKNOWN and rule-specific grounding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..grounding.shape_grounder import length_violation_map, resized_instance_mask
from ..perception.scene_graph import SceneGraph
from ..rules.rule_schema import RuleSpec
from .violation_result import RuleStatus, ViolationResult


@dataclass
class ExecutionContext:
    scene: SceneGraph
    map_size: int = 256
    entity_verification: Mapping[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def verification_confidence(self, entity_type: str) -> float:
        if entity_type in self.entity_verification:
            return float(np.clip(self.entity_verification[entity_type], 0.0, 1.0))
        instances = self.scene.instances_of(entity_type)
        if not instances:
            return float(np.clip(self.scene.grounding_confidence, 0.0, 1.0))
        return float(np.clip(
            min(
                self.scene.grounding_confidence,
                float(np.mean([getattr(row, "confidence", 0.0) for row in instances])),
            ),
            0.0,
            1.0,
        ))


def _range(rule: RuleSpec) -> tuple[Optional[float], Optional[float]]:
    lower = rule.parameters.get("lower")
    upper = rule.parameters.get("upper")
    return (
        None if lower is None else float(lower),
        None if upper is None else float(upper),
    )


def _range_violation(value: float, lower: float, upper: float) -> float:
    if lower <= value <= upper:
        return 0.0
    distance = lower - value if value < lower else value - upper
    return float(np.clip(distance / max(upper - lower, abs(value), 1e-6), 0.0, 1.0))


class LengthRangeExecutor:
    def execute(self, rule: RuleSpec, context: ExecutionContext) -> ViolationResult:
        entity = rule.entity_types[0]
        instances = context.scene.instances_of(entity)
        confidence = context.verification_confidence(entity)
        lower, upper = _range(rule)
        if lower is None or upper is None or lower > upper:
            return ViolationResult.unknown(
                rule.rule_id, "invalid_length_range", map_size=context.map_size,
                grounding_strategy="instance_mask_endpoint_difference",
            )
        if confidence < float(rule.parameters.get("minimum_confidence", 0.5)):
            return ViolationResult.unknown(
                rule.rule_id, "entity_verification_insufficient",
                evidence_confidence=confidence, map_size=context.map_size,
                grounding_strategy="instance_mask_endpoint_difference",
            )
        if not instances:
            return ViolationResult.unknown(
                rule.rule_id, "no_grounded_instances",
                evidence_confidence=confidence, map_size=context.map_size,
                grounding_strategy="instance_mask_endpoint_difference",
            )
        measured = []
        for instance in instances:
            value = getattr(instance, "attributes", {}).get("length")
            if value is None or not np.isfinite(float(value)):
                continue
            measured.append((instance, float(value)))
        if not measured:
            return ViolationResult.unknown(
                rule.rule_id, "length_not_observed",
                evidence_confidence=confidence, map_size=context.map_size,
                grounding_strategy="instance_mask_endpoint_difference",
            )
        violations = [
            (_range_violation(value, lower, upper), instance, value)
            for instance, value in measured
        ]
        score, instance, value = max(violations, key=lambda row: row[0])
        if score <= 0.0:
            return ViolationResult(
                rule.rule_id, RuleStatus.SATISFIED, 0.0, confidence,
                grounding_strategy="instance_mask_endpoint_difference",
                anomaly_map=np.zeros((context.map_size, context.map_size), dtype=np.float32),
                attribution_uncertainty=1.0 - confidence,
                details={"values": [row[2] for row in violations], "range": [lower, upper]},
            )
        target = float(np.clip(value, lower, upper))
        reference_endpoints = None
        if (
            rule.parameters.get("reference_endpoint_a") is not None
            and rule.parameters.get("reference_endpoint_b") is not None
        ):
            reference_endpoints = (
                rule.parameters["reference_endpoint_a"],
                rule.parameters["reference_endpoint_b"],
            )
        repair = length_violation_map(
            instance, target, context.map_size,
            reference_endpoints=reference_endpoints,
        )
        return ViolationResult(
            rule.rule_id, RuleStatus.VIOLATED, score, confidence,
            witness_instances=[str(instance.instance_id)],
            grounding_strategy="instance_mask_endpoint_difference",
            anomaly_map=repair.anomaly_map * score,
            attribution_uncertainty=float(np.mean(repair.uncertainty_map[repair.anomaly_map > 0]))
            if np.any(repair.anomaly_map > 0) else 1.0,
            details={
                "actual": value, "range": [lower, upper],
                "counterfactual_target": target, "repair": repair.strategy,
            },
        )


class AspectRangeExecutor:
    attribute = "aspect_ratio"

    def execute(self, rule: RuleSpec, context: ExecutionContext) -> ViolationResult:
        return _execute_mask_attribute_range(
            rule, context, self.attribute, "instance_mask"
        )


class ColorRangeExecutor:
    attribute = "color_h"

    def execute(self, rule: RuleSpec, context: ExecutionContext) -> ViolationResult:
        return _execute_mask_attribute_range(
            rule, context, self.attribute, "instance_mask"
        )


def _execute_mask_attribute_range(
    rule: RuleSpec,
    context: ExecutionContext,
    attribute: str,
    strategy: str,
) -> ViolationResult:
    entity = rule.entity_types[0]
    confidence = context.verification_confidence(entity)
    lower, upper = _range(rule)
    if lower is None or upper is None or lower > upper:
        return ViolationResult.unknown(
            rule.rule_id, f"invalid_{attribute}_range", map_size=context.map_size,
            grounding_strategy=strategy,
        )
    if confidence < float(rule.parameters.get("minimum_confidence", 0.5)):
        return ViolationResult.unknown(
            rule.rule_id, "entity_verification_insufficient",
            evidence_confidence=confidence, map_size=context.map_size,
            grounding_strategy=strategy,
        )
    observed = []
    for instance in context.scene.instances_of(entity):
        value = getattr(instance, "attributes", {}).get(attribute)
        if value is not None and np.isfinite(float(value)):
            observed.append((instance, float(value)))
    if not observed:
        return ViolationResult.unknown(
            rule.rule_id, f"{attribute}_not_observed",
            evidence_confidence=confidence, map_size=context.map_size,
            grounding_strategy=strategy,
        )
    rows = [(_range_violation(value, lower, upper), instance, value)
            for instance, value in observed]
    score, instance, value = max(rows, key=lambda row: row[0])
    if score <= 0.0:
        return ViolationResult(
            rule.rule_id, RuleStatus.SATISFIED, 0.0, confidence,
            grounding_strategy=strategy,
            anomaly_map=np.zeros((context.map_size, context.map_size), dtype=np.float32),
            attribution_uncertainty=1.0 - confidence,
            details={"attribute": attribute, "values": [row[2] for row in rows]},
        )
    witness = resized_instance_mask(instance, context.map_size) * score
    return ViolationResult(
        rule.rule_id, RuleStatus.VIOLATED, score, confidence,
        witness_instances=[str(instance.instance_id)], grounding_strategy=strategy,
        anomaly_map=witness, attribution_uncertainty=1.0 - confidence,
        details={"attribute": attribute, "actual": value, "range": [lower, upper]},
    )


class CountEqExecutor:
    def execute(self, rule: RuleSpec, context: ExecutionContext) -> ViolationResult:
        entity = rule.entity_types[0]
        expected = rule.parameters.get("expected")
        confidence = context.verification_confidence(entity)
        if expected is None or int(expected) < 0:
            return ViolationResult.unknown(
                rule.rule_id, "invalid_expected_count", map_size=context.map_size,
                grounding_strategy="instance_mask_or_slot_probability",
            )
        if not context.scene.observation_complete or confidence < float(
            rule.parameters.get("minimum_confidence", 0.5)
        ):
            return ViolationResult.unknown(
                rule.rule_id, "count_observation_incomplete",
                evidence_confidence=confidence, map_size=context.map_size,
                grounding_strategy="instance_mask_or_slot_probability",
            )
        instances = context.scene.instances_of(entity)
        actual = len(instances)
        expected = int(expected)
        if actual == expected:
            return ViolationResult(
                rule.rule_id, RuleStatus.SATISFIED, 0.0, confidence,
                grounding_strategy="instance_mask_or_slot_probability",
                anomaly_map=np.zeros((context.map_size, context.map_size), dtype=np.float32),
                attribution_uncertainty=1.0 - confidence,
                details={"expected": expected, "actual": actual},
            )
        score = float(np.clip(abs(actual - expected) / max(expected, 1), 0.0, 1.0))
        anomaly = np.zeros((context.map_size, context.map_size), dtype=np.float32)
        witness_instances = []
        missing_slots = []
        uncertainty = 1.0 - confidence
        if actual > expected:
            extras = instances[expected:]
            witness_instances = [str(row.instance_id) for row in extras]
            for instance in extras:
                anomaly = np.maximum(
                    anomaly, resized_instance_mask(instance, context.map_size) * score
                )
        else:
            missing_count = expected - actual
            candidates = sorted(context.scene.slot_probability_maps.items())
            for slot_id, probability in candidates[:missing_count]:
                missing_slots.append(str(slot_id))
                resized = np.asarray(probability, dtype=np.float32)
                if resized.shape != anomaly.shape:
                    import cv2
                    resized = cv2.resize(
                        resized, anomaly.shape[::-1], interpolation=cv2.INTER_LINEAR
                    )
                anomaly = np.maximum(anomaly, np.clip(resized, 0.0, 1.0) * score)
            if len(missing_slots) < missing_count:
                missing_slots.extend(
                    f"unlocalized_missing_{index}"
                    for index in range(len(missing_slots), missing_count)
                )
                uncertainty = max(uncertainty, 0.8)
        return ViolationResult(
            rule.rule_id, RuleStatus.VIOLATED, score, confidence,
            witness_instances=witness_instances, missing_slots=missing_slots,
            grounding_strategy="instance_mask_or_slot_probability",
            anomaly_map=anomaly, attribution_uncertainty=uncertainty,
            details={"expected": expected, "actual": actual},
        )


__all__ = [
    "ExecutionContext", "LengthRangeExecutor", "AspectRangeExecutor",
    "ColorRangeExecutor", "CountEqExecutor",
]
