"""原子谓词执行器：视觉事实 -> 三态结论 -> 规则专属空间责任图。

所有执行器遵循相同顺序：先检查规则参数，再检查实体/观测可靠性，然后计算
真值，最后只为 ``VIOLATED`` 生成 anomaly map。INP reconstruction residual 不在
此处出现，因此“外观正常但数量/关系错误”的逻辑异常不会被视觉残差否决。
"""

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
    """单张 query 的可执行视觉事实；这里故意没有 GT 或异常标签字段。"""
    scene: SceneGraph
    map_size: int = 256
    entity_verification: Mapping[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def verification_confidence(self, entity_type: str) -> float:
        """合并实例检测与实体对应可靠性，只决定能否判断，不决定是否异常。"""
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
    """执行显式 PCA 主轴长度约束，并用端点反事实差异定位。"""
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
        # 证据不足必须 UNKNOWN；返回 SATISFIED 会造成系统性漏检和伪支持。
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
        # 最近的合法边界就是局部反事实目标；变化的端点区域承担空间责任。
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
    # 属性范围规则定位到真正超界的实例 mask，而不是整张框或 INP 热区。
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
    """执行数量约束：多余实例用实例 mask，缺失实例用支持集槽位概率图。"""
    def execute(self, rule: RuleSpec, context: ExecutionContext) -> ViolationResult:
        entity = rule.entity_types[0]
        expected = rule.parameters.get("expected")
        tolerance = rule.parameters.get("tolerance", 0)
        confidence = context.verification_confidence(entity)
        if (
            expected is None or tolerance is None
            or float(expected) != int(float(expected))
            or float(tolerance) != int(float(tolerance))
            or int(float(expected)) < 0 or int(float(tolerance)) < 0
        ):
            return ViolationResult.unknown(
                rule.rule_id, "invalid_count_parameters", map_size=context.map_size,
                grounding_strategy="instance_mask_or_slot_probability",
            )
        # 检测不完整时，少检测到实例不能被解释成真实“缺件”。
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
        expected = int(float(expected))
        tolerance = int(float(tolerance))
        if abs(actual - expected) <= tolerance:
            return ViolationResult(
                rule.rule_id, RuleStatus.SATISFIED, 0.0, confidence,
                grounding_strategy="instance_mask_or_slot_probability",
                anomaly_map=np.zeros((context.map_size, context.map_size), dtype=np.float32),
                attribution_uncertainty=1.0 - confidence,
                details={"expected": expected, "actual": actual, "tolerance": tolerance},
            )
        excess = abs(actual - expected) - tolerance
        score = float(np.clip(excess / max(expected, 1), 0.0, 1.0))
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
            # 缺失实体没有 query mask，只能由 support 对齐得到的 slot map 定位。
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
            details={"expected": expected, "actual": actual, "tolerance": tolerance},
        )


def _count_observation_confidence(
    context: ExecutionContext,
    entity_type: str,
) -> tuple[bool, float]:
    """Return the calibrated count-completeness decision for one entity.

    Count completeness is entity-specific.  A low-confidence container must
    not make an otherwise calibrated peach count UNKNOWN, and the presence of
    one detected object must not certify that all almonds were detected.  The
    perception stage therefore supplies a validation-derived mapping in
    ``ExecutionContext.metadata``.  Without that mapping we fail closed.
    """
    complete = context.metadata.get("count_observation_complete", {})
    reliability = context.metadata.get("count_observation_confidence", {})
    if not isinstance(complete, Mapping) or not isinstance(reliability, Mapping):
        return False, 0.0
    flag = bool(complete.get(entity_type, False))
    confidence = float(np.clip(reliability.get(entity_type, 0.0), 0.0, 1.0))
    return flag, confidence


class CountInExecutor:
    """Execute ``count(entity, region) == expected +/- tolerance``.

    Region membership uses the instance centroid and the region's pixel mask.
    For a missing-in-region violation, an observed instance in another region
    is a concrete witness and is localized directly.  If the entity is absent
    everywhere, the logical score is still valid only when count completeness
    was calibrated on independent normal data; spatial attribution then uses a
    support slot map if available and otherwise reports high uncertainty.
    """

    strategy = "region_mask_and_instance_witness"

    @staticmethod
    def _contains(region: Any, instance: Any) -> bool:
        mask = np.asarray(region.mask, dtype=bool)
        height, width = mask.shape
        x = int(np.clip(np.floor(float(instance.centroid[0]) * width), 0, width - 1))
        y = int(np.clip(np.floor(float(instance.centroid[1]) * height), 0, height - 1))
        return bool(mask[y, x])

    def execute(self, rule: RuleSpec, context: ExecutionContext) -> ViolationResult:
        if len(rule.entity_types) != 2:
            return ViolationResult.unknown(
                rule.rule_id, "count_in_requires_entity_and_region",
                map_size=context.map_size, grounding_strategy=self.strategy,
            )
        entity, region_type = rule.entity_types
        expected = rule.parameters.get("expected")
        tolerance = rule.parameters.get("tolerance", 0)
        if (
            expected is None or tolerance is None
            or float(expected) != int(float(expected))
            or float(tolerance) != int(float(tolerance))
            or int(float(expected)) < 0 or int(float(tolerance)) < 0
        ):
            return ViolationResult.unknown(
                rule.rule_id, "invalid_count_parameters",
                map_size=context.map_size, grounding_strategy=self.strategy,
            )

        regions = context.scene.instances_of(region_type)
        if len(regions) != 1:
            return ViolationResult.unknown(
                rule.rule_id, "region_grounding_not_unique",
                map_size=context.map_size, grounding_strategy=self.strategy,
                details={"region_type": region_type, "region_count": len(regions)},
            )
        complete, entity_confidence = _count_observation_confidence(context, entity)
        region_confidence = float(np.clip(regions[0].confidence, 0.0, 1.0))
        confidence = min(entity_confidence, region_confidence)
        minimum = float(rule.parameters.get("minimum_confidence", 0.5))
        if not complete or confidence < minimum:
            return ViolationResult.unknown(
                rule.rule_id, "count_observation_incomplete",
                evidence_confidence=confidence, map_size=context.map_size,
                grounding_strategy=self.strategy,
                details={"entity_type": entity, "region_type": region_type},
            )

        all_instances = context.scene.instances_of(entity)
        inside = [row for row in all_instances if self._contains(regions[0], row)]
        outside = [row for row in all_instances if row not in inside]
        actual = len(inside)
        expected = int(float(expected))
        tolerance = int(float(tolerance))
        delta = actual - expected
        base_details = {
            "entity_type": entity, "region_type": region_type,
            "expected": expected, "actual": actual, "tolerance": tolerance,
            "total_observed": len(all_instances),
        }
        if abs(delta) <= tolerance:
            return ViolationResult(
                rule.rule_id, RuleStatus.SATISFIED, 0.0, confidence,
                grounding_strategy=self.strategy,
                anomaly_map=np.zeros((context.map_size, context.map_size), dtype=np.float32),
                attribution_uncertainty=1.0 - confidence,
                details=base_details,
            )

        magnitude = abs(delta) - tolerance
        score = float(np.clip(magnitude / max(expected, 1), 0.0, 1.0))
        anomaly = np.zeros((context.map_size, context.map_size), dtype=np.float32)
        witnesses = []
        missing_slots = []
        uncertainty = 1.0 - confidence
        if delta > tolerance:
            # Which of several in-region instances is the extra one is not
            # identifiable without slot matching, so all are marked and the
            # ambiguity is recorded explicitly.
            witnesses = [str(row.instance_id) for row in inside]
            for instance in inside:
                anomaly = np.maximum(
                    anomaly, resized_instance_mask(instance, context.map_size) * score
                )
            uncertainty = max(uncertainty, 0.5 if len(inside) > 1 else uncertainty)
        else:
            # Misplaced instances are stronger witnesses than a generic empty
            # slot: this is the expected path for LOCO compartment swaps.
            needed = max(expected - actual - tolerance, 0)
            misplaced = sorted(outside, key=lambda row: row.confidence, reverse=True)[:needed]
            witnesses = [str(row.instance_id) for row in misplaced]
            for instance in misplaced:
                anomaly = np.maximum(
                    anomaly, resized_instance_mask(instance, context.map_size) * score
                )
            remaining = needed - len(misplaced)
            if remaining:
                prefix = f"{rule.rule_id}:"
                slots = [
                    (slot_id, values) for slot_id, values
                    in sorted(context.scene.slot_probability_maps.items())
                    if str(slot_id).startswith(prefix)
                ][:remaining]
                for slot_id, probability in slots:
                    resized = np.asarray(probability, dtype=np.float32)
                    if resized.shape != anomaly.shape:
                        import cv2
                        resized = cv2.resize(
                            resized, anomaly.shape[::-1], interpolation=cv2.INTER_LINEAR
                        )
                    anomaly = np.maximum(anomaly, np.clip(resized, 0.0, 1.0) * score)
                    missing_slots.append(str(slot_id))
                for index in range(len(slots), remaining):
                    missing_slots.append(f"unlocalized_missing_{index}")
                uncertainty = max(uncertainty, 0.8)
        return ViolationResult(
            rule.rule_id, RuleStatus.VIOLATED, score, confidence,
            witness_instances=witnesses, missing_slots=missing_slots,
            grounding_strategy=self.strategy, anomaly_map=anomaly,
            attribution_uncertainty=uncertainty, details=base_details,
        )


def _presence_observation(
    context: ExecutionContext,
    entity_type: str,
) -> tuple[bool, float]:
    """Presence completeness: does the detector reliably see this entity at all?

    Presence is a weaker requirement than exact counting.  A region that either
    contains the entity or has been calibrated to reliably reveal it supports a
    logical decision; if no calibration exists we fail closed.
    """
    metadata = context.metadata
    complete_map = metadata.get(
        "presence_observation_complete",
        metadata.get("count_observation_complete", {}),
    )
    confidence_map = metadata.get(
        "presence_observation_confidence",
        metadata.get("count_observation_confidence", {}),
    )
    if not isinstance(complete_map, Mapping) or not isinstance(confidence_map, Mapping):
        return False, 0.0
    return (
        bool(complete_map.get(entity_type, False)),
        float(np.clip(confidence_map.get(entity_type, 0.0), 0.0, 1.0)),
    )


def _unique_region(context: ExecutionContext, region_type: str) -> Optional[Any]:
    regions = context.scene.instances_of(region_type)
    if len(regions) != 1:
        return None
    return regions[0]


def _centroid_inside(region: Any, instance: Any) -> bool:
    mask = np.asarray(region.mask, dtype=bool)
    height, width = mask.shape
    x = int(np.clip(np.floor(float(instance.centroid[0]) * width), 0, width - 1))
    y = int(np.clip(np.floor(float(instance.centroid[1]) * height), 0, height - 1))
    return bool(mask[y, x])


def _mean_centroid(instances) -> tuple[float, float]:
    xs = [float(row.centroid[0]) for row in instances]
    ys = [float(row.centroid[1]) for row in instances]
    return float(np.mean(xs)), float(np.mean(ys))


class ExistsExecutor:
    """Presence predicate ``exists(entity[, region])``.

    Qwen encodes "granola is present in the right compartment" as COUNT=1.
    Exact-count semantics would also reject a scene containing two almonds,
    which is normal for LOCO.  This executor therefore tests presence
    (``actual >= minimum_count``), and keeps the observation-completeness
    contract: an uncalibrated detector yields ``UNKNOWN``, never "missing".
    """

    strategy = "presence_region_mask"

    def execute(self, rule: RuleSpec, context: ExecutionContext) -> ViolationResult:
        if len(rule.entity_types) not in {1, 2}:
            return ViolationResult.unknown(
                rule.rule_id, "exists_requires_entity_and_optional_region",
                map_size=context.map_size, grounding_strategy=self.strategy,
            )
        entity = rule.entity_types[0]
        region_type = rule.entity_types[1] if len(rule.entity_types) == 2 else None
        minimum = rule.parameters.get("minimum_count", 1)
        try:
            minimum_integer = int(minimum)
            valid = minimum_integer >= 0 and minimum_integer == float(minimum)
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            return ViolationResult.unknown(
                rule.rule_id, "invalid_minimum_count",
                map_size=context.map_size, grounding_strategy=self.strategy,
            )

        complete, entity_confidence = _presence_observation(context, entity)
        region = None
        region_confidence = 1.0
        if region_type is not None:
            region = _unique_region(context, region_type)
            if region is None:
                return ViolationResult.unknown(
                    rule.rule_id, "region_grounding_not_unique",
                    map_size=context.map_size, grounding_strategy=self.strategy,
                    details={"region_type": region_type},
                )
            region_confidence = float(np.clip(region.confidence, 0.0, 1.0))
        confidence = min(entity_confidence, region_confidence)
        if not complete or confidence < float(rule.parameters.get("minimum_confidence", 0.5)):
            return ViolationResult.unknown(
                rule.rule_id, "presence_observation_incomplete",
                evidence_confidence=confidence, map_size=context.map_size,
                grounding_strategy=self.strategy,
                details={"entity_type": entity, "region_type": region_type},
            )

        observed = context.scene.instances_of(entity)
        if region is None:
            inside = list(observed)
        else:
            inside = [row for row in observed if _centroid_inside(region, row)]
        actual = len(inside)
        details = {
            "entity_type": entity, "region_type": region_type,
            "minimum_count": minimum_integer, "actual": actual,
            "total_observed": len(observed),
        }
        if actual >= minimum_integer:
            return ViolationResult(
                rule.rule_id, RuleStatus.SATISFIED, 0.0, confidence,
                grounding_strategy=self.strategy,
                anomaly_map=np.zeros((context.map_size, context.map_size), dtype=np.float32),
                attribution_uncertainty=1.0 - confidence, details=details,
            )
        deficit = minimum_integer - actual
        score = float(np.clip(deficit / max(minimum_integer, 1), 0.0, 1.0))
        # An observed instance outside the region is a concrete "misplaced"
        # witness; with nothing observed at all only the region itself is known.
        misplaced = [row for row in observed if region is not None and row not in inside]
        witnesses = [str(row.instance_id) for row in misplaced[:deficit]]
        anomaly = np.zeros((context.map_size, context.map_size), dtype=np.float32)
        for instance in misplaced[:deficit]:
            anomaly = np.maximum(
                anomaly, resized_instance_mask(instance, context.map_size) * score
            )
        uncertainty = 1.0 - confidence
        if not witnesses:
            # Nothing observed anywhere: the only witness is the empty region,
            # and localisation inside it is explicitly uncertain.
            if region is not None:
                resized = np.asarray(region.mask, dtype=np.float32)
                if resized.shape != anomaly.shape:
                    import cv2
                    resized = cv2.resize(
                        resized, anomaly.shape[::-1], interpolation=cv2.INTER_NEAREST
                    )
                anomaly = np.maximum(anomaly, resized * score * 0.3)
            uncertainty = max(uncertainty, 0.8)
        return ViolationResult(
            rule.rule_id, RuleStatus.VIOLATED, score, confidence,
            witness_instances=witnesses, grounding_strategy=self.strategy,
            anomaly_map=anomaly, attribution_uncertainty=uncertainty,
            details=details,
        )


class _PairRelationExecutor:
    """Shared evidence gate for two-entity spatial relations."""

    strategy = "relation_centroid"
    relation_name = "relation"

    def _resolve(self, rule: RuleSpec, context: ExecutionContext):
        if len(rule.entity_types) != 2:
            return None
        source_type, target_type = rule.entity_types
        sources = context.scene.instances_of(source_type)
        targets = context.scene.instances_of(target_type)
        if not sources or not targets:
            return None
        source_confidence = context.verification_confidence(source_type)
        target_confidence = context.verification_confidence(target_type)
        confidence = min(source_confidence, target_confidence)
        if confidence < float(rule.parameters.get("minimum_confidence", 0.3)):
            return ("insufficient", confidence)
        if not context.scene.observation_complete:
            return ("incomplete", confidence)
        return (sources, targets, confidence)

    def _threshold(self, rule: RuleSpec, default: float = 0.05) -> float:
        try:
            value = float(rule.parameters.get("threshold", default))
        except (TypeError, ValueError, OverflowError):
            return default
        return value if np.isfinite(value) else default


class AboveExecutor(_PairRelationExecutor):
    """``above(source, target)``: source centroid is vertically above target."""

    relation_name = "above"

    def execute(self, rule: RuleSpec, context: ExecutionContext) -> ViolationResult:
        resolved = self._resolve(rule, context)
        if resolved is None:
            return ViolationResult.unknown(
                rule.rule_id, "no_grounded_instances",
                map_size=context.map_size, grounding_strategy=self.strategy,
            )
        if isinstance(resolved[0], str):
            return ViolationResult.unknown(
                rule.rule_id, "spatial_evidence_" + resolved[0],
                evidence_confidence=resolved[1], map_size=context.map_size,
                grounding_strategy=self.strategy,
            )
        sources, targets, confidence = resolved
        threshold = self._threshold(rule)
        _, source_y = _mean_centroid(sources)
        _, target_y = _mean_centroid(targets)
        # y grows downwards: "above" means the source has the smaller y.
        margin = target_y - source_y
        details = {
            "source": list(rule.entity_types[:1]), "target": list(rule.entity_types[1:]),
            "source_mean_y": source_y, "target_mean_y": target_y,
            "margin": margin, "threshold": threshold,
        }
        if margin >= threshold:
            return ViolationResult(
                rule.rule_id, RuleStatus.SATISFIED, 0.0, confidence,
                grounding_strategy=self.strategy,
                anomaly_map=np.zeros((context.map_size, context.map_size), dtype=np.float32),
                attribution_uncertainty=1.0 - confidence, details=details,
            )
        score = float(np.clip((threshold - margin) / max(abs(threshold), 1e-6), 0.0, 1.0))
        anomaly = np.zeros((context.map_size, context.map_size), dtype=np.float32)
        for instance in list(sources) + list(targets):
            anomaly = np.maximum(
                anomaly, resized_instance_mask(instance, context.map_size)
            )
        return ViolationResult(
            rule.rule_id, RuleStatus.VIOLATED, score, confidence,
            witness_instances=[str(row.instance_id) for row in list(sources) + list(targets)],
            grounding_strategy=self.strategy, anomaly_map=anomaly * score,
            attribution_uncertainty=1.0 - confidence, details=details,
        )


class LeftOfExecutor(_PairRelationExecutor):
    """``left_of(source, target)``: source centroid is horizontally left."""

    relation_name = "left_of"

    def execute(self, rule: RuleSpec, context: ExecutionContext) -> ViolationResult:
        resolved = self._resolve(rule, context)
        if resolved is None:
            return ViolationResult.unknown(
                rule.rule_id, "no_grounded_instances",
                map_size=context.map_size, grounding_strategy=self.strategy,
            )
        if isinstance(resolved[0], str):
            return ViolationResult.unknown(
                rule.rule_id, "spatial_evidence_" + resolved[0],
                evidence_confidence=resolved[1], map_size=context.map_size,
                grounding_strategy=self.strategy,
            )
        sources, targets, confidence = resolved
        threshold = self._threshold(rule)
        source_x, _ = _mean_centroid(sources)
        target_x, _ = _mean_centroid(targets)
        margin = target_x - source_x
        details = {
            "source_mean_x": source_x, "target_mean_x": target_x,
            "margin": margin, "threshold": threshold,
        }
        if margin >= threshold:
            return ViolationResult(
                rule.rule_id, RuleStatus.SATISFIED, 0.0, confidence,
                grounding_strategy=self.strategy,
                anomaly_map=np.zeros((context.map_size, context.map_size), dtype=np.float32),
                attribution_uncertainty=1.0 - confidence, details=details,
            )
        score = float(np.clip((threshold - margin) / max(abs(threshold), 1e-6), 0.0, 1.0))
        anomaly = np.zeros((context.map_size, context.map_size), dtype=np.float32)
        for instance in list(sources) + list(targets):
            anomaly = np.maximum(
                anomaly, resized_instance_mask(instance, context.map_size)
            )
        return ViolationResult(
            rule.rule_id, RuleStatus.VIOLATED, score, confidence,
            witness_instances=[str(row.instance_id) for row in list(sources) + list(targets)],
            grounding_strategy=self.strategy, anomaly_map=anomaly * score,
            attribution_uncertainty=1.0 - confidence, details=details,
        )


class AlignedVerticalExecutor(_PairRelationExecutor):
    """``aligned_vertical(a[, b])``: instance x-centroids share one column.

    With one entity type all its instances must be in one column; with two
    types the type means must share a column.  The spread is normalised by
    image width, so ``threshold`` is directly comparable across images.
    """

    relation_name = "aligned_vertical"

    def execute(self, rule: RuleSpec, context: ExecutionContext) -> ViolationResult:
        if len(rule.entity_types) not in {1, 2}:
            return ViolationResult.unknown(
                rule.rule_id, "aligned_vertical_requires_one_or_two_entities",
                map_size=context.map_size, grounding_strategy=self.strategy,
            )
        confidence = context.verification_confidence(rule.entity_types[0])
        if len(rule.entity_types) == 2:
            confidence = min(
                confidence,
                context.verification_confidence(rule.entity_types[1]),
            )
        if not context.scene.observation_complete or confidence < float(
            rule.parameters.get("minimum_confidence", 0.3)
        ):
            return ViolationResult.unknown(
                rule.rule_id, "spatial_evidence_insufficient",
                evidence_confidence=confidence, map_size=context.map_size,
                grounding_strategy=self.strategy,
            )
        if len(rule.entity_types) == 1:
            first = context.scene.instances_of(rule.entity_types[0])
            second = []
        else:
            first = context.scene.instances_of(rule.entity_types[0])
            second = context.scene.instances_of(rule.entity_types[1])
        instances = list(first) + list(second)
        if len(instances) < 2:
            return ViolationResult.unknown(
                rule.rule_id, "need_two_instances",
                evidence_confidence=confidence, map_size=context.map_size,
                grounding_strategy=self.strategy,
            )
        threshold = self._threshold(rule, default=0.1)
        xs = [float(row.centroid[0]) for row in instances]
        spread = max(xs) - min(xs)
        details = {
            "entity_types": list(rule.entity_types), "x_spread": spread,
            "threshold": threshold,
        }
        if spread <= threshold:
            return ViolationResult(
                rule.rule_id, RuleStatus.SATISFIED, 0.0, confidence,
                grounding_strategy=self.strategy,
                anomaly_map=np.zeros((context.map_size, context.map_size), dtype=np.float32),
                attribution_uncertainty=1.0 - confidence, details=details,
            )
        score = float(np.clip((spread - threshold) / max(abs(threshold), 1e-6), 0.0, 1.0))
        anomaly = np.zeros((context.map_size, context.map_size), dtype=np.float32)
        for instance in instances:
            anomaly = np.maximum(
                anomaly, resized_instance_mask(instance, context.map_size)
            )
        return ViolationResult(
            rule.rule_id, RuleStatus.VIOLATED, score, confidence,
            witness_instances=[str(row.instance_id) for row in instances],
            grounding_strategy=self.strategy, anomaly_map=anomaly * score,
            attribution_uncertainty=1.0 - confidence, details=details,
        )


class DisjointExecutor(_PairRelationExecutor):
    """``disjoint(a, b)``: no observed instance pair overlaps beyond tolerance."""

    relation_name = "disjoint"

    def execute(self, rule: RuleSpec, context: ExecutionContext) -> ViolationResult:
        resolved = self._resolve(rule, context)
        if resolved is None:
            return ViolationResult.unknown(
                rule.rule_id, "no_grounded_instances",
                map_size=context.map_size, grounding_strategy=self.strategy,
            )
        if isinstance(resolved[0], str):
            return ViolationResult.unknown(
                rule.rule_id, "spatial_evidence_" + resolved[0],
                evidence_confidence=resolved[1], map_size=context.map_size,
                grounding_strategy=self.strategy,
            )
        sources, targets, confidence = resolved
        threshold = self._threshold(rule, default=0.01)
        worst = 0.0
        worst_pair = None
        for left in sources:
            for right in targets:
                left_mask = np.asarray(left.mask, dtype=bool)
                right_mask = np.asarray(right.mask, dtype=bool)
                intersection = np.logical_and(left_mask, right_mask).sum()
                union = np.logical_or(left_mask, right_mask).sum()
                iou = float(intersection) / float(union) if union else 0.0
                if iou > worst:
                    worst = iou
                    worst_pair = (left, right)
        details = {
            "max_intersection_over_union": worst, "tolerance": threshold,
        }
        if worst <= threshold:
            return ViolationResult(
                rule.rule_id, RuleStatus.SATISFIED, 0.0, confidence,
                grounding_strategy=self.strategy,
                anomaly_map=np.zeros((context.map_size, context.map_size), dtype=np.float32),
                attribution_uncertainty=1.0 - confidence, details=details,
            )
        score = float(np.clip((worst - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0))
        anomaly = np.zeros((context.map_size, context.map_size), dtype=np.float32)
        for instance in worst_pair:
            anomaly = np.maximum(
                anomaly, resized_instance_mask(instance, context.map_size)
            )
        return ViolationResult(
            rule.rule_id, RuleStatus.VIOLATED, score, confidence,
            witness_instances=[str(row.instance_id) for row in worst_pair],
            grounding_strategy=self.strategy, anomaly_map=anomaly * score,
            attribution_uncertainty=1.0 - confidence, details=details,
        )


__all__ = [
    "ExecutionContext", "LengthRangeExecutor", "AspectRangeExecutor",
    "ColorRangeExecutor", "CountEqExecutor", "CountInExecutor",
    "ExistsExecutor", "AboveExecutor", "LeftOfExecutor",
    "AlignedVerticalExecutor", "DisjointExecutor",
]
