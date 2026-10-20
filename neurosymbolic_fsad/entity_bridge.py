"""V13 entity-token bridge and bidirectional correction for neuro-symbolic FSAD."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np

from .orthogonal_symbolic import GenericLogicInference


def _roi_token(feature_grids: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """Pool mean and standard deviation from every INP feature grid."""
    grids = np.asarray(feature_grids, dtype=np.float32)
    if grids.ndim != 4:
        raise ValueError("feature_grids must be [L,C,H,W]")
    _, _, height, width = grids.shape
    x0, y0, x1, y1 = np.clip(np.asarray(box, dtype=float), 0.0, 1.0)
    left = min(width - 1, max(0, int(math.floor(x0 * width))))
    top = min(height - 1, max(0, int(math.floor(y0 * height))))
    right = min(width, max(left + 1, int(math.ceil(x1 * width))))
    bottom = min(height, max(top + 1, int(math.ceil(y1 * height))))
    region = grids[:, :, top:bottom, left:right]
    mean = region.mean(axis=(-2, -1))
    std = region.std(axis=(-2, -1))
    token = np.concatenate([mean.reshape(-1), std.reshape(-1)]).astype(np.float32)
    norm = max(float(np.linalg.norm(token)), 1e-8)
    return token / norm


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-8)
    return float(np.clip(1.0 - float(np.dot(left, right)) / denominator, 0.0, 2.0))


def _box_from_center_size(center, size) -> list:
    cx, cy = [float(value) for value in center]
    width, height = [float(value) for value in size]
    return [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]


def _box_mask(box: Sequence[float], image_size: int) -> np.ndarray:
    result = np.zeros((image_size, image_size), dtype=np.float32)
    x0, y0, x1, y1 = np.clip(np.asarray(box, dtype=float), 0.0, 1.0)
    left, top = int(x0 * image_size), int(y0 * image_size)
    right = max(left + 1, int(x1 * image_size))
    bottom = max(top + 1, int(y1 * image_size))
    result[top:bottom, left:right] = 1.0
    return result


def _top_mean(values: np.ndarray, fraction: float = 0.01) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    count = max(1, int(math.ceil(flat.size * fraction)))
    return float(np.partition(flat, flat.size - count)[-count:].mean())


@dataclass
class EntityBridgeInference:
    score: float
    anomaly_map: np.ndarray
    raw_symbolic: GenericLogicInference
    diagnostics: Dict[str, Any]


class EntityTokenBridge:
    """Bridge GroundingDINO slots to INP features and correct both directions."""

    def __init__(self, engine, state: Mapping[str, Any]):
        self.engine = engine
        self.state = dict(state)
        self.prototypes = np.asarray(state["slot_token_prototypes"], dtype=np.float32)
        self.thresholds = np.asarray(state["slot_distance_thresholds"], dtype=np.float32)
        self.scales = np.asarray(state["slot_distance_scales"], dtype=np.float32)
        self.support_score_upper = float(state["support_score_upper"])
        self.entity_set_threshold = float(state["entity_set_distance_threshold"])
        self.entity_set_scale = float(state["entity_set_distance_scale"])
        self.rule_reliability = {
            str(key): float(value)
            for key, value in state["rule_reliability"].items()
        }
        self.map_size = int(state["map_size"])
        self.rescue_gain = float(state.get("rescue_gain", 0.35))
        self.suppression_gain = float(state.get("suppression_gain", 0.20))

    @classmethod
    def fit(
        cls, engine, support_images: Sequence[Any], support_feature_grids: np.ndarray,
        support_maps: np.ndarray, support_scores: np.ndarray,
    ) -> "EntityTokenBridge":
        if engine.grounder is None:
            raise ValueError("entity bridge requires a fitted open-vocabulary grounder")
        grids = np.asarray(support_feature_grids, dtype=np.float32)
        if len(grids) != len(support_images):
            raise ValueError("support feature grids are not aligned")
        expected = int(engine.grounder.expected_count)
        if expected < 2:
            raise ValueError(
                "entity-set bridge requires at least two repeated normal entities"
            )
        slot_rows = [[] for _ in range(expected)]
        entity_sets = []
        support_results = []
        for image, feature in zip(support_images, grids):
            result = engine.evaluate(image)
            support_results.append(result)
            detections = list(result.diagnostics.get("detections", []))
            if len(detections) != expected:
                continue
            current_tokens = []
            for slot, detection in enumerate(detections):
                token = _roi_token(feature, detection["box"])
                slot_rows[slot].append(token)
                current_tokens.append(token)
            entity_sets.append(current_tokens)
        if any(len(row) < 3 for row in slot_rows):
            raise ValueError("fewer than three grounded normal tokens for an entity slot")
        prototypes = np.stack([
            np.mean(np.stack(row), axis=0) for row in slot_rows
        ]).astype(np.float32)
        prototypes /= np.maximum(
            np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8
        )
        thresholds, scales = [], []
        for row in slot_rows:
            leave_one_out = []
            for index, token in enumerate(row):
                others = [value for j, value in enumerate(row) if j != index]
                prototype = np.mean(np.stack(others), axis=0)
                leave_one_out.append(_cosine_distance(token, prototype))
            values = np.asarray(leave_one_out, dtype=float)
            median = float(np.median(values))
            mad = 1.4826 * float(np.median(np.abs(values - median)))
            thresholds.append(float(np.max(values) + max(mad, 0.005)))
            scales.append(max(mad, float(np.std(values)), 0.01))

        # Permutation-invariant within-image calibration. At query time this
        # distinguishes an actually missing entity from a detector miss by
        # comparing the expected-slot token with the other entities in the
        # same image, avoiding cross-image domain shift and slot swaps.
        set_distances = []
        for tokens in entity_sets:
            for index, token in enumerate(tokens):
                others = [value for j, value in enumerate(tokens) if j != index]
                if others:
                    set_distances.append(_cosine_distance(
                        token, np.mean(np.stack(others), axis=0)
                    ))
        set_values = np.asarray(set_distances, dtype=float)
        set_median = float(np.median(set_values))
        set_mad = 1.4826 * float(np.median(np.abs(set_values - set_median)))

        augmentation_reliability = float(
            engine.grounder.state.get("support_augmentation_match_rate", 0.0)
        )
        reliability = {}
        for rule in engine.rules:
            name = str(rule["id"])
            if str(rule["fact"]) == "count_error":
                reliability[name] = augmentation_reliability
            else:
                violations = [
                    float(result.atoms[name]["violation"])
                    for result in support_results
                ]
                reliability[name] = float(np.clip(
                    1.0 - np.mean(np.asarray(violations) > 0.0), 0.25, 1.0
                ))
        score_values = np.asarray(support_scores, dtype=float)
        state = {
            "schema_version": "13.0-entity-token-bidirectional-bridge",
            "slot_token_prototypes": prototypes.tolist(),
            "slot_distance_thresholds": thresholds,
            "slot_distance_scales": scales,
            "entity_set_distance_threshold": float(
                np.max(set_values) + max(set_mad, 0.005)
            ),
            "entity_set_distance_scale": max(
                set_mad, float(np.std(set_values)), 0.01
            ),
            "support_score_upper": float(
                np.max(score_values) + max(float(np.std(score_values)), 0.005)
            ),
            "rule_reliability": reliability,
            "map_size": int(np.asarray(support_maps).shape[-1]),
            # Conservative frozen gains selected on the two development
            # categories; all remaining categories must use these unchanged.
            "rescue_gain": 0.035,
            "suppression_gain": 0.020,
            "query_or_test_labels_used": False,
            "training_signal": "normal_support_plus_feature_counterfactual_margin",
        }
        return cls(engine, state)

    def state_dict(self) -> Dict[str, Any]:
        return dict(self.state)

    def _slot_boxes(self, symbolic: GenericLogicInference) -> Tuple[list, set, set]:
        grounder = self.engine.grounder
        detections = list(symbolic.diagnostics.get("detections", []))
        expected = int(grounder.expected_count)
        boxes = [
            _box_from_center_size(center, size)
            for center, size in zip(
                grounder.prototype["absolute_center"],
                grounder.prototype["absolute_size"],
            )
        ]
        missing = set(symbolic.diagnostics.get("unmatched_slots", []))
        extras = set(symbolic.diagnostics.get("unmatched_detections", []))
        if len(detections) == expected:
            boxes = [row["box"] for row in detections]
            missing = set()
            extras = set()
        else:
            for slot, detection_index in symbolic.diagnostics.get("assignments", []):
                boxes[int(slot)] = detections[int(detection_index)]["box"]
        return boxes, missing, extras

    def infer(
        self, image: Any, feature_grids: np.ndarray, baseline_map: np.ndarray,
        baseline_score: float,
    ) -> EntityBridgeInference:
        symbolic = self.engine.evaluate(image)
        baseline = np.asarray(baseline_map, dtype=np.float32)
        boxes, missing, extras = self._slot_boxes(symbolic)
        tokens = np.stack([_roi_token(feature_grids, box) for box in boxes])
        distances = np.asarray([
            _cosine_distance(token, prototype)
            for token, prototype in zip(tokens, self.prototypes)
        ], dtype=np.float32)
        # Slotwise distances are diagnostic only. Count correction uses the
        # permutation-invariant relation to entities from the same query.
        token_violations = np.zeros_like(distances)
        present_indices = [index for index in range(len(boxes)) if index not in missing]
        if present_indices:
            present_reference = np.mean(tokens[present_indices], axis=0)
            for index in missing:
                relation_distance = _cosine_distance(
                    tokens[index], present_reference
                )
                token_violations[index] = math.tanh(max(
                    0.0, relation_distance - self.entity_set_threshold
                ) / max(self.entity_set_scale, 1e-8))

        corrected_truths, corrected_maps, corrected_rows = {}, {}, {}
        count_missing_strength = float(max(
            [token_violations[index] for index in missing], default=0.0
        ))
        count_localizer = None
        for name, atom in symbolic.atoms.items():
            raw_violation = float(atom["violation"])
            reliability = self.rule_reliability.get(name, 0.5)
            if str(atom["fact"]) == "count_error":
                localizer = np.asarray(
                    symbolic.localization_maps[name], dtype=np.float32
                )
                count_localizer = localizer
                selected = localizer > 0.05
                local_values = baseline[selected] if np.any(selected) else np.asarray([])
                local_top = _top_mean(local_values, 0.10) if local_values.size else 0.0
                neural_gate = float(
                    local_top > float(np.quantile(baseline, 0.995))
                )
                # Missing entities can be rescued by the shared INP entity
                # token even if the scalar baseline is low. Extra detections
                # remain detector-ambiguous and require a local INP hotspot.
                bridge_support = max(
                    count_missing_strength,
                    neural_gate if extras else 0.0,
                )
            else:
                # A normal count/geometry rule does not imply that an object
                # cannot have a surface defect. Do not use unrelated relation
                # predicates to override the baseline without grounded token
                # evidence for that specific predicate.
                bridge_support = 0.0
            corrected_violation = float(np.clip(
                raw_violation * reliability * bridge_support, 0.0, 1.0
            ))
            localizer = np.asarray(
                symbolic.localization_maps[name], dtype=np.float32
            )
            corrected_truths[name] = 1.0 - corrected_violation
            corrected_maps[name] = 1.0 - corrected_violation * localizer
            corrected_rows[name] = {
                "raw_violation": raw_violation,
                "reliability": reliability,
                "entity_feature_support": bridge_support,
                "corrected_violation": corrected_violation,
            }
        corrected_anomaly, logic_trace = self.engine.program.evaluate(corrected_truths)
        corrected_map, _ = self.engine.program.evaluate(corrected_maps)
        corrected_map = np.asarray(corrected_map, dtype=np.float32)

        # Rescue does not require an existing INP hotspot: reliable symbolic
        # violation plus abnormal entity-token evidence can add a new residual.
        rescue = np.clip(
            float(corrected_anomaly) * corrected_map * self.rescue_gain,
            0.0, 1.0,
        )
        # Negative correction is scoped to an apparent missing slot that the
        # INP entity token identifies as normal. A normal cardinality rule must
        # never suppress unrelated surface anomalies elsewhere in the object.
        false_missing_confidence = (
            (1.0 - count_missing_strength)
            if missing and not extras else 0.0
        )
        excess = np.maximum(baseline - self.support_score_upper, 0.0)
        suppression = (
            self.suppression_gain * false_missing_confidence
            * (count_localizer if count_localizer is not None else 0.0) * excess
        )
        fused = np.clip(baseline + rescue - suppression, 0.0, 1.0)
        # Preserve the exact baseline image score unless correction adds
        # positive evidence. Pixel-level negative correction remains available
        # for detector-error regions without globally lowering an anomaly.
        score = max(float(baseline_score), _top_mean(fused, fraction=0.01))
        return EntityBridgeInference(
            score=score,
            anomaly_map=fused.astype(np.float32),
            raw_symbolic=symbolic,
            diagnostics={
                "slot_boxes": boxes,
                "missing_slots": sorted(missing),
                "extra_detections": sorted(extras),
                "slot_token_distances": distances.tolist(),
                "slot_token_violations": token_violations.tolist(),
                "corrected_rules": corrected_rows,
                "corrected_symbolic_anomaly": float(corrected_anomaly),
                "logic_trace": logic_trace,
                "false_missing_confidence": float(false_missing_confidence),
                "rescue_max": float(np.max(rescue)),
                "suppression_max": float(np.max(suppression)),
                "baseline_score_input": float(baseline_score),
            },
        )


__all__ = ["EntityBridgeInference", "EntityTokenBridge", "_roi_token"]
