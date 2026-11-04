"""Evaluate relation atoms against a query scene graph.

Every atom returns a truth value *and* a responsibility map on the CLIP patch
grid.  The map is a runtime diagnostic, not part of the atom syntax.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from symbolicAD.config import Config
from symbolicAD.models.clip_backbone import CLIPBackbone
from symbolicAD.models.text_embeddings import encode_prompts_cached
from symbolicAD.rules.executor import AtomicResult
from symbolicAD.rules.schema import RelationAtom, RuleKB

from .geometry import bbox_iou, containment, mahalanobis, pair_feature, pair_geometry
from .instance_extractor import lab_color_similarity, named_color_lab
from .scene_graph import ComponentInstance, SceneGraph
from .support_memory import SupportMemory


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _union_masks(
    instances: Sequence[ComponentInstance],
    grid_size: tuple[int, int],
) -> np.ndarray:
    result = np.zeros(grid_size, dtype=np.float32)
    for instance in instances:
        result = np.maximum(result, instance.mask.astype(np.float32))
    return result


def _weighted_instance_masks(
    instances: Sequence[ComponentInstance],
    component_map: Optional[np.ndarray],
    weights: Sequence[float],
    grid_size: tuple[int, int],
) -> np.ndarray:
    """Soft responsibility: instance support times a continuous violation weight."""
    result = np.zeros(grid_size, dtype=np.float32)
    for instance, weight in zip(instances, weights):
        soft = np.asarray(instance.mask, dtype=np.float32)
        if component_map is not None:
            support = np.asarray(component_map, dtype=np.float32)
            if support.shape == grid_size:
                soft = soft * (0.25 + 0.75 * np.clip(support, 0.0, 1.0))
        result = np.maximum(result, np.clip(soft * float(weight), 0.0, 1.0))
    return result


def calibrate_atom_results(
    atom_results: Dict[str, AtomicResult],
    support_memory: Optional[SupportMemory],
    config: Config,
) -> Dict[str, AtomicResult]:
    """Convert atom truths into support-relative normalities.

    A raw truth of 0.7 is only a violation if normal support images all scored
    near 1.0 for the same atom.  This prevents badly grounded LLM rules from
    flooding every query with false anomalies.
    """
    if support_memory is None or not config.logic.support_calibration:
        return atom_results
    for atom_id, result in atom_results.items():
        if not result.valid or result.weight <= 0:
            continue
        if result.kind == "AttributeMatch":
            # Attribute atoms already normalise against q05/q50 support scores.
            continue
        stats = support_memory.atom_stats.get(atom_id)
        if not stats:
            continue
        raw = float(result.truth)
        low = float(stats.get("q05", 0.0))
        high = float(stats.get("q50", 1.0))
        gap = high - low
        if gap < 1e-3:
            # Deterministic atom on support (e.g. CountEq always true).  Keep
            # the raw truth; mapping a degenerate gap to zero would erase the
            # only violation signal these atoms can carry.
            calibrated = raw
        elif raw >= high:
            calibrated = 1.0
        elif raw <= low:
            calibrated = 0.0
        else:
            width = max(
                gap,
                float(getattr(config.logic, "calibration_min_gap", 0.10)),
            )
            calibrated = (raw - low) / max(width, 1e-6)
        result.evidence = dict(result.evidence)
        result.evidence["raw_truth"] = raw
        result.evidence["support_q05"] = low
        result.evidence["support_q50"] = high
        result.truth = _clip01(calibrated)
        result.violation = float(1.0 - result.truth)
    return atom_results


def _expected_mask(
    support_memory: Optional[SupportMemory],
    component: str,
    grid_size: tuple[int, int],
) -> np.ndarray:
    if support_memory is not None and support_memory.has_component(component):
        memory = support_memory.components[component]
        if memory.expected_mask.shape == grid_size:
            return memory.expected_mask.astype(np.float32)
    return np.zeros(grid_size, dtype=np.float32)


def _slot_mask(
    support_memory: Optional[SupportMemory],
    component: str,
    grid_size: tuple[int, int],
) -> np.ndarray:
    if support_memory is not None and support_memory.has_component(component):
        memory = support_memory.components[component]
        if memory.slot_masks.size and memory.slot_masks.shape[1:] == grid_size:
            return np.clip(memory.slot_masks.max(axis=0), 0.0, 1.0).astype(np.float32)
        return memory.expected_mask.astype(np.float32)
    return np.zeros(grid_size, dtype=np.float32)


def _component_map(scene_graph: SceneGraph, component: str) -> np.ndarray:
    value = scene_graph.component_map(component)
    if value is None:
        grid_size = scene_graph.grid_size()
        return np.zeros(grid_size, dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _gaussian_truth(value: float, target: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-6)
    return float(math.exp(-((float(value) - float(target)) ** 2) / (2.0 * sigma**2)))


def _heuristic_relation(relation: str, instance_a, instance_b) -> float:
    dx = float(instance_b.center[0] - instance_a.center[0])
    dy = float(instance_b.center[1] - instance_a.center[1])
    if relation == "Inside":
        return _clip01(containment(instance_a, instance_b))
    if relation in {"LeftOf", "RightOf"}:
        signed = dx if relation == "LeftOf" else -dx
        return _clip01(0.5 + 2.0 * signed)
    if relation in {"Above", "Below"}:
        # y grows downwards: Above(A,B) means A.cy < B.cy, i.e. dy > 0.
        signed = dy if relation == "Above" else -dy
        return _clip01(0.5 + 2.0 * signed)
    if relation == "CenterAligned":
        return _clip01(1.0 - abs(dx) / 0.25)
    return 0.0


def _evaluate_exist(
    atom: RelationAtom,
    scene_graph: SceneGraph,
    support_memory: Optional[SupportMemory],
    config: Config,
) -> AtomicResult:
    component = atom.components[0]
    instances = scene_graph.instances_for(component)
    grid_size = scene_graph.grid_size()
    if not instances:
        truth = 0.0
        responsibility = _expected_mask(support_memory, component, grid_size)
        evidence = {"count": 0, "missing": True, "localized": bool(responsibility.any())}
    else:
        max_confidence = float(max(instance.confidence for instance in instances))
        truth = max_confidence
        if support_memory is not None and support_memory.has_component(component):
            memory = support_memory.components[component]
            low = float(memory.exist_q05)
            high = float(memory.exist_q50)
            if high > low + 1e-6:
                truth = _clip01((max_confidence - low) / (high - low))
        responsibility = _component_map(scene_graph, component)
        evidence = {
            "count": len(instances),
            "missing": False,
            "max_confidence": max_confidence,
            "localized": True,
        }
    return _result(atom, truth, responsibility, evidence)


def _evaluate_count_eq(
    atom: RelationAtom,
    scene_graph: SceneGraph,
    support_memory: Optional[SupportMemory],
    config: Config,
) -> AtomicResult:
    component = atom.components[0]
    target = float(atom.args[1])
    instances = scene_graph.instances_for(component)
    grid_size = scene_graph.grid_size()
    estimated = float(len(instances))
    soft_count = float(sum(instance.confidence for instance in instances))
    sigma = max(0.5, float(config.logic.count_sigma) * max(1.0, 0.25 * target))
    truth = _gaussian_truth(estimated, target, sigma)

    responsibility = np.zeros(grid_size, dtype=np.float32)
    if estimated > target + 0.25 and instances:
        ordered = sorted(instances, key=lambda item: item.confidence, reverse=True)
        extra = ordered[max(int(round(target)), 0):]
        if not extra:
            extra = ordered[-1:]
        if config.logic.soft_responsibility:
            responsibility = _weighted_instance_masks(
                extra,
                _component_map(scene_graph, component),
                [1.0] * len(extra),
                grid_size,
            )
        else:
            responsibility = _union_masks(extra, grid_size)
        evidence = {
            "estimated_count": estimated,
            "soft_count": soft_count,
            "target": target,
            "mode": "excess",
            "missing": False,
            "extra_instances": [item.instance_id for item in extra],
        }
    elif estimated < target - 0.25:
        covered = _union_masks(instances, grid_size)
        expected = _expected_mask(support_memory, component, grid_size)
        responsibility = np.clip(expected - covered, 0.0, 1.0)
        if not responsibility.any():
            responsibility = _slot_mask(support_memory, component, grid_size)
        evidence = {
            "estimated_count": estimated,
            "soft_count": soft_count,
            "target": target,
            "mode": "missing",
            "missing": True,
            "missing_expected_mass": float(responsibility.sum()),
        }
    else:
        evidence = {
            "estimated_count": estimated,
            "soft_count": soft_count,
            "target": target,
            "mode": "ok",
        }
    return _result(atom, truth, responsibility, evidence)


def _evaluate_count_in_range(
    atom: RelationAtom,
    scene_graph: SceneGraph,
    support_memory: Optional[SupportMemory],
    config: Config,
) -> AtomicResult:
    component = atom.components[0]
    low = float(atom.args[1])
    high = float(atom.args[2])
    instances = scene_graph.instances_for(component)
    count = float(len(instances))
    grid_size = scene_graph.grid_size()
    sigma = max(
        0.5,
        float(config.logic.count_sigma) * max(1.0, 0.25 * (high - low + 1.0)),
    )

    responsibility = np.zeros(grid_size, dtype=np.float32)
    if low <= count <= high:
        truth = 1.0
        evidence = {
            "count": count,
            "low": low,
            "high": high,
            "mode": "ok",
            "missing": False,
        }
    elif count < low:
        truth = _gaussian_truth(count, low, sigma)
        covered = _union_masks(instances, grid_size)
        expected = _expected_mask(support_memory, component, grid_size)
        responsibility = np.clip(expected - covered, 0.0, 1.0)
        if not responsibility.any():
            responsibility = _slot_mask(support_memory, component, grid_size)
        evidence = {
            "count": count,
            "low": low,
            "high": high,
            "mode": "missing",
            "missing": True,
            "missing_expected_mass": float(responsibility.sum()),
        }
    else:
        truth = _gaussian_truth(count, high, sigma)
        ordered = sorted(instances, key=lambda item: item.confidence, reverse=True)
        extra = ordered[int(round(high)):]
        if not extra:
            extra = ordered[-1:]
        if config.logic.soft_responsibility:
            responsibility = _weighted_instance_masks(
                extra,
                _component_map(scene_graph, component),
                [1.0] * len(extra),
                grid_size,
            )
        else:
            responsibility = _union_masks(extra, grid_size)
        evidence = {
            "count": count,
            "low": low,
            "high": high,
            "mode": "excess",
            "missing": False,
            "extra_instances": [item.instance_id for item in extra],
        }
    return _result(atom, truth, responsibility, evidence)


def _evaluate_attribute_match(
    atom: RelationAtom,
    scene_graph: SceneGraph,
    support_memory: Optional[SupportMemory],
    config: Config,
    backbone: Optional[CLIPBackbone],
) -> AtomicResult:
    component = atom.components[0]
    attribute = atom.attribute or ""
    value = atom.attribute_value or ""
    instances = scene_graph.instances_for(component)
    grid_size = scene_graph.grid_size()

    if not instances:
        responsibility = _expected_mask(support_memory, component, grid_size)
        return _result(
            atom,
            0.0,
            responsibility,
            {
                "attribute": attribute,
                "value": value,
                "count": 0,
                "missing": True,
                "localized": bool(responsibility.any()),
            },
        )

    color_target = (
        named_color_lab(value)
        if attribute.lower() in {"color", "colour"}
        else None
    )
    if color_target is not None:
        per_instance = np.asarray(
            [
                lab_color_similarity(
                    item.color_feature,
                    color_target,
                    sigma=float(config.logic.color_sigma),
                )
                for item in instances
            ],
            dtype=np.float32,
        )
    else:
        if backbone is None:
            return _result(
                atom,
                1.0,
                np.zeros(grid_size, dtype=np.float32),
                {
                    "attribute": attribute,
                    "value": value,
                    "unsupported_backbone": True,
                },
            )
        embeddings = encode_prompts_cached(
            backbone,
            atom.attribute_prompts,
            cache_dir=config.text_embedding_dir,
            enabled=config.grounding.text_embedding_cache,
        )
        features = np.stack([item.feature for item in instances], axis=0)
        similarities = features @ embeddings.T
        per_instance = similarities.max(axis=1)
    score = float(per_instance.max())

    stats: Dict[str, float] = {}
    if support_memory is not None:
        stats = support_memory.attribute_stats.get(atom.atom_id, {})
    q05 = stats.get("q05")
    q50 = stats.get("q50")
    if q05 is not None and q50 is not None and float(q50) > float(q05) + 1e-6:
        if score >= float(q50):
            truth = 1.0
        elif score <= float(q05):
            truth = 0.0
        else:
            width = max(
                float(q50) - float(q05),
                float(getattr(config.logic, "attribute_min_gap", 0.05)),
            )
            truth = _clip01((score - float(q05)) / width)
        threshold = float(q05)
    else:
        truth = _clip01(1.0 / (1.0 + math.exp(-(score - 0.25) / 0.05)))
        threshold = 0.2

    failing = [
        (instance, similarity)
        for instance, similarity in zip(instances, per_instance)
        if float(similarity) < threshold
    ]
    if failing and config.logic.soft_responsibility:
        severity = [
            1.0 - min(1.0, float(similarity) / max(float(threshold), 1e-6))
            for _, similarity in failing
        ]
        responsibility = _weighted_instance_masks(
            [instance for instance, _ in failing],
            _component_map(scene_graph, component),
            severity,
            grid_size,
        )
    else:
        responsibility = _union_masks(
            [instance for instance, _ in failing], grid_size
        )
    if not responsibility.any() and truth < config.logic.fail_threshold:
        responsibility = _component_map(scene_graph, component)

    return _result(
        atom,
        truth,
        responsibility,
        {
            "attribute": attribute,
            "value": value,
            "score": score,
            "per_instance_scores": [float(item) for item in per_instance],
            "q05": q05,
            "q50": q50,
            "grounding": "lab_color" if color_target is not None else "clip_attribute",
            "missing": False,
        },
    )


def _evaluate_position_match(
    atom: RelationAtom,
    scene_graph: SceneGraph,
    support_memory: Optional[SupportMemory],
    config: Config,
) -> AtomicResult:
    component = atom.components[0]
    region = atom.position_region or "normal"
    instances = scene_graph.instances_for(component)
    grid_size = scene_graph.grid_size()
    expected = _expected_mask(support_memory, component, grid_size)
    if not instances:
        return _result(
            atom,
            0.0,
            expected,
            {
                "region": region,
                "count": 0,
                "missing": True,
                "localized": bool(expected.any()),
            },
        )

    scale = max(float(expected.max()), 1e-6)
    scores = []
    for instance in instances:
        mask = np.asarray(instance.mask, dtype=np.float32)
        denominator = max(float(mask.sum()), 1e-6)
        scores.append(_clip01(float((expected * mask).sum()) / denominator / scale))
    truth = float(np.mean(scores)) if scores else 0.0
    failing = [
        (instance, score)
        for instance, score in zip(instances, scores)
        if score < config.logic.fail_threshold
    ]
    if failing and config.logic.soft_responsibility:
        severity = [
            1.0 - min(
                1.0, float(score) / max(float(config.logic.fail_threshold), 1e-6)
            )
            for _, score in failing
        ]
        responsibility = _weighted_instance_masks(
            [instance for instance, _ in failing],
            _component_map(scene_graph, component),
            severity,
            grid_size,
        )
    else:
        responsibility = _union_masks(
            [instance for instance, _ in failing], grid_size
        )
    if truth < config.logic.fail_threshold:
        covered = _union_masks(instances, grid_size)
        responsibility = np.maximum(
            responsibility,
            np.clip(expected - covered, 0.0, 1.0),
        )
    return _result(
        atom,
        truth,
        responsibility,
        {
            "region": region,
            "count": len(instances),
            "position_scores": scores,
            "missing": False,
            "expected_mass": float(expected.sum()),
        },
    )


def _evaluate_count_ratio(
    atom: RelationAtom,
    scene_graph: SceneGraph,
    support_memory: Optional[SupportMemory],
    config: Config,
) -> AtomicResult:
    component_a, component_b = atom.components[0], atom.components[1]
    target = float(atom.args[2])
    instances_a = scene_graph.instances_for(component_a)
    instances_b = scene_graph.instances_for(component_b)
    count_a = float(len(instances_a))
    count_b = float(len(instances_b))
    ratio = count_a / (count_b + 1e-6)
    truth = _gaussian_truth(ratio, target, config.logic.ratio_sigma)

    grid_size = scene_graph.grid_size()
    responsibility = np.zeros(grid_size, dtype=np.float32)
    if ratio > target:
        responsibility = _union_masks(instances_a, grid_size)
        mode = "a_excess"
    elif ratio < target:
        responsibility = _expected_mask(support_memory, component_a, grid_size)
        if not responsibility.any():
            responsibility = _union_masks(instances_b, grid_size)
        mode = "a_missing_or_b_excess"
    else:
        mode = "ok"
    return _result(
        atom,
        truth,
        responsibility,
        {
            "count_a": count_a,
            "count_b": count_b,
            "ratio": ratio,
            "target": target,
            "mode": mode,
            "missing": mode == "a_missing_or_b_excess",
        },
    )


def _one_to_one_cost(
    instances_a: Sequence[ComponentInstance],
    instances_b: Sequence[ComponentInstance],
    relation_memory,
    config: Config,
) -> np.ndarray:
    cost = np.zeros((len(instances_a), len(instances_b)), dtype=np.float64)
    lambda_feature = float(config.logic.lambda_feature)
    lambda_relation = float(config.logic.lambda_relation)
    if relation_memory is None or relation_memory.pair_prototype is None:
        lambda_feature, lambda_relation = 1.0, 0.0
    for row, instance_a in enumerate(instances_a):
        for col, instance_b in enumerate(instances_b):
            appearance = float(np.dot(instance_a.feature, instance_b.feature))
            if lambda_relation > 0 and relation_memory is not None:
                vector = pair_feature(instance_a, instance_b).astype(np.float32)
                prototype = relation_memory.pair_prototype
                denominator = float(
                    np.linalg.norm(vector) * np.linalg.norm(prototype) + 1e-8
                )
                relation_similarity = float(np.dot(vector, prototype) / denominator)
            else:
                relation_similarity = appearance
            similarity = (
                lambda_feature * appearance + lambda_relation * relation_similarity
            )
            cost[row, col] = 1.0 - similarity
    return cost


def _evaluate_one_to_one(
    atom: RelationAtom,
    scene_graph: SceneGraph,
    support_memory: Optional[SupportMemory],
    config: Config,
) -> AtomicResult:
    component_a, component_b = atom.components[0], atom.components[1]
    instances_a = scene_graph.instances_for(component_a)
    instances_b = scene_graph.instances_for(component_b)
    grid_size = scene_graph.grid_size()

    if not instances_a or not instances_b:
        missing = component_a if not instances_a else component_b
        responsibility = _expected_mask(support_memory, missing, grid_size)
        return _result(
            atom,
            0.0,
            responsibility,
            {
                "count_a": len(instances_a),
                "count_b": len(instances_b),
                "missing": missing,
                "localized": bool(responsibility.any()),
            },
        )

    relation_memory = (
        support_memory.relation("OneToOne", component_a, component_b)
        if support_memory is not None
        else None
    )
    cost = _one_to_one_cost(instances_a, instances_b, relation_memory, config)
    n_a, n_b = cost.shape
    dummy = float(config.logic.dummy_match_cost)
    large = 1e6
    size = n_a + n_b
    matrix = np.full((size, size), large, dtype=np.float64)
    matrix[:n_a, :n_b] = cost
    matrix[np.arange(n_a), n_b + np.arange(n_a)] = dummy
    matrix[n_a + np.arange(n_b), np.arange(n_b)] = dummy
    matrix[n_a:, n_b:] = 0.0
    rows, cols = linear_sum_assignment(matrix)

    real_matches = []
    for row, col in zip(rows, cols):
        if row < n_a and col < n_b:
            real_matches.append((int(row), int(col), float(cost[row, col])))
    matched_rows = {row for row, _, _ in real_matches}
    matched_cols = {col for _, col, _ in real_matches}
    unmatched_a = [index for index in range(n_a) if index not in matched_rows]
    unmatched_b = [index for index in range(n_b) if index not in matched_cols]
    bad_matches = [
        (row, col)
        for row, col, value in real_matches
        if value > dummy
    ]

    if real_matches:
        quality = float(np.mean([1.0 - value for _, _, value in real_matches]))
    else:
        quality = 0.0
    coverage = len(real_matches) / float(max(n_a, n_b))
    truth = _clip01(quality * coverage)

    responsibility = np.zeros(grid_size, dtype=np.float32)
    if unmatched_a:
        responsibility = np.maximum(
            responsibility,
            _union_masks([instances_a[index] for index in unmatched_a], grid_size),
        )
    if unmatched_b:
        responsibility = np.maximum(
            responsibility,
            _slot_mask(support_memory, component_b, grid_size),
        )
    if bad_matches:
        bad_instances = [instances_a[row] for row, _ in bad_matches] + [
            instances_b[col] for _, col in bad_matches
        ]
        responsibility = np.maximum(
            responsibility, _union_masks(bad_instances, grid_size)
        )

    return _result(
        atom,
        truth,
        responsibility,
        {
            "count_a": n_a,
            "count_b": n_b,
            "matches": [
                {
                    "a": instances_a[row].instance_id,
                    "b": instances_b[col].instance_id,
                    "cost": value,
                }
                for row, col, value in real_matches
            ],
            "unmatched_a": [instances_a[index].instance_id for index in unmatched_a],
            "unmatched_b": [instances_b[index].instance_id for index in unmatched_b],
            "bad_matches": [
                {
                    "a": instances_a[row].instance_id,
                    "b": instances_b[col].instance_id,
                }
                for row, col in bad_matches
            ],
            "mean_cost": float(np.mean([value for _, _, value in real_matches]))
            if real_matches
            else 1.0,
        },
    )


def _evaluate_spatial(
    atom: RelationAtom,
    scene_graph: SceneGraph,
    support_memory: Optional[SupportMemory],
    config: Config,
) -> AtomicResult:
    component_a, component_b = atom.components[0], atom.components[1]
    relation = atom.spatial_relation or "Inside"
    instances_a = scene_graph.instances_for(component_a)
    instances_b = scene_graph.instances_for(component_b)
    grid_size = scene_graph.grid_size()

    if not instances_a or not instances_b:
        missing = component_a if not instances_a else component_b
        responsibility = _expected_mask(support_memory, missing, grid_size)
        return _result(
            atom,
            0.0,
            responsibility,
            {
                "relation": relation,
                "count_a": len(instances_a),
                "count_b": len(instances_b),
                "missing": missing,
                "localized": bool(responsibility.any()),
            },
        )

    memory = (
        support_memory.relation(relation, component_a, component_b)
        if support_memory is not None
        else None
    )
    best_percent = []
    best_partner = []
    best_scores = []
    for instance_a in instances_a:
        scores = []
        for instance_b in instances_b:
            heuristic = _heuristic_relation(relation, instance_a, instance_b)
            if memory is not None and memory.sample_count >= 2:
                distance = mahalanobis(
                    pair_geometry(instance_a, instance_b),
                    memory.mean_geometry,
                    memory.inverse_covariance,
                )
                memory_score = float(math.exp(-0.5 * distance))
                score = 0.5 * heuristic + 0.5 * memory_score
            else:
                score = heuristic
            scores.append(score)
        best_index = int(np.argmax(scores))
        best_scores.append(float(scores[best_index]))
        best_partner.append(instances_b[best_index])
        best_percent.append(containment(instance_a, instances_b[best_index]))

    truth = _clip01(float(np.mean(best_scores)))
    failing = [
        (instance, score)
        for instance, score in zip(instances_a, best_scores)
        if score < config.logic.fail_threshold
    ]
    if failing and config.logic.soft_responsibility:
        severity = [
            1.0 - min(
                1.0, float(score) / max(float(config.logic.fail_threshold), 1e-6)
            )
            for _, score in failing
        ]
        responsibility = _weighted_instance_masks(
            [instance for instance, _ in failing],
            _component_map(scene_graph, component_a),
            severity,
            grid_size,
        )
    else:
        responsibility = _union_masks(
            [instance for instance, _ in failing], grid_size
        )
    if not responsibility.any() and truth < config.logic.fail_threshold:
        # The source component A is the subject of Relation(A, B).  Projecting
        # the expected mask of B (often a large container such as Bottle) made
        # a local relation failure light up most of the object.
        responsibility = _expected_mask(support_memory, component_a, grid_size)

    return _result(
        atom,
        truth,
        responsibility,
        {
            "relation": relation,
            "count_a": len(instances_a),
            "count_b": len(instances_b),
            "best_scores": best_scores,
            "best_partners": [item.instance_id for item in best_partner],
            "containment": best_percent,
            "used_support_memory": bool(memory is not None and memory.sample_count >= 2),
        },
    )


def _result(
    atom: RelationAtom,
    truth: float,
    responsibility: np.ndarray,
    evidence: Dict[str, object],
) -> AtomicResult:
    truth = _clip01(truth)
    responsibility = np.clip(
        np.asarray(responsibility, dtype=np.float32), 0.0, 1.0
    )
    return AtomicResult(
        atom_id=atom.atom_id,
        expression=atom.expression,
        kind=atom.kind,
        truth=truth,
        violation=float(1.0 - truth),
        responsibility_map=responsibility,
        valid=atom.weight > 0,
        weight=float(atom.weight),
        evidence=evidence,
    )


def evaluate_atoms(
    scene_graph: SceneGraph,
    rule_kb: RuleKB,
    support_memory: Optional[SupportMemory],
    config: Config,
    backbone: Optional[CLIPBackbone] = None,
) -> Dict[str, AtomicResult]:
    results: Dict[str, AtomicResult] = {}
    for atom in rule_kb.atoms:
        if atom.weight <= 0:
            grid_size = scene_graph.grid_size()
            results[atom.atom_id] = AtomicResult(
                atom_id=atom.atom_id,
                expression=atom.expression,
                kind=atom.kind,
                truth=1.0,
                violation=0.0,
                responsibility_map=np.zeros(grid_size, dtype=np.float32),
                valid=False,
                weight=0.0,
                evidence={"disabled": True},
            )
            continue
        if atom.kind == "Exist":
            result = _evaluate_exist(atom, scene_graph, support_memory, config)
        elif atom.kind == "CountEq":
            result = _evaluate_count_eq(atom, scene_graph, support_memory, config)
        elif atom.kind == "CountInRange":
            result = _evaluate_count_in_range(atom, scene_graph, support_memory, config)
        elif atom.kind == "AttributeMatch":
            result = _evaluate_attribute_match(
                atom, scene_graph, support_memory, config, backbone
            )
        elif atom.kind == "PositionMatch":
            result = _evaluate_position_match(
                atom, scene_graph, support_memory, config
            )
        elif atom.kind == "CountRatio":
            result = _evaluate_count_ratio(atom, scene_graph, support_memory, config)
        elif atom.kind == "OneToOne":
            result = _evaluate_one_to_one(atom, scene_graph, support_memory, config)
        elif atom.kind == "SpatialRelation":
            result = _evaluate_spatial(atom, scene_graph, support_memory, config)
        else:  # pragma: no cover - validator blocks unknown kinds
            grid_size = scene_graph.grid_size()
            result = AtomicResult(
                atom_id=atom.atom_id,
                expression=atom.expression,
                kind=atom.kind,
                truth=1.0,
                violation=0.0,
                responsibility_map=np.zeros(grid_size, dtype=np.float32),
                valid=False,
                weight=0.0,
                evidence={"unsupported": True},
            )
        results[atom.atom_id] = result
    return results
