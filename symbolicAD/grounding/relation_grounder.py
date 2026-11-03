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
from symbolicAD.rules.executor import AtomicResult
from symbolicAD.rules.schema import RelationAtom, RuleKB

from .geometry import bbox_iou, containment, mahalanobis, pair_feature, pair_geometry
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
        signed = -dy if relation == "Above" else dy
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
        truth = _clip01(max(instance.confidence for instance in instances))
        responsibility = _component_map(scene_graph, component)
        evidence = {
            "count": len(instances),
            "missing": False,
            "max_confidence": float(truth),
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
    estimated = float(sum(instance.confidence for instance in instances))
    sigma = max(0.5, float(config.logic.count_sigma) * max(1.0, 0.25 * target))
    truth = _gaussian_truth(estimated, target, sigma)

    responsibility = np.zeros(grid_size, dtype=np.float32)
    if estimated > target + 0.25 and instances:
        ordered = sorted(instances, key=lambda item: item.confidence, reverse=True)
        extra = ordered[max(int(round(target)), 0):]
        if not extra:
            extra = ordered[-1:]
        responsibility = _union_masks(extra, grid_size)
        evidence = {
            "estimated_count": estimated,
            "target": target,
            "mode": "excess",
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
            "target": target,
            "mode": "missing",
            "missing_expected_mass": float(responsibility.sum()),
        }
    else:
        evidence = {
            "estimated_count": estimated,
            "target": target,
            "mode": "ok",
        }
    return _result(atom, truth, responsibility, evidence)


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
        instance
        for instance, score in zip(instances_a, best_scores)
        if score < config.logic.fail_threshold
    ]
    responsibility = _union_masks(failing, grid_size)
    if 1.0 - truth > 1e-3:
        expected = _expected_mask(support_memory, component_b, grid_size)
        responsibility = np.maximum(responsibility, expected * (1.0 - truth))

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
