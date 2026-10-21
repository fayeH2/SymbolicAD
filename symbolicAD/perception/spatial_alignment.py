"""将实体身份、形状和位置证据分开的 Hungarian 实例对齐。

三种 cost 会单独写入 :class:`AlignmentMatch`，便于诊断“认错实体”还是“位置
异常”。位置项只用于建立对应，不能反过来冒充身份置信度；未超过最大代价的匹配
才会进入规则执行，其余实例显式进入 unmatched 集合。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AlignmentMatch:
    reference_index: int
    query_index: int
    identity_cost: float
    shape_cost: float
    position_cost: float
    confidence: float


@dataclass(frozen=True)
class AlignmentResult:
    matches: Tuple[AlignmentMatch, ...]
    unmatched_reference: Tuple[int, ...]
    unmatched_query: Tuple[int, ...]
    cost_matrix: np.ndarray


def _relative_difference(left: float, right: float) -> float:
    return float(min(abs(left - right) / max(abs(left), abs(right), 1e-6), 2.0))


def _embedding_cost(left, right) -> float:
    a, b = getattr(left, "embedding", None), getattr(right, "embedding", None)
    if a is None or b is None:
        return 0.0 if left.entity_type == right.entity_type else 1.0
    a, b = np.asarray(a, dtype=float).reshape(-1), np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        return 1.0
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
    return float(np.clip(1.0 - float(np.dot(a, b)) / denominator, 0.0, 2.0))


def align_instances(
    reference: Sequence,
    query: Sequence,
    *,
    identity_weight: float = 0.55,
    shape_weight: float = 0.30,
    position_weight: float = 0.15,
    maximum_cost: float = 1.0,
) -> AlignmentResult:
    """在全局一一匹配约束下对齐实例，同时保留三项原始代价。"""
    if not reference or not query:
        return AlignmentResult(
            tuple(), tuple(range(len(reference))), tuple(range(len(query))),
            np.empty((len(reference), len(query)), dtype=np.float32),
        )
    costs = np.zeros((len(reference), len(query)), dtype=np.float32)
    components = {}
    for i, left in enumerate(reference):
        for j, right in enumerate(query):
            identity = _embedding_cost(left, right)
            left_shape = getattr(left, "attributes", {})
            right_shape = getattr(right, "attributes", {})
            shape_terms = [
                _relative_difference(
                    float(left_shape[key]), float(right_shape[key])
                )
                for key in ("length", "width", "area")
                if key in left_shape and key in right_shape
            ]
            shape = float(np.mean(shape_terms)) if shape_terms else 0.5
            position = float(np.linalg.norm(
                np.asarray(left.centroid, dtype=float)
                - np.asarray(right.centroid, dtype=float)
            ))
            # 加权只用于 assignment；components 中仍保留各项代价以便审计。
            total = (
                identity_weight * identity
                + shape_weight * shape
                + position_weight * position
            )
            costs[i, j] = total
            components[(i, j)] = (identity, shape, position)
    rows, columns = linear_sum_assignment(costs)
    matches: List[AlignmentMatch] = []
    matched_reference, matched_query = set(), set()
    for i, j in zip(rows.tolist(), columns.tolist()):
        # Hungarian 一定会给出配对，但高代价配对不代表可靠对应，必须拒绝。
        if float(costs[i, j]) > float(maximum_cost):
            continue
        identity, shape, position = components[(i, j)]
        matches.append(AlignmentMatch(
            i, j, identity, shape, position,
            float(np.clip(1.0 - costs[i, j] / max(maximum_cost, 1e-8), 0.0, 1.0)),
        ))
        matched_reference.add(i)
        matched_query.add(j)
    return AlignmentResult(
        tuple(matches),
        tuple(sorted(set(range(len(reference))) - matched_reference)),
        tuple(sorted(set(range(len(query))) - matched_query)),
        costs,
    )


__all__ = ["AlignmentMatch", "AlignmentResult", "align_instances"]
