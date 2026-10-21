"""通过局部反事实修复估计规则违反的空间责任。

这里不生成“看起来正常”的 RGB 图，而是在规则参数空间构造最小修复：把实际
长度移动到最近合法长度，并只标记需要删除/补足的端点范围。缺失区域本质上比
已观测的多余区域不确定，因此 uncertainty 会更高。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RepairMap:
    anomaly_map: np.ndarray
    uncertainty_map: np.ndarray
    strategy: str


def length_endpoint_repair(
    instance,
    target_length: float,
    *,
    map_size: int = 256,
    reference_endpoints=None,
) -> RepairMap:
    """返回把实例长度修到 ``target_length`` 时发生变化的端点责任图。"""
    attributes = getattr(instance, "attributes", getattr(instance, "features", {}))
    endpoint_a = attributes.get("endpoint_a")
    endpoint_b = attributes.get("endpoint_b")
    actual = float(attributes.get("length", 0.0))
    anomaly = np.zeros((map_size, map_size), dtype=np.float32)
    uncertainty = np.ones_like(anomaly)
    if endpoint_a is None or endpoint_b is None or actual <= 0.0:
        return RepairMap(anomaly, uncertainty, "length_endpoint_unavailable")
    a = np.asarray(endpoint_a, dtype=float)
    b = np.asarray(endpoint_b, dtype=float)
    center = (a + b) / 2.0
    direction = b - a
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return RepairMap(anomaly, uncertainty, "length_endpoint_unavailable")
    direction /= norm
    if reference_endpoints is not None:
        # 有支持集参考端点时先消除端点顺序二义性，再计算缺失/多余范围。
        reference_a = np.asarray(reference_endpoints[0], dtype=float)
        reference_b = np.asarray(reference_endpoints[1], dtype=float)
        direct = np.linalg.norm(a - reference_a) + np.linalg.norm(b - reference_b)
        swapped = np.linalg.norm(a - reference_b) + np.linalg.norm(b - reference_a)
        target_a, target_b = (
            (reference_a, reference_b) if direct <= swapped
            else (reference_b, reference_a)
        )
    else:
        target_a = center - direction * target_length / 2.0
        target_b = center + direction * target_length / 2.0

    def point(value: np.ndarray) -> tuple[int, int]:
        clipped = np.clip(value, 0.0, 1.0)
        return (
            int(round(clipped[0] * (map_size - 1))),
            int(round(clipped[1] * (map_size - 1))),
        )

    radius = max(2, int(round(map_size * 0.025)))
    if actual > target_length:
        # 多余部分在 query 中真实可见，空间归因置信度较高。
        cv2.line(anomaly, point(a), point(target_a), 1.0, radius)
        cv2.line(anomaly, point(b), point(target_b), 1.0, radius)
        uncertainty[anomaly > 0] = 0.15
        strategy = "remove_excess_endpoints"
    else:
        # 缺失部分是反事实预测而非已观测像素，因此显式提高不确定性。
        cv2.line(anomaly, point(a), point(target_a), 1.0, radius)
        cv2.line(anomaly, point(b), point(target_b), 1.0, radius)
        uncertainty[anomaly > 0] = 0.55
        strategy = (
            "add_missing_reference_endpoint_extent"
            if reference_endpoints is not None else "add_missing_endpoint_extent"
        )
    anomaly = cv2.GaussianBlur(anomaly, (0, 0), 2.0)
    if float(anomaly.max()) > 0:
        anomaly /= float(anomaly.max())
    return RepairMap(anomaly.astype(np.float32), uncertainty, strategy)


__all__ = ["RepairMap", "length_endpoint_repair"]
