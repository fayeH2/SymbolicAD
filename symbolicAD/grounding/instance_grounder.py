"""组件 mask grounding 入口，并显式记录人工/自动证据来源。

正式无监督推理只能使用 ``from_binary_observation`` 或
``from_probability_map``。``from_manual_mask`` 用于诊断上界，返回值会将
``uses_ground_truth_during_inference`` 标为真，防止 GT 组件 mask 混入正式结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ..perception.component_instances import (
    ComponentInstance, instances_from_binary_mask, instances_from_label_mask,
)
from ..perception.scene_graph import SceneGraph, build_scene_graph


@dataclass(frozen=True)
class InstanceGrounding:
    scene: SceneGraph
    source: str
    uses_ground_truth_during_inference: bool


class InstanceGrounder:
    """把不同 mask 来源统一转换成同一种 SceneGraph。"""
    def from_manual_mask(
        self,
        label_mask: np.ndarray,
        *,
        entity_type: str,
        image: Optional[np.ndarray] = None,
        diagnostic_only: bool = True,
    ) -> InstanceGrounding:
        instances = instances_from_label_mask(
            label_mask, entity_type=entity_type, image=image,
            source="manual_mask", confidence=1.0,
        )
        scene = build_scene_graph(
            instances, {index: entity_type for index in range(len(instances))}
        )
        scene.grounding_confidence = 1.0
        return InstanceGrounding(scene, "manual_mask", bool(diagnostic_only))

    def from_binary_observation(
        self,
        binary_mask: np.ndarray,
        *,
        entity_type: str,
        image: Optional[np.ndarray] = None,
        confidence: float = 0.8,
        min_area: int = 4,
    ) -> InstanceGrounding:
        instances = instances_from_binary_mask(
            binary_mask, entity_type=entity_type, image=image,
            min_area=min_area, source="automatic_binary_mask",
            confidence=confidence,
        )
        scene = build_scene_graph(
            instances, {index: entity_type for index in range(len(instances))}
        )
        scene.grounding_confidence = float(np.clip(confidence, 0.0, 1.0))
        return InstanceGrounding(scene, "automatic", False)

    def from_probability_map(
        self,
        probability: np.ndarray,
        *,
        entity_type: str,
        image: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        min_area: int = 4,
    ) -> InstanceGrounding:
        values = np.asarray(probability, dtype=np.float32)
        binary = values >= float(threshold)
        # 空概率图对应 confidence=0，随后 count executor 会输出 UNKNOWN 而非缺件。
        confidence = float(np.mean(values[binary])) if np.any(binary) else 0.0
        result = self.from_binary_observation(
            binary, entity_type=entity_type, image=image,
            confidence=confidence, min_area=min_area,
        )
        result.scene.observation_complete = confidence >= 0.5
        return result


__all__ = ["InstanceGrounding", "InstanceGrounder"]
