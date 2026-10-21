"""由正常 support 对齐结果学习缺失组件的槽位概率图。

缺失物体在 query 中没有实例 mask，所以不能画一个任意检测框。槽位图是 support
中同一语义位置 mask 的均值；``support_frequency`` 暴露该槽位在 K-shot 下是否
稳定。它只提供定位证据，不负责决定数量规则真假。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class SlotPrototype:
    slot_id: str
    entity_type: str
    probability_map: np.ndarray
    support_frequency: float


class SlotGrounder:
    @staticmethod
    def fit(
        aligned_masks: Mapping[str, Sequence[np.ndarray]],
        *,
        entity_type: str,
        map_size: int = 256,
    ) -> Dict[str, SlotPrototype]:
        result = {}
        for slot_id, masks in aligned_masks.items():
            # 先对齐再 resize；若上游没有建立同一 slot，对应 mask 不应放在一起。
            resized = [
                cv2.resize(
                    np.asarray(mask, dtype=np.float32), (map_size, map_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                for mask in masks if mask is not None
            ]
            if not resized:
                continue
            result[str(slot_id)] = SlotPrototype(
                str(slot_id), entity_type,
                np.clip(np.mean(np.stack(resized), axis=0), 0.0, 1.0).astype(np.float32),
                float(len(resized) / max(len(masks), 1)),
            )
        return result

    @staticmethod
    def attach(scene, slots: Mapping[str, SlotPrototype]) -> None:
        scene.slot_probability_maps = {
            slot_id: slot.probability_map for slot_id, slot in slots.items()
        }


__all__ = ["SlotPrototype", "SlotGrounder"]
