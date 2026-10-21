"""关系违反的 witness 图：两个参与实例 mask 加可选关系连线。

该图表达“哪些实体/关系承担责任”，不把两个 bbox 之间的全部区域都当异常。
关系 executor 应先确认 pair 身份和关系真值，再调用本函数。
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .shape_grounder import resized_instance_mask


def relation_witness_map(
    source,
    target,
    *,
    map_size: int = 256,
    include_connection: bool = True,
) -> np.ndarray:
    result = np.maximum(
        resized_instance_mask(source, map_size),
        resized_instance_mask(target, map_size),
    )
    if include_connection:
        def point(instance):
            x, y = instance.centroid
            return int(round(x * (map_size - 1))), int(round(y * (map_size - 1)))
        cv2.line(result, point(source), point(target), 1.0, max(1, map_size // 128))
    return result.astype(np.float32)


__all__ = ["relation_witness_map"]
