"""把开放词表检测结果变成可执行 SceneGraph / ExecutionContext。

设计约束（与 ``reasoning.predicate_executor`` 的契约一致）：

* 场景事实只包含视觉观测，不含任何异常标签、GT mask 或 query 真值。
* 数量/存在类执行器要求逐实体的观测完整性校准。校准结果由
  ``facts.calibration`` 在独立 train 正常图上估计后注入 ``metadata``；
  未校准的实体一律 ``UNKNOWN``，绝不把"没检测到"当成"真实缺失"。
* 实例 mask 是检测框内的实心区域。当前只声称框几何、实例计数、区域包含和
  质心关系可用；像素级轮廓类谓词仍需 SAM 精修后才可信。
* ``type_hierarchy`` 把上位词（如 ``fruit``）映射到实际检测到的下位词
  （``peach``/``orange``），因此 ``count_in(fruit, left_compartment)`` 不会
  因为感知层没有名为 ``fruit`` 的检测而误判为缺件。
* 检测记录可序列化，因此可以在多 GPU 上并行抽取并离线重建场景图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from ...perception.component_instances import ComponentInstance
from ...perception.scene_graph import SceneGraph, build_scene_graph
from ...reasoning.predicate_executor import ExecutionContext
from .calibration import CountCalibration
from .open_vocab import Detection, DetectionBatch, GroundingDinoDetector, Vocabulary


DEFAULT_BREAKFAST_VOCABULARY = Vocabulary({
    "peach": ("peach",),
    "orange": ("orange",),
    "granola": ("granola", "cereal", "muesli"),
    "banana_slice": ("banana",),
    "almond": ("almond",),
    "bento_box": ("bento box", "lunch box", "bento", "lunchbox"),
})

DEFAULT_SUPERTYPES: Mapping[str, Tuple[str, ...]] = {
    "fruit": ("peach", "orange"),
}


@dataclass(frozen=True)
class SceneFactConfig:
    """事实构建配置；阈值会随 ``ExecutionContext.metadata`` 一起留痕。"""

    vocabulary: Vocabulary = DEFAULT_BREAKFAST_VOCABULARY
    container_entity: str = "bento_box"
    left_entity: str = "left_compartment"
    right_entity: str = "right_compartment"
    supertypes: Mapping[str, Tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_SUPERTYPES)
    )
    divider_search_fraction: Tuple[float, float] = (0.25, 0.75)
    min_objects: int = 1
    map_size: int = 128


def estimate_vertical_divider(
    gray: np.ndarray,
    container_box: Tuple[float, float, float, float],
    search_fraction: Tuple[float, float] = (0.25, 0.75),
) -> Tuple[int, float]:
    """在容器中部搜索贯穿性的竖直分隔线，返回 ``(x_pixel, 相对位置)``。"""

    x0, y0, x1, y1 = container_box
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    lower = int(x0 + search_fraction[0] * width)
    upper = int(x0 + search_fraction[1] * width)
    rows = slice(int(y0 + 0.05 * height), int(y1 - 0.05 * height))
    gradient = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    column = gradient[rows].mean(axis=0)
    if column.size >= 21:
        column = cv2.GaussianBlur(column.reshape(1, -1), (1, 21), 0).ravel()
    segment = column[lower:upper]
    divider = (
        int((x0 + x1) / 2.0) if segment.size == 0
        else int(lower + int(np.argmax(segment)))
    )
    return divider, float((divider - x0) / width)


def _rectangle_mask(
    shape: Tuple[int, int], box: Tuple[float, float, float, float]
) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = box
    mask = np.zeros((height, width), dtype=bool)
    left = int(np.clip(np.floor(x0), 0, width - 1))
    right = int(np.clip(np.ceil(x1), 1, width))
    top = int(np.clip(np.floor(y0), 0, height - 1))
    bottom = int(np.clip(np.ceil(y1), 1, height))
    mask[top:bottom, left:right] = True
    return mask


def estimate_container_box(
    image: np.ndarray, minimum_area_fraction: float = 0.4
) -> Tuple[Tuple[float, float, float, float], float]:
    """从"亮托盘 + 暗背景"估计容器几何，返回 ``(归一化 box, 置信度)``。

    LOCO 是固定机位，托盘永远是最亮的大连通域。这条几何事实与实体语义、
    规则和异常标签无关，因此可以安全地作为隔间划分的兜底来源；当检测器
    同时给出一致的 bento box 时以检测器为准。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    if height < 8 or width < 8:
        return (0.0, 0.0, 1.0, 1.0), 0.0
    threshold, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary = (gray > threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count < 2:
        return (0.0, 0.0, 1.0, 1.0), 0.0
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[index]
    box = (
        float(x) / width, float(y) / height,
        float(x + w) / width, float(y + h) / height,
    )
    area_fraction = (w * h) / float(width * height)
    confidence = 1.0 if area_fraction >= float(minimum_area_fraction) else 0.0
    return box, confidence


def detection_record(
    image_path: str | Path,
    image_size: Tuple[int, int],
    batch: DetectionBatch,
    foreground_box: Optional[Sequence[float]] = None,
    foreground_confidence: Optional[float] = None,
    divider_fraction: Optional[float] = None,
) -> Dict[str, Any]:
    """可序列化的检测记录；多 GPU 抽取后离线重建场景图。"""
    return {
        "image_path": str(Path(image_path).resolve()),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "foreground_box": (
            None if foreground_box is None else [float(v) for v in foreground_box]
        ),
        "foreground_confidence": (
            None if foreground_confidence is None else float(foreground_confidence)
        ),
        "divider_fraction": (
            None if divider_fraction is None else float(divider_fraction)
        ),
        "detections": [
            {
                "entity_type": row.entity_type,
                "phrase": row.phrase,
                "score": float(row.score),
                "box_xyxy": [float(value) for value in row.box_xyxy],
            }
            for row in batch.detections
        ],
        "rejected": [dict(row) for row in batch.rejected],
    }


def batch_from_record(record: Mapping[str, Any]) -> DetectionBatch:
    detections = tuple(
        Detection(
            entity_type=str(row["entity_type"]),
            phrase=str(row.get("phrase", "")),
            score=float(row["score"]),
            box_xyxy=tuple(float(value) for value in row["box_xyxy"]),
        )
        for row in record.get("detections", [])
    )
    return DetectionBatch(detections, tuple(record.get("rejected", ())))


def select_container(
    detections: Sequence[Any],
    foreground_box: Optional[Sequence[float]],
    foreground_confidence: float,
    image_size: Tuple[int, int],
    container_entity: str,
) -> Tuple[Tuple[float, float, float, float], float, Optional[str], str]:
    """Return ``(box, confidence, detector_phrase, source)`` for the tray."""
    width, height = image_size
    containers = [
        row for row in detections if row.entity_type == container_entity
    ]
    if containers:
        container = max(containers, key=lambda row: row.score)
        return (
            tuple(float(value) for value in container.box_xyxy),
            max(float(container.score), float(foreground_confidence)),
            container.phrase,
            "groundingdino_bento_box",
        )
    if foreground_box is not None and float(foreground_confidence) > 0.0:
        box = tuple(
            float(value) * (width if index % 2 == 0 else height)
            for index, value in enumerate(foreground_box)
        )
        return box, float(foreground_confidence), None, "bright_foreground_silhouette"
    return (0.0, 0.0, float(width), float(height)), 0.0, None, "full_frame_unverified"


class SceneFactBuilder:
    """图像 -> 可序列化检测记录 -> ``ExecutionContext``。"""

    def __init__(
        self,
        detector: GroundingDinoDetector,
        config: Optional[SceneFactConfig] = None,
    ) -> None:
        self.detector = detector
        self.config = config or SceneFactConfig()

    def extract(self, image_path: str | Path) -> Dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image)
        batch = self.detector.detect(image, self.config.vocabulary)
        box, confidence = estimate_container_box(rgb)
        container_box, _, _, _ = select_container(
            batch.detections, box, confidence, image.size,
            self.config.container_entity,
        )
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, divider_fraction = estimate_vertical_divider(
            gray, container_box, self.config.divider_search_fraction
        )
        return detection_record(
            image_path, image.size, batch, box, confidence, divider_fraction,
        )

    def build(
        self,
        image_path: str | Path,
        calibration: Optional[CountCalibration] = None,
    ) -> ExecutionContext:
        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image)
        batch = self.detector.detect(image, self.config.vocabulary)
        box, confidence = estimate_container_box(rgb)
        container_box, _, _, _ = select_container(
            batch.detections, box, confidence, image.size,
            self.config.container_entity,
        )
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, divider_fraction = estimate_vertical_divider(
            gray, container_box, self.config.divider_search_fraction
        )
        record = detection_record(
            image_path, image.size, batch, box, confidence, divider_fraction,
        )
        return self.build_from_record(record, calibration=calibration, image=rgb)

    def build_from_record(
        self,
        record: Mapping[str, Any],
        *,
        calibration: Optional[CountCalibration] = None,
        image: Optional[np.ndarray] = None,
        edited: bool = False,
        intervention: Optional[Mapping[str, Any]] = None,
    ) -> ExecutionContext:
        batch = batch_from_record(record)
        width, height = (int(value) for value in record["image_size"])
        gray = None
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = np.zeros((height, width), dtype=np.uint8)

        containers = [
            row for row in batch.detections
            if row.entity_type == self.config.container_entity
        ]
        objects = [
            row for row in batch.detections
            if row.entity_type != self.config.container_entity
        ]
        foreground = record.get("foreground_box")
        foreground_confidence = float(record.get("foreground_confidence") or 0.0)
        container_box, container_confidence, container_phrase, container_source = (
            select_container(
                batch.detections, foreground, foreground_confidence,
                (width, height), self.config.container_entity,
            )
        )
        stored_divider = record.get("divider_fraction")
        if stored_divider is not None:
            # Reuse the geometry measured on the real pixels, so rebuilding a
            # fact from a cached record never silently changes the divider.
            divider_fraction = float(stored_divider)
            divider_x = int(
                container_box[0]
                + divider_fraction * (container_box[2] - container_box[0])
            )
        else:
            divider_x, divider_fraction = estimate_vertical_divider(
                gray, container_box, self.config.divider_search_fraction
            )

        components: List[ComponentInstance] = []
        component_types: Dict[int, str] = {}
        for index, detection in enumerate(objects):
            mask = _rectangle_mask((height, width), detection.box_xyxy)
            if not np.any(mask):
                continue
            component_types[len(components)] = detection.entity_type
            components.append(ComponentInstance.from_mask(
                f"{detection.entity_type}_{index}", detection.entity_type, mask,
                image=image, confidence=detection.score,
                source="groundingdino_swint_ogc",
                attributes={
                    "detection_phrase": detection.phrase,
                    "detection_score": detection.score,
                    "box_xyxy": list(detection.box_xyxy),
                },
            ))

        left_box = (container_box[0], container_box[1], float(divider_x), container_box[3])
        right_box = (float(divider_x), container_box[1], container_box[2], container_box[3])
        for entity_type, box in (
            (self.config.left_entity, left_box),
            (self.config.right_entity, right_box),
        ):
            mask = _rectangle_mask((height, width), box)
            if not np.any(mask):
                continue
            component_types[len(components)] = entity_type
            components.append(ComponentInstance.from_mask(
                f"{entity_type}_0", entity_type, mask,
                image=image, confidence=float(np.clip(container_confidence, 0.0, 1.0)),
                source="compartment_split",
                attributes={"box_xyxy": list(box)},
            ))

        scene = build_scene_graph(components, component_types)
        scene.type_hierarchy = {
            supertype: tuple(subtypes)
            for supertype, subtypes in self.config.supertypes.items()
        }
        object_scores = [row.score for row in objects]
        mean_object = float(np.mean(object_scores)) if object_scores else 0.0
        scene.grounding_confidence = float(np.clip(
            0.5 * container_confidence + 0.5 * mean_object, 0.0, 1.0
        ))
        scene.observation_complete = bool(
            container_confidence > 0.0
            and len(objects) >= self.config.min_objects
        )

        metadata: Dict[str, Any] = {
            "image_path": str(record.get("image_path", "")),
            "image_size": [width, height],
            "container_detected": container_confidence > 0.0,
            "container_source": container_source,
            "container_phrase": container_phrase,
            "container_confidence": float(container_confidence),
            "container_box": list(container_box),
            "divider_x": int(divider_x),
            "divider_fraction": divider_fraction,
            "num_object_detections": len(objects),
            "rejected_detections": [dict(row) for row in batch.rejected],
            "detector": "groundingdino_swint_ogc",
            "edited_image": bool(edited),
        }
        if intervention is not None:
            metadata["intervention"] = dict(intervention)
        if calibration is not None:
            metadata.update(calibration.metadata())
        else:
            # Without calibration the executors must fail closed.
            metadata.update({
                "count_observation_complete": {},
                "count_observation_confidence": {},
                "presence_observation_complete": {},
                "presence_observation_confidence": {},
                "calibration_split": "uncalibrated",
            })
        return ExecutionContext(
            scene, map_size=self.config.map_size, metadata=metadata
        )

    # -- calibration helpers ------------------------------------------------
    def entity_counts(
        self,
        record: Mapping[str, Any],
    ) -> Dict[str, Tuple[int, float]]:
        """逐实体 ``(count, top_score)``；含上位词展开。"""
        context = self.build_from_record(record)
        counts: Dict[str, Tuple[int, float]] = {}
        names = set(self.config.vocabulary.entity_types)
        names.discard(self.config.container_entity)
        names.update(self.config.supertypes)
        for name in names:
            instances = context.scene.instances_of(name)
            top = max(
                (float(getattr(row, "confidence", 0.0)) for row in instances),
                default=0.0,
            )
            counts[name] = (len(instances), top)
        return counts


__all__ = [
    "DEFAULT_BREAKFAST_VOCABULARY", "DEFAULT_SUPERTYPES", "SceneFactConfig",
    "SceneFactBuilder", "estimate_vertical_divider", "detection_record",
    "batch_from_record",
]
