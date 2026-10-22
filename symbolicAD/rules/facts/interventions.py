"""像素级语义干预与独立视觉确认。

``L_cf`` 要求的是一个真实的"同一场景、语义被改坏"的视觉对照，而不是把规则
自己的真值翻转。本模块因此只做两件事：

1. 在真实 normal 图像上执行确定性的像素编辑（擦除、平移、竖直翻转）；
2. 用**同一套感知**重新检测编辑后的图像，并由一个只读取检测事实、不读取规则
   状态的确认器判断"预期的语义变化是否真的发生了"。

只有确认器通过时，``CounterfactualPair.validated`` 才为 ``True``。确认器给出
的逐项测量（前后计数、质心位移）会写进 ``validation_evidence``，因此任何一条
反事实都可以回溯到具体像素和具体测量。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .scene_builder import batch_from_record, select_container


def _pad_box(box: Sequence[float], pad: int, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    height, width = shape
    x0, y0, x1, y1 = (float(value) for value in box)
    return (
        int(np.clip(np.floor(x0) - pad, 0, width - 1)),
        int(np.clip(np.floor(y0) - pad, 0, height - 1)),
        int(np.clip(np.ceil(x1) + pad, 1, width)),
        int(np.clip(np.ceil(y1) + pad, 1, height)),
    )


def _rect_mask(shape: Tuple[int, int], box: Tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = 255
    return mask


def _detection_masks(
    record: Mapping[str, Any], shape: Tuple[int, int], pad: int = 2
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for row in record.get("detections", []):
        x0, y0, x1, y1 = _pad_box(row["box_xyxy"], pad, shape)
        mask[y0:y1, x0:x1] = 255
    return mask


def _background_colour(
    image: np.ndarray,
    region_box: Tuple[int, int, int, int],
    record: Mapping[str, Any],
) -> Tuple[int, int, int]:
    """估计容器背景色：区域内、检测框之外、亮度较高像素的中位色。"""
    shape = image.shape[:2]
    region = _rect_mask(shape, region_box) > 0
    covered = _detection_masks(record, shape, pad=4) > 0
    candidate = region & ~covered
    pixels = image[candidate]
    if pixels.size == 0:
        return (240, 240, 240)
    brightness = pixels.reshape(-1, 3).mean(axis=1)
    bright = pixels.reshape(-1, 3)[brightness > 150]
    chosen = bright if bright.size else pixels.reshape(-1, 3)
    median = np.median(chosen, axis=0)
    return tuple(int(np.clip(value, 0, 255)) for value in median)


def _fill(image: np.ndarray, box: Tuple[int, int, int, int], colour: Tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    x0, y0, x1, y1 = box
    output[y0:y1, x0:x1] = np.asarray(colour, dtype=np.uint8)
    return output


@dataclass(frozen=True)
class PixelEdit:
    kind: str
    description: str
    boxes: Tuple[Sequence[float], ...] = ()
    target_entity: Optional[str] = None
    source_entity: Optional[str] = None
    extra: Mapping[str, Any] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "boxes": [list(box) for box in self.boxes],
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "extra": dict(self.extra or {}),
        }


@dataclass(frozen=True)
class EditedImage:
    image: np.ndarray
    edit: PixelEdit
    changed_pixels: int
    sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edit": self.edit.to_dict(),
            "changed_pixels": self.changed_pixels,
            "sha256": self.sha256,
        }


def _hash_image(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _finish(original: np.ndarray, edited: np.ndarray, edit: PixelEdit) -> EditedImage:
    changed = int(np.count_nonzero(np.any(edited != original, axis=2)))
    return EditedImage(edited, edit, changed, _hash_image(edited))


def empty_region(
    image: np.ndarray,
    record: Mapping[str, Any],
    region_box: Sequence[float],
    *,
    inset: int = 6,
) -> Optional[EditedImage]:
    """清空一个隔间的全部内容，保留容器边框。

    这是最强的"存在性"干预：它同时否证该隔间内所有 ``exists`` 规则，而且
    不依赖擦除单个小目标时检测器的残余响应。
    """
    shape = image.shape[:2]
    x0, y0, x1, y1 = _pad_box(region_box, 0, shape)
    x0, y0, x1, y1 = x0 + inset, y0 + inset, x1 - inset, y1 - inset
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    colour = _background_colour(
        image, _pad_box(region_box, 0, shape), record,
    )
    edited = _fill(image.copy(), (x0, y0, x1, y1), colour)
    return _finish(image, edited, PixelEdit(
        kind="empty_region",
        description="erase every object inside one compartment",
        boxes=((x0, y0, x1, y1),),
        extra={"region": list(region_box)},
    ))


def erase_entity(
    image: np.ndarray,
    record: Mapping[str, Any],
    entity_type: str,
    region_box: Sequence[float],
    *,
    pad: int = 6,
) -> Optional[EditedImage]:
    boxes = [
        row["box_xyxy"] for row in record.get("detections", [])
        if row["entity_type"] == entity_type
    ]
    if not boxes:
        return None
    shape = image.shape[:2]
    region = _pad_box(region_box, 0, shape)
    colour = _background_colour(image, region, record)
    edited = image.copy()
    padded = []
    for box in boxes:
        padded_box = _pad_box(box, pad, shape)
        edited = _fill(edited, padded_box, colour)
        padded.append(padded_box)
    return _finish(image, edited, PixelEdit(
        kind="erase_entity",
        description=f"erase every observed {entity_type} instance",
        boxes=tuple(boxes),
        source_entity=entity_type,
    ))


def relocate_entity_to_region(
    image: np.ndarray,
    record: Mapping[str, Any],
    entity_type: str,
    source_region_box: Sequence[float],
    target_region_box: Sequence[float],
    *,
    pad: int = 6,
    anchor: Tuple[float, float] = (0.5, 0.78),
) -> Optional[EditedImage]:
    """把该实体最靠上（最靠近区域入口）的实例整体搬到另一区域。"""
    shape = image.shape[:2]
    candidates = [
        row["box_xyxy"] for row in record.get("detections", [])
        if row["entity_type"] == entity_type
    ]
    if not candidates:
        return None
    source_region = _pad_box(source_region_box, 0, shape)
    sx0, sy0, sx1, sy1 = source_region
    inside = [
        box for box in candidates
        if sx0 <= (box[0] + box[2]) / 2 <= sx1 and sy0 <= (box[1] + box[3]) / 2 <= sy1
    ]
    if not inside:
        return None
    box = min(inside, key=lambda row: row[1])
    patch_box = _pad_box(box, pad, shape)
    patch = image[patch_box[1]:patch_box[3], patch_box[0]:patch_box[2]].copy()
    colour = _background_colour(image, source_region, record)

    tx0, ty0, tx1, ty1 = _pad_box(target_region_box, 0, shape)
    patch_height, patch_width = patch.shape[:2]
    centre_x = int(tx0 + anchor[0] * (tx1 - tx0))
    centre_y = int(ty0 + anchor[1] * (ty1 - ty0))
    left = int(np.clip(centre_x - patch_width // 2, tx0, max(tx0, tx1 - patch_width)))
    top = int(np.clip(centre_y - patch_height // 2, ty0, max(ty0, ty1 - patch_height)))
    edited = _fill(image.copy(), patch_box, colour)
    edited[top:top + patch_height, left:left + patch_width] = patch
    return _finish(image, edited, PixelEdit(
        kind="relocate_entity_to_region",
        description=(
            f"move the topmost observed {entity_type} from the source region "
            "into the target region"
        ),
        boxes=(tuple(box), (left, top, left + patch_width, top + patch_height)),
        source_entity=entity_type,
        extra={"source_region": list(source_region_box),
               "target_region": list(target_region_box)},
    ))


def flip_region_vertical(
    image: np.ndarray,
    record: Mapping[str, Any],
    region_box: Sequence[float],
) -> Optional[EditedImage]:
    """竖直翻转一个隔间的内容：上方实体与下方实体互换位置。"""
    shape = image.shape[:2]
    x0, y0, x1, y1 = _pad_box(region_box, 0, shape)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    edited = image.copy()
    region = image[y0:y1, x0:x1]
    flipped = region[::-1, :, :]
    # 只替换区域内容，保留隔间边框附近 4px，避免破坏容器检测。
    inset = 4
    if y1 - y0 <= 2 * inset or x1 - x0 <= 2 * inset:
        return None
    inner_h = (y1 - y0) - 2 * inset
    edited[y0 + inset:y1 - inset, x0:x1] = flipped[inset:inset + inner_h, :, :]
    return _finish(image, edited, PixelEdit(
        kind="flip_region_vertical",
        description="vertically mirror the contents of one compartment",
        boxes=(tuple(region_box),),
        extra={"region": list(region_box)},
    ))


def separate_entity_horizontally(
    image: np.ndarray,
    record: Mapping[str, Any],
    entity_type: str,
    region_box: Sequence[float],
    *,
    pad: int = 6,
    shift_fraction: float = 0.35,
) -> Optional[EditedImage]:
    """把同类型实例中最下方的一个水平搬开，破坏竖直对齐。"""
    shape = image.shape[:2]
    candidates = [
        row["box_xyxy"] for row in record.get("detections", [])
        if row["entity_type"] == entity_type
    ]
    if len(candidates) < 2:
        return None
    x0, y0, x1, y1 = _pad_box(region_box, 0, shape)
    box = max(candidates, key=lambda row: row[3])
    patch_box = _pad_box(box, pad, shape)
    patch = image[patch_box[1]:patch_box[3], patch_box[0]:patch_box[2]].copy()
    colour = _background_colour(image, (x0, y0, x1, y1), record)
    patch_height, patch_width = patch.shape[:2]
    left = int(np.clip(
        patch_box[0] + shift_fraction * (x1 - x0), x0, max(x0, x1 - patch_width)
    ))
    top = int(np.clip(patch_box[1], y0, max(y0, y1 - patch_height)))
    edited = _fill(image.copy(), patch_box, colour)
    edited[top:top + patch_height, left:left + patch_width] = patch
    return _finish(image, edited, PixelEdit(
        kind="separate_entity_horizontally",
        description=f"shift the lowest {entity_type} horizontally inside its region",
        boxes=(tuple(box), (left, top, left + patch_width, top + patch_height)),
        source_entity=entity_type,
        extra={"region": list(region_box)},
    ))


# ---------------------------------------------------------------------------
# Independent visual confirmation
# ---------------------------------------------------------------------------


def _entity_stats(
    record: Mapping[str, Any], entity_type: str, region_box: Optional[Sequence[float]]
) -> Tuple[int, float, float]:
    """Return ``(count, mean_x, mean_y)`` for an entity, optionally in a region."""
    shape = (int(record["image_size"][1]), int(record["image_size"][0]))
    region = None
    if region_box is not None:
        region = _rect_mask(shape, _pad_box(region_box, 0, shape)) > 0
    xs, ys, count = [], [], 0
    for row in record.get("detections", []):
        if row["entity_type"] != entity_type:
            continue
        x0, y0, x1, y1 = row["box_xyxy"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if region is not None:
            xi = int(np.clip(cx, 0, shape[1] - 1))
            yi = int(np.clip(cy, 0, shape[0] - 1))
            if not region[yi, xi]:
                continue
        count += 1
        xs.append(cx)
        ys.append(cy)
    if not xs:
        return 0, 0.0, 0.0
    return count, float(np.mean(xs)), float(np.mean(ys))


def confirm_intervention(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    edit: PixelEdit,
) -> Tuple[bool, Dict[str, Any]]:
    """Independently verify that the intended semantic change is observable.

    The checker consumes detector output only.  It never looks at rule statuses,
    so a rule cannot certify its own counterfactual.
    """
    evidence: Dict[str, Any] = {"edit": edit.to_dict()}
    before_complete = bool(before.get("detections"))
    after_complete = bool(after.get("detections"))
    evidence["detections_present_before"] = before_complete
    evidence["detections_present_after"] = after_complete
    if not before_complete or not after_complete:
        return False, evidence
    # The container must still be recoverable, otherwise the edit destroyed the
    # scene and the pair says nothing about the semantic change.  Use exactly
    # the same container resolution as fact building (detector box or the
    # brightness silhouette), never a rule-derived geometry.
    container = "bento_box"
    before_box, before_conf, _, before_source = select_container(
        batch_from_record(before).detections,
        before.get("foreground_box"),
        float(before.get("foreground_confidence") or 0.0),
        (int(before["image_size"][0]), int(before["image_size"][1])),
        container,
    )
    after_box, after_conf, _, after_source = select_container(
        batch_from_record(after).detections,
        after.get("foreground_box"),
        float(after.get("foreground_confidence") or 0.0),
        (int(after["image_size"][0]), int(after["image_size"][1])),
        container,
    )
    evidence["container_before"] = {
        "box": list(before_box), "confidence": before_conf, "source": before_source,
    }
    evidence["container_after"] = {
        "box": list(after_box), "confidence": after_conf, "source": after_source,
    }
    if not (before_conf > 0.0 and after_conf > 0.0):
        return False, evidence

    kind = edit.kind
    if kind == "empty_region":
        region = edit.extra["region"]
        present_before = {}
        for row in before.get("detections", []):
            entity = row["entity_type"]
            if entity == container:
                continue
            count, _, _ = _entity_stats(before, entity, region)
            if count:
                present_before[entity] = count
        after_counts = {
            entity: _entity_stats(after, entity, region)[0]
            for entity in present_before
        }
        evidence.update({
            "entities_present_before": present_before,
            "counts_after": after_counts,
        })
        confirmed = bool(present_before) and all(
            value == 0 for value in after_counts.values()
        )
        return bool(confirmed), evidence

    if kind == "erase_entity":
        entity = edit.source_entity
        before_count, _, _ = _entity_stats(before, entity, None)
        after_count, _, _ = _entity_stats(after, entity, None)
        evidence.update({
            "entity_type": entity,
            "count_before": before_count, "count_after": after_count,
        })
        confirmed = before_count >= 1 and after_count < before_count
        return bool(confirmed), evidence

    if kind == "relocate_entity_to_region":
        entity = edit.source_entity
        source_region = edit.extra["source_region"]
        target_region = edit.extra["target_region"]
        before_source, _, _ = _entity_stats(before, entity, source_region)
        after_source, _, _ = _entity_stats(after, entity, source_region)
        before_target, _, _ = _entity_stats(before, entity, target_region)
        after_target, _, _ = _entity_stats(after, entity, target_region)
        evidence.update({
            "entity_type": entity,
            "source_count_before": before_source,
            "source_count_after": after_source,
            "target_count_before": before_target,
            "target_count_after": after_target,
        })
        confirmed = (
            after_source < before_source and after_target > before_target
        )
        return bool(confirmed), evidence

    if kind == "flip_region_vertical":
        region = edit.extra["region"]
        shapes = sorted({
            row["entity_type"] for row in before.get("detections", [])
            if row["entity_type"] not in {container}
        })
        if edit.target_entity:
            shapes = [edit.target_entity]
        rows = []
        for entity in shapes:
            _, before_x, before_y = _entity_stats(before, entity, region)
            _, after_x, after_y = _entity_stats(after, entity, region)
            rows.append({
                "entity_type": entity,
                "mean_y_before": before_y, "mean_y_after": after_y,
            })
        evidence["entity_y_positions"] = rows

        def centroid(record, entity):
            count, _, y = _entity_stats(record, entity, region)
            return (y if count else None)

        pair = edit.extra.get("comparison_pair")
        if pair:
            first, second = pair
            before_first, before_second = centroid(before, first), centroid(before, second)
            after_first, after_second = centroid(after, first), centroid(after, second)
            evidence.update({
                "comparison_pair": [first, second],
                "before_order": [before_first, before_second],
                "after_order": [after_first, after_second],
            })
            if None in (before_first, before_second, after_first, after_second):
                return False, evidence
            confirmed = (
                before_first < before_second and after_first > after_second
            )
            return bool(confirmed), evidence
        # Without an explicit comparison pair, require that every entity moved.
        confirmed = bool(rows) and all(
            abs(row["mean_y_after"] - row["mean_y_before"]) > 1.0 for row in rows
        )
        return bool(confirmed), evidence

    if kind == "separate_entity_horizontally":
        entity = edit.source_entity
        region = edit.extra["region"]
        shape = (int(before["image_size"][1]), int(before["image_size"][0]))
        mask = _rect_mask(shape, _pad_box(region, 0, shape)) > 0
        def xs(record):
            values = []
            for row in record.get("detections", []):
                if row["entity_type"] != entity:
                    continue
                x0, y0, x1, y1 = row["box_xyxy"]
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                xi = int(np.clip(cx, 0, shape[1] - 1))
                yi = int(np.clip(cy, 0, shape[0] - 1))
                if mask[yi, xi]:
                    values.append(cx)
            return values
        before_xs, after_xs = xs(before), xs(after)
        before_spread = (max(before_xs) - min(before_xs)) if len(before_xs) > 1 else 0.0
        after_spread = (max(after_xs) - min(after_xs)) if len(after_xs) > 1 else 0.0
        evidence.update({
            "entity_type": entity,
            "x_centres_before": before_xs, "x_centres_after": after_xs,
            "x_spread_before": before_spread, "x_spread_after": after_spread,
        })
        confirmed = (
            len(after_xs) > 1 and after_spread > before_spread + 1.0
        )
        return bool(confirmed), evidence

    evidence["reason"] = "unknown_intervention_kind"
    return False, evidence


__all__ = [
    "PixelEdit", "EditedImage", "empty_region", "erase_entity",
    "relocate_entity_to_region",
    "flip_region_vertical", "separate_entity_horizontally",
    "confirm_intervention",
]
