"""MVTec LOCO saturated per-region overlap（sPRO）定位评价。

This module follows the dataset paper's protocol:

* each non-zero GT gray value identifies a defect type;
* its saturation threshold is the absolute/relative value in Appendix D;
* sPRO is averaged over connected GT regions, not over pixels;
* the reported score is normalized area under the sPRO/FPR curve to 5% FPR;
* logical and structural defects are evaluated separately, with good images
  included in both subsets for false-positive measurement.

GT 只允许进入本评价模块；任何推理 API 都不接收 GT mask，从接口上隔断诊断标签
泄漏。输入 map 会 resize 回原始 GT 分辨率，因为官方绝对饱和阈值以原图像素计数。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class SaturationSpec:
    absolute_pixels: Optional[float] = None
    relative_area: Optional[float] = None

    def threshold(self, region_area: int) -> float:
        if self.absolute_pixels is not None:
            return float(min(max(self.absolute_pixels, 1.0), region_area))
        factor = 1.0 if self.relative_area is None else self.relative_area
        return float(min(max(factor * region_area, 1.0), region_area))


def _absolute(value: float) -> SaturationSpec:
    return SaturationSpec(absolute_pixels=float(value))


def _relative(value: float = 1.0) -> SaturationSpec:
    return SaturationSpec(relative_area=float(value))


# Values are transcribed from Tables 11--15 of the LOCO paper.  Structural
# defects all use relative_area=1, i.e. ordinary PRO.  The table is keyed by
# original mask gray value, so GT masks must be loaded without binarization.
DEFAULT_SATURATION_TABLE: Dict[str, Dict[int, SaturationSpec]] = {
    "breakfast_box": {
        **{value: _relative() for value in range(234, 244)},
        244: _absolute(268700), 245: _absolute(184400),
        246: _absolute(200200), 247: _absolute(100100),
        248: _absolute(100100), 249: _absolute(84300),
        250: _absolute(100100),
        **{value: _relative() for value in range(251, 256)},
    },
    "screw_bag": {
        **{value: _relative() for value in range(236, 240)},
        240: _absolute(11000), 241: _absolute(5500),
        242: _absolute(15000), 243: _absolute(7500),
        244: _absolute(20600), 245: _absolute(29600),
        246: _absolute(11000), 247: _absolute(5500),
        248: _absolute(15000), 249: _absolute(7500),
        250: _absolute(20600), 251: _absolute(29600),
        252: _absolute(18000), 253: _relative(),
        254: _absolute(9000), 255: _absolute(9000),
    },
    "pushpins": {
        **{value: _relative() for value in range(248, 253)},
        253: _absolute(6300), 254: _absolute(12600),
        255: _absolute(6300),
    },
    "splicing_connectors": {
        **{value: _relative() for value in range(235, 240)},
        240: _absolute(18000), 241: _relative(), 242: _relative(),
        243: _relative(), 244: _relative(), 245: _absolute(18000),
        246: _relative(0.5), 247: _absolute(18000),
        248: _absolute(103600), 249: _relative(),
        250: _absolute(120000), 251: _absolute(76100),
        252: _absolute(53300), 253: _absolute(22100),
        254: _absolute(44300), 255: _absolute(67100),
    },
    "juice_bottle": {
        **{value: _relative() for value in range(238, 249)},
        249: _absolute(32700), 250: _absolute(70400),
        251: _relative(), 252: _relative(),
        253: _absolute(140800), 254: _absolute(32700),
        255: _absolute(70400),
        240: _absolute(4500),
    },
}


@dataclass(frozen=True)
class LocoEvaluationSample:
    sample_id: str
    category: str
    anomaly_type: str  # good | logical | structural
    ground_truth: np.ndarray
    maps: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class SproCurve:
    false_positive_rates: np.ndarray
    saturated_overlaps: np.ndarray
    normalized_area: float
    maximum_fpr: float
    region_count: int


class MapFrame(str, Enum):
    """Coordinate frame carried by an anomaly map; array size is not a frame."""

    ORIGINAL = "original"
    FULL_IMAGE_NORMALIZED = "full_image_normalized"
    MODEL_CENTER_CROP = "model_center_crop"


@dataclass(frozen=True)
class ScoreMapGeometry:
    frame: MapFrame
    input_size: int = 448
    crop_size: int = 392
    outside_value: float = 0.0


def _regions(
    category: str,
    ground_truth: np.ndarray,
    saturation_table: Mapping[str, Mapping[int, SaturationSpec]],
) -> Tuple[Tuple[np.ndarray, float], ...]:
    gt = np.asarray(ground_truth)
    if gt.ndim == 3:
        gt = gt[..., 0]
    table = saturation_table.get(category)
    if table is None:
        raise KeyError(f"no LOCO saturation table for category {category!r}")
    result = []
    # 不可先把 GT 二值化：同一图中的灰度值编码不同缺陷类型和饱和阈值。
    for gray_value in sorted(int(value) for value in np.unique(gt) if value > 0):
        spec = table.get(gray_value)
        if spec is None:
            raise KeyError(
                f"no saturation threshold for {category} GT value {gray_value}"
            )
        count, labels = cv2.connectedComponents(
            (gt == gray_value).astype(np.uint8), connectivity=8
        )
        for component in range(1, count):
            region = labels == component
            area = int(np.count_nonzero(region))
            if area:
                result.append((region, spec.threshold(area)))
    return tuple(result)


def project_score_map_to_original(
    score_map: np.ndarray,
    shape: Tuple[int, int],
    geometry: Optional[ScoreMapGeometry] = None,
) -> np.ndarray:
    """Project a declared map frame to raw-image coordinates.

    A resized map with missing geometry fails closed.  This prevents a
    center-crop INP map from being silently stretched over pixels the model
    never observed, and prevents direct fusion with full-image rule maps.
    """
    values = np.asarray(score_map, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("anomaly maps must be two-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("anomaly maps must contain only finite values")
    if geometry is None:
        if values.shape != shape:
            raise ValueError(
                "map geometry is required when map and original shapes differ"
            )
        geometry = ScoreMapGeometry(MapFrame.ORIGINAL)
    frame = MapFrame(geometry.frame)
    if frame == MapFrame.ORIGINAL:
        if values.shape != shape:
            raise ValueError("ORIGINAL-frame map shape must equal ground truth shape")
        return values
    if frame == MapFrame.FULL_IMAGE_NORMALIZED:
        return cv2.resize(values, shape[::-1], interpolation=cv2.INTER_LINEAR)
    if frame != MapFrame.MODEL_CENTER_CROP:
        raise ValueError(f"unsupported map frame: {frame}")
    input_size = int(geometry.input_size)
    crop_size = int(geometry.crop_size)
    if input_size <= 0 or crop_size <= 0 or crop_size > input_size:
        raise ValueError("center-crop geometry must satisfy 0 < crop <= input")
    offset = (input_size - crop_size) // 2
    if offset * 2 + crop_size != input_size:
        raise ValueError("center-crop inverse currently requires symmetric integer padding")
    crop = cv2.resize(values, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    canvas = np.full(
        (input_size, input_size), float(geometry.outside_value), dtype=np.float32
    )
    canvas[offset:offset + crop_size, offset:offset + crop_size] = crop
    valid = np.zeros((input_size, input_size), dtype=np.uint8)
    valid[offset:offset + crop_size, offset:offset + crop_size] = 1
    restored = cv2.resize(canvas, shape[::-1], interpolation=cv2.INTER_LINEAR)
    restored_valid = cv2.resize(valid, shape[::-1], interpolation=cv2.INTER_NEAREST)
    restored[restored_valid == 0] = float(geometry.outside_value)
    return restored


def fuse_in_original_frame(
    left: np.ndarray,
    left_geometry: ScoreMapGeometry,
    right: np.ndarray,
    right_geometry: ScoreMapGeometry,
    shape: Tuple[int, int],
) -> np.ndarray:
    """Probability-OR fusion after, never before, coordinate projection."""
    lhs = np.clip(project_score_map_to_original(left, shape, left_geometry), 0.0, 1.0)
    rhs = np.clip(project_score_map_to_original(right, shape, right_geometry), 0.0, 1.0)
    return (1.0 - (1.0 - lhs) * (1.0 - rhs)).astype(np.float32)


def _curve_thresholds(normal_scores: np.ndarray, maximum_fpr: float, count: int) -> np.ndarray:
    # Quantiles concentrate thresholds exactly in the FPR range that is
    # integrated.  End points above max and below the target quantile ensure
    # interpolation at FPR=0 and FPR=maximum_fpr.
    quantiles = np.linspace(1.0, 1.0 - maximum_fpr, max(int(count), 2))
    try:
        values = np.quantile(normal_scores, quantiles, method="higher")
    except TypeError:  # NumPy < 1.22
        values = np.quantile(normal_scores, quantiles, interpolation="higher")
    epsilon = max(float(np.ptp(normal_scores)) * 1e-7, 1e-12)
    return np.unique(np.concatenate([
        [float(np.max(normal_scores)) + epsilon], values,
        [float(np.min(values)) - epsilon],
    ]))[::-1]


def _strict_counts_above(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Count ``value > threshold`` for all ascending thresholds in one pass."""
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.ndim != 1 or np.any(thresholds[1:] < thresholds[:-1]):
        raise ValueError("thresholds must be a one-dimensional ascending array")
    flat = np.asarray(values, dtype=np.float32).ravel()
    ranks = np.searchsorted(thresholds, flat, side="left")
    histogram = np.bincount(ranks, minlength=len(thresholds) + 1)
    suffix = np.cumsum(histogram[::-1], dtype=np.int64)[::-1]
    return suffix[1:]


def _upper_tail_thresholds(
    samples: Sequence[LocoEvaluationSample],
    map_name: str,
    geometry: Optional[ScoreMapGeometry],
    maximum_fpr: float,
    count: int,
) -> Tuple[np.ndarray, int]:
    """Collect exactly the global upper FPR tail without materializing all pixels."""
    negative_count = int(sum(
        np.count_nonzero(np.asarray(sample.ground_truth)[..., 0] == 0)
        if np.asarray(sample.ground_truth).ndim == 3
        else np.count_nonzero(np.asarray(sample.ground_truth) == 0)
        for sample in samples
    ))
    if negative_count <= 0:
        raise ValueError("no non-defect pixels available for FPR")
    first_rank = int(np.ceil((1.0 - maximum_fpr) * (negative_count - 1)))
    keep_count = negative_count - first_rank
    tail = np.empty(0, dtype=np.float32)
    for sample in samples:
        gt = np.asarray(sample.ground_truth)
        if gt.ndim == 3:
            gt = gt[..., 0]
        score = project_score_map_to_original(
            sample.maps[map_name], gt.shape, geometry
        )
        tail = np.concatenate((tail, score[gt == 0].astype(np.float32, copy=False)))
        if tail.size > keep_count:
            boundary = tail.size - keep_count
            tail = np.partition(tail, boundary)[boundary:]
    tail.sort()
    quantiles = np.linspace(1.0, 1.0 - maximum_fpr, max(int(count), 2))
    global_ranks = np.ceil(quantiles * (negative_count - 1)).astype(np.int64)
    local_ranks = np.clip(global_ranks - first_rank, 0, tail.size - 1)
    values = tail[local_ranks]
    thresholds = np.unique(np.concatenate((
        values,
        [np.nextafter(tail[-1], np.float32(np.inf))],
        [np.nextafter(tail[0], np.float32(-np.inf))],
    )).astype(np.float32))
    return thresholds, negative_count


def evaluate_spro(
    samples: Sequence[LocoEvaluationSample],
    map_name: str,
    anomaly_type: str,
    *,
    maximum_fpr: float = 0.05,
    threshold_count: int = 1000,
    saturation_table: Mapping[str, Mapping[int, SaturationSpec]] = DEFAULT_SATURATION_TABLE,
    geometry: Optional[ScoreMapGeometry] = None,
) -> SproCurve:
    """在 good + 指定异常类型子集上评价一个 map family。"""
    if anomaly_type not in {"logical", "structural"}:
        raise ValueError("anomaly_type must be 'logical' or 'structural'")
    # logical/structural 分开测，但两者都带上 good 图以计算真实像素 FPR。
    subset = [
        sample for sample in samples
        if sample.anomaly_type in {"good", anomaly_type}
    ]
    if not subset:
        raise ValueError("evaluation subset is empty")
    region_count = 0
    for sample in subset:
        gt = np.asarray(sample.ground_truth)
        if gt.ndim == 3:
            gt = gt[..., 0]
        if map_name not in sample.maps:
            raise KeyError(f"sample {sample.sample_id} has no map {map_name!r}")
        if sample.anomaly_type != "good":
            region_count += len(_regions(sample.category, gt, saturation_table))
    if not region_count:
        raise ValueError(f"no {anomaly_type} GT regions were found")
    thresholds, negative_count = _upper_tail_thresholds(
        subset, map_name, geometry, maximum_fpr, threshold_count
    )
    false_positives = np.zeros(len(thresholds), dtype=np.int64)
    overlap_sum = np.zeros(len(thresholds), dtype=np.float64)
    for sample in subset:
        gt = np.asarray(sample.ground_truth)
        if gt.ndim == 3:
            gt = gt[..., 0]
        score = project_score_map_to_original(
            sample.maps[map_name], gt.shape, geometry
        )
        false_positives += _strict_counts_above(score[gt == 0], thresholds)
        if sample.anomaly_type == "good":
            continue
        for region, saturation in _regions(sample.category, gt, saturation_table):
            hits = _strict_counts_above(score[region], thresholds)
            overlap_sum += np.minimum(hits.astype(np.float64) / saturation, 1.0)
    fprs = false_positives.astype(np.float64) / float(negative_count)
    overlaps = overlap_sum / float(region_count)
    order = np.argsort(fprs, kind="stable")
    fprs, overlaps = fprs[order], overlaps[order]
    # Collapse duplicate FPRs conservatively to their best achievable sPRO.
    unique_fpr = np.unique(fprs)
    unique_overlap = np.asarray([
        float(np.max(overlaps[fprs == value])) for value in unique_fpr
    ])
    if unique_fpr[-1] < maximum_fpr:
        unique_fpr = np.append(unique_fpr, maximum_fpr)
        unique_overlap = np.append(unique_overlap, unique_overlap[-1])
    else:
        boundary = float(np.interp(maximum_fpr, unique_fpr, unique_overlap))
        keep = unique_fpr < maximum_fpr
        unique_fpr = np.append(unique_fpr[keep], maximum_fpr)
        unique_overlap = np.append(unique_overlap[keep], boundary)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area = float(trapezoid(unique_overlap, unique_fpr) / maximum_fpr)
    return SproCurve(
        unique_fpr.astype(np.float32), unique_overlap.astype(np.float32),
        float(np.clip(area, 0.0, 1.0)), maximum_fpr, region_count,
    )


def evaluate_map_families(
    samples: Sequence[LocoEvaluationSample],
    *,
    maximum_fpr: float = 0.05,
    threshold_count: int = 1000,
    saturation_table: Mapping[str, Mapping[int, SaturationSpec]] = DEFAULT_SATURATION_TABLE,
    map_geometries: Optional[Mapping[str, ScoreMapGeometry]] = None,
) -> Dict[str, Dict[str, float]]:
    """用同一协议分别评价逐规则图、总逻辑图和最终融合图。"""
    names = sorted({name for sample in samples for name in sample.maps})
    output = {}
    for name in names:
        logical = evaluate_spro(
            samples, name, "logical", maximum_fpr=maximum_fpr,
            threshold_count=threshold_count, saturation_table=saturation_table,
            geometry=None if map_geometries is None else map_geometries.get(name),
        )
        structural = evaluate_spro(
            samples, name, "structural", maximum_fpr=maximum_fpr,
            threshold_count=threshold_count, saturation_table=saturation_table,
            geometry=None if map_geometries is None else map_geometries.get(name),
        )
        output[name] = {
            "logical_spro_5fpr": logical.normalized_area,
            "structural_spro_5fpr": structural.normalized_area,
            "mean_spro_5fpr": (
                logical.normalized_area + structural.normalized_area
            ) / 2.0,
            "logical_region_count": logical.region_count,
            "structural_region_count": structural.region_count,
        }
    return output


__all__ = [
    "SaturationSpec", "DEFAULT_SATURATION_TABLE", "LocoEvaluationSample",
    "SproCurve", "MapFrame", "ScoreMapGeometry",
    "project_score_map_to_original", "fuse_in_original_frame",
    "evaluate_spro", "evaluate_map_families",
]
