"""Support-only segmentation, dynamic expressions and sparse expert routing.

This module deliberately does not inspect a query batch or its labels.  It fits
colour prototypes, normal intervals, expression confidence and MoE gates from
the few normal support images.  The query path only evaluates the frozen model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from symbolicAD.config import SymbolicDetConfig


@dataclass
class DynamicExpression:
    expression_id: str
    family: str
    feature: str
    lower: float
    upper: float
    scale: float
    confidence: float
    support_values: List[float] = field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.lower:.4f} <= {self.feature} <= {self.upper:.4f}"

    def violation(self, value: float, temperature: float = 1.0) -> float:
        if not np.isfinite(value):
            return 0.0
        distance = max(self.lower - value, value - self.upper, 0.0)
        if distance <= 0.0:
            return 0.0
        normalised = distance / max(self.scale * max(temperature, 1.0e-6), 1.0e-6)
        return float(1.0 - np.exp(-normalised))


@dataclass
class ExpertOutput:
    family: str
    score: float
    anomaly_map: np.ndarray
    violations: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class SymbolicDetOutput:
    score: float
    anomaly_map: np.ndarray
    confidence: float
    expert_weights: Dict[str, float]
    expert_scores: Dict[str, float]
    violations: List[Dict[str, object]]

    def trace(self) -> Dict[str, object]:
        return {
            "score": float(self.score),
            "confidence": float(self.confidence),
            "expert_weights": dict(self.expert_weights),
            "expert_scores": dict(self.expert_scores),
            "violations": list(self.violations),
        }


@dataclass
class _Observation:
    features: Dict[str, float]
    masks: Dict[str, np.ndarray]
    foreground: np.ndarray


def _rgb_array(image, size: int) -> np.ndarray:
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def _lab01(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab / 255.0


def _border_pixels(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [values[0], values[-1], values[1:-1, 0], values[1:-1, -1]], axis=0
    )


def _clean_binary(mask: np.ndarray, kernel_size: int, min_area: int) -> np.ndarray:
    binary = mask.astype(np.uint8)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    clean = np.zeros_like(binary)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_area:
            clean[labels == index] = 1
    return clean.astype(bool)


def _fit_kmeans(values: np.ndarray, clusters: int, iterations: int = 16) -> np.ndarray:
    """Small deterministic k-means with farthest-point initialisation."""
    values = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    if values.shape[0] == 0:
        return np.zeros((1, 3), dtype=np.float32)
    clusters = max(1, min(int(clusters), values.shape[0]))
    centres = [np.median(values, axis=0)]
    for _ in range(1, clusters):
        distance = np.min(
            np.stack([np.sum((values - item) ** 2, axis=1) for item in centres]),
            axis=0,
        )
        centres.append(values[int(np.argmax(distance))])
    centres_array = np.asarray(centres, dtype=np.float32)
    for _ in range(iterations):
        distances = np.sum(
            (values[:, None, :] - centres_array[None, :, :]) ** 2, axis=2
        )
        labels = np.argmin(distances, axis=1)
        updated = centres_array.copy()
        for index in range(clusters):
            members = values[labels == index]
            if members.size:
                updated[index] = np.mean(members, axis=0)
        if np.allclose(updated, centres_array, atol=1.0e-5):
            break
        centres_array = updated
    order = np.lexsort(
        (centres_array[:, 2], centres_array[:, 1], centres_array[:, 0])
    )
    return centres_array[order]


def _mask_geometry(mask: np.ndarray) -> Dict[str, float]:
    ys, xs = np.where(mask)
    height, width = mask.shape
    if xs.size == 0:
        return {
            "area": 0.0,
            "cx": 0.5,
            "cy": 0.5,
            "aspect": 1.0,
            "fill": 0.0,
            "components": 0.0,
        }
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    box_area = max((x1 - x0) * (y1 - y0), 1)
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), 8)
    return {
        "area": float(mask.mean()),
        "cx": float((xs.mean() + 0.5) / width),
        "cy": float((ys.mean() + 0.5) / height),
        "aspect": float((x1 - x0) / max(y1 - y0, 1)),
        "fill": float(xs.size / box_area),
        "components": float(max(count - 1, 0)),
    }


class SegmentFirstSymbolicDet:
    """Fit normal dynamic expressions and infer localised violations."""

    FAMILIES = ("appearance", "structure", "relation")

    def __init__(self, config: SymbolicDetConfig):
        self.config = config
        self.color_centres = np.zeros((0, 3), dtype=np.float32)
        self.background_lab = np.zeros(3, dtype=np.float32)
        self.expressions: List[DynamicExpression] = []
        self.expected_masks: Dict[str, np.ndarray] = {}
        self.expert_weights: Dict[str, float] = {}
        self.support_count = 0

    def _foreground(self, lab: np.ndarray) -> np.ndarray:
        distance = np.linalg.norm(lab - self.background_lab[None, None, :], axis=2)
        border_distance = _border_pixels(distance[..., None]).reshape(-1)
        median = float(np.median(border_distance))
        mad = float(np.median(np.abs(border_distance - median)))
        threshold = max(
            float(self.config.foreground_color_distance), median + 4.0 * mad
        )
        min_area = max(
            2,
            int(float(self.config.min_segment_area_ratio) * lab.shape[0] * lab.shape[1]),
        )
        return _clean_binary(
            distance > threshold,
            int(self.config.morphology_kernel),
            min_area,
        )

    def _observe(self, image) -> _Observation:
        size = int(self.config.analysis_size)
        rgb = _rgb_array(image, size)
        lab = _lab01(rgb)
        foreground = self._foreground(lab)
        features: Dict[str, float] = {}
        masks: Dict[str, np.ndarray] = {}

        geometry = _mask_geometry(foreground)
        for name, value in geometry.items():
            key = f"global.{name}"
            features[key] = value
            masks[key] = foreground.astype(np.float32)

        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(grey, 60, 160) > 0
        features["global.edge_density"] = float(edges[foreground].mean()) if foreground.any() else 0.0
        masks["global.edge_density"] = foreground.astype(np.float32)

        rows, cols = int(self.config.grid_rows), int(self.config.grid_cols)
        for row in range(rows):
            y0, y1 = row * size // rows, (row + 1) * size // rows
            for col in range(cols):
                x0, x1 = col * size // cols, (col + 1) * size // cols
                cell = np.zeros((size, size), dtype=bool)
                cell[y0:y1, x0:x1] = True
                region = cell & foreground
                prefix = f"cell.{row}.{col}"
                features[f"{prefix}.occupancy"] = float(region.sum() / max(cell.sum(), 1))
                # Responsibility follows actual foreground. During inference
                # it is unioned with the support expectation, so a missing
                # component still paints where it should be without producing
                # rectangular heat outside the object.
                masks[f"{prefix}.occupancy"] = region.astype(np.float32)
                for channel, name in enumerate(("L", "a", "b")):
                    features[f"{prefix}.{name}"] = (
                        float(lab[..., channel][region].mean()) if region.any() else float("nan")
                    )
                    masks[f"{prefix}.{name}"] = region.astype(np.float32)
                features[f"{prefix}.edge"] = (
                    float(edges[region].mean()) if region.any() else 0.0
                )
                masks[f"{prefix}.edge"] = region.astype(np.float32)

        cluster_masks: Dict[int, np.ndarray] = {}
        if foreground.any() and self.color_centres.size:
            distance = np.sum(
                (lab[:, :, None, :] - self.color_centres[None, None, :, :]) ** 2,
                axis=3,
            )
            labels = np.argmin(distance, axis=2)
            for index in range(self.color_centres.shape[0]):
                mask = foreground & (labels == index)
                cluster_masks[index] = mask
                stats = _mask_geometry(mask)
                for name in ("area", "cx", "cy", "components"):
                    key = f"cluster.{index}.{name}"
                    features[key] = stats[name]
                    masks[key] = mask.astype(np.float32)

        for first in range(self.color_centres.shape[0]):
            first_mask = cluster_masks.get(first, np.zeros_like(foreground))
            first_stats = _mask_geometry(first_mask)
            if first_stats["area"] <= 0.0:
                continue
            for second in range(first + 1, self.color_centres.shape[0]):
                second_mask = cluster_masks.get(second, np.zeros_like(foreground))
                second_stats = _mask_geometry(second_mask)
                if second_stats["area"] <= 0.0:
                    continue
                prefix = f"relation.{first}.{second}"
                dx = second_stats["cx"] - first_stats["cx"]
                dy = second_stats["cy"] - first_stats["cy"]
                features[f"{prefix}.dx"] = float(dx)
                features[f"{prefix}.dy"] = float(dy)
                features[f"{prefix}.distance"] = float(np.hypot(dx, dy))
                pair_mask = (first_mask | second_mask).astype(np.float32)
                masks[f"{prefix}.dx"] = pair_mask
                masks[f"{prefix}.dy"] = pair_mask
                masks[f"{prefix}.distance"] = pair_mask
        return _Observation(features=features, masks=masks, foreground=foreground)

    def _feature_family(self, key: str) -> str:
        if key.startswith("relation."):
            return "relation"
        if key.endswith((".L", ".a", ".b", ".edge")) or key == "global.edge_density":
            return "appearance"
        return "structure"

    def _margin(self, key: str, values: np.ndarray) -> float:
        if key.endswith((".L", ".a", ".b", ".edge")) or key == "global.edge_density":
            floor = float(self.config.color_margin)
        elif key.endswith((".cx", ".cy", ".dx", ".dy", ".distance")):
            floor = float(self.config.position_margin)
        elif key.endswith(".components"):
            floor = float(self.config.count_margin)
        else:
            floor = float(self.config.occupancy_margin)
        relative = abs(float(np.median(values))) * float(self.config.relative_margin)
        return max(floor, relative, 1.0e-4)

    def fit(self, support_images: Sequence) -> "SegmentFirstSymbolicDet":
        if not support_images:
            raise ValueError("SegmentFirstSymbolicDet requires normal support images")
        size = int(self.config.analysis_size)
        labs = [_lab01(_rgb_array(image, size)) for image in support_images]
        border = np.concatenate([_border_pixels(lab) for lab in labs], axis=0)
        self.background_lab = np.median(border, axis=0).astype(np.float32)

        foreground_pixels = []
        for lab in labs:
            mask = self._foreground(lab)
            pixels = lab[mask]
            if pixels.shape[0] > 12000:
                step = max(1, pixels.shape[0] // 12000)
                pixels = pixels[::step]
            if pixels.size:
                foreground_pixels.append(pixels)
        pixels = np.concatenate(foreground_pixels, axis=0) if foreground_pixels else border
        self.color_centres = _fit_kmeans(pixels, int(self.config.color_clusters))

        observations = [self._observe(image) for image in support_images]
        self.support_count = len(observations)
        common = set(observations[0].features)
        for observation in observations[1:]:
            common &= set(observation.features)
        self.expressions = []
        self.expected_masks = {}
        for index, key in enumerate(sorted(common)):
            values = np.asarray([item.features[key] for item in observations], dtype=np.float32)
            if not np.isfinite(values).all():
                continue
            margin = self._margin(key, values)
            span = float(values.max() - values.min())
            confidence = float(np.exp(-span / max(2.0 * margin, 1.0e-6)))
            if confidence < float(self.config.min_expression_confidence):
                continue
            expression = DynamicExpression(
                expression_id=f"D{index:03d}",
                family=self._feature_family(key),
                feature=key,
                lower=float(values.min() - margin),
                upper=float(values.max() + margin),
                scale=float(margin),
                confidence=confidence,
                support_values=[float(value) for value in values],
            )
            self.expressions.append(expression)
            self.expected_masks[key] = np.mean(
                np.stack([item.masks[key] for item in observations]), axis=0
            ).astype(np.float32)

        base = {
            "appearance": float(self.config.appearance_weight),
            "structure": float(self.config.structure_weight),
            "relation": float(self.config.relation_weight),
        }
        quality = {}
        for family in self.FAMILIES:
            values = [item.confidence for item in self.expressions if item.family == family]
            quality[family] = float(np.mean(values)) if values else 0.0
        raw = {family: base[family] * quality[family] for family in self.FAMILIES}
        total = sum(raw.values())
        self.expert_weights = {
            family: (raw[family] / total if total > 0.0 else 0.0)
            for family in self.FAMILIES
        }
        return self

    def _expert_output(self, family: str, observation: _Observation) -> ExpertOutput:
        size = int(self.config.analysis_size)
        anomaly_map = np.zeros((size, size), dtype=np.float32)
        violations: List[Dict[str, object]] = []
        for expression in self.expressions:
            if expression.family != family:
                continue
            value = observation.features.get(expression.feature)
            if value is None:
                continue
            severity = expression.violation(
                float(value), float(self.config.violation_temperature)
            ) * expression.confidence
            if severity <= 0.0:
                continue
            current = observation.masks.get(expression.feature)
            expected = self.expected_masks.get(expression.feature)
            if current is None or expected is None:
                continue
            responsibility = np.maximum(current, expected)
            anomaly_map = np.maximum(anomaly_map, severity * responsibility)
            violations.append(
                {
                    "expression_id": expression.expression_id,
                    "family": family,
                    "expression": expression.text,
                    "feature": expression.feature,
                    "value": float(value),
                    "severity": float(severity),
                }
            )
        violations.sort(key=lambda item: float(item["severity"]), reverse=True)
        score = max((float(item["severity"]) for item in violations), default=0.0)
        return ExpertOutput(family, score, anomaly_map, violations[:12])

    def infer(self, image) -> SymbolicDetOutput:
        if not self.expressions:
            raise RuntimeError("fit() must be called before infer()")
        observation = self._observe(image)
        outputs = {
            family: self._expert_output(family, observation)
            for family in self.FAMILIES
        }
        # Per-image sparse routing is a frozen function of expert evidence and
        # support-fitted priors. It does not inspect other query images. This
        # lets a pure relation violation activate the relation expert even when
        # appearance was the most stable expert on the two support examples.
        ranked = sorted(
            self.FAMILIES,
            key=lambda family: (
                outputs[family].score,
                self.expert_weights.get(family, 0.0),
            ),
            reverse=True,
        )
        keep = set(ranked[: max(1, int(self.config.expert_top_k))])
        routed_raw = {
            family: (
                self.expert_weights.get(family, 0.0)
                * (0.25 + 0.75 * outputs[family].score)
                if family in keep
                else 0.0
            )
            for family in self.FAMILIES
        }
        routed_total = sum(routed_raw.values())
        routed_weights = {
            family: (
                routed_raw[family] / routed_total if routed_total > 0.0 else 0.0
            )
            for family in self.FAMILIES
        }
        size = int(self.config.analysis_size)
        anomaly_map = np.zeros((size, size), dtype=np.float32)
        score = 0.0
        violations: List[Dict[str, object]] = []
        for family, weight in routed_weights.items():
            output = outputs[family]
            anomaly_map += float(weight) * output.anomaly_map
            score += float(weight) * output.score
            for item in output.violations:
                enriched = dict(item)
                enriched["expert_weight"] = float(weight)
                violations.append(enriched)
        violations.sort(
            key=lambda item: float(item["severity"]) * float(item["expert_weight"]),
            reverse=True,
        )
        base_weights = {
            "appearance": float(self.config.appearance_weight),
            "structure": float(self.config.structure_weight),
            "relation": float(self.config.relation_weight),
        }
        selected_mass = sum(
            base_weights[family]
            for family, weight in self.expert_weights.items()
            if weight > 0.0
        )
        confidence = float(np.clip(selected_mass / max(sum(base_weights.values()), 1e-6), 0.0, 1.0))
        return SymbolicDetOutput(
            score=float(np.clip(score, 0.0, 1.0)),
            anomaly_map=np.clip(anomaly_map, 0.0, 1.0),
            confidence=confidence,
            expert_weights=routed_weights,
            expert_scores={family: float(output.score) for family, output in outputs.items()},
            violations=violations[:20],
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": 1,
            "protocol": "normal_support_only",
            "support_count": int(self.support_count),
            "background_lab": self.background_lab.tolist(),
            "color_centres": self.color_centres.tolist(),
            "expert_weights": dict(self.expert_weights),
            "expressions": [
                {**asdict(item), "text": item.text} for item in self.expressions
            ],
        }
