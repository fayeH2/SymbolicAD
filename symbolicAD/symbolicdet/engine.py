"""CLIP-feature segmentation and a frozen dynamic-expression executor.

The model is *compiled* from normal support CLIP patch features.  Query-time
code only assigns CLIP patches to the frozen visual prototypes and evaluates
the frozen expressions.  No LLM and no query-set adaptation exist here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image

from symbolicAD.config import SymbolicDetConfig


STATIC_SYMBOLICDET_VERSION = "clip_patch_static_v1"


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


def _rgb_array(image, height: int, width: int) -> np.ndarray:
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def _lab01(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0


def _normalise_patches(patches: np.ndarray) -> np.ndarray:
    patches = np.asarray(patches, dtype=np.float32)
    if patches.ndim != 3:
        raise ValueError(f"expected [D,H,W] CLIP patches, got {patches.shape}")
    vectors = patches.transpose(1, 2, 0)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=2, keepdims=True), 1e-8)


def _fit_kmeans(values: np.ndarray, clusters: int, iterations: int = 20) -> np.ndarray:
    """Deterministic cosine k-means for normal CLIP patch prototypes."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("visual prototype fitting requires non-empty [N,D] features")
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    clusters = max(1, min(int(clusters), values.shape[0]))
    centres = [values[0]]
    for _ in range(1, clusters):
        similarity = np.max(values @ np.stack(centres, axis=0).T, axis=1)
        centres.append(values[int(np.argmin(similarity))])
    centres_array = np.stack(centres, axis=0)
    for _ in range(iterations):
        labels = np.argmax(values @ centres_array.T, axis=1)
        updated = centres_array.copy()
        for index in range(clusters):
            members = values[labels == index]
            if members.size:
                centre = members.mean(axis=0)
                updated[index] = centre / max(float(np.linalg.norm(centre)), 1e-8)
        if np.allclose(updated, centres_array, atol=1e-5):
            break
        centres_array = updated
    return centres_array.astype(np.float32)


def _mask_geometry(mask: np.ndarray) -> Dict[str, float]:
    ys, xs = np.where(mask)
    height, width = mask.shape
    if xs.size == 0:
        return {"area": 0.0, "cx": 0.5, "cy": 0.5, "components": 0.0}
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), 8)
    return {
        "area": float(mask.mean()),
        "cx": float((xs.mean() + 0.5) / width),
        "cy": float((ys.mean() + 0.5) / height),
        "components": float(max(count - 1, 0)),
    }


class SegmentFirstSymbolicDet:
    """Frozen static expressions grounded in CLIP patch segmentation."""

    FAMILIES = ("appearance", "structure", "relation")

    def __init__(self, config: SymbolicDetConfig):
        self.config = config
        self.visual_prototypes = np.zeros((0, 0), dtype=np.float32)
        self.expressions: List[DynamicExpression] = []
        self.expected_masks: Dict[str, np.ndarray] = {}
        self.expert_weights: Dict[str, float] = {}
        self.support_count = 0
        self.grid_size = (0, 0)

    def _observe(self, image, patch_features: np.ndarray) -> _Observation:
        vectors = _normalise_patches(patch_features)
        height, width, channels = vectors.shape
        if self.visual_prototypes.shape[1:] != (channels,):
            raise ValueError(
                f"CLIP feature dimension {channels} does not match static program "
                f"{self.visual_prototypes.shape}"
            )
        if self.grid_size != (height, width):
            raise ValueError(
                f"CLIP grid {(height, width)} does not match static program {self.grid_size}"
            )
        labels = np.argmax(vectors.reshape(-1, channels) @ self.visual_prototypes.T, axis=1)
        labels = labels.reshape(height, width)
        rgb = _rgb_array(image, height, width)
        lab = _lab01(rgb)
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(grey, 60, 160) > 0

        features: Dict[str, float] = {}
        masks: Dict[str, np.ndarray] = {}
        cluster_masks: Dict[int, np.ndarray] = {}
        for index in range(self.visual_prototypes.shape[0]):
            mask = labels == index
            cluster_masks[index] = mask
            geometry = _mask_geometry(mask)
            for name, value in geometry.items():
                key = f"cluster.{index}.{name}"
                features[key] = value
                masks[key] = mask.astype(np.float32)
            for channel, name in enumerate(("L", "a", "b")):
                key = f"cluster.{index}.{name}"
                features[key] = float(lab[..., channel][mask].mean()) if mask.any() else float("nan")
                masks[key] = mask.astype(np.float32)
            key = f"cluster.{index}.edge"
            features[key] = float(edges[mask].mean()) if mask.any() else 0.0
            masks[key] = mask.astype(np.float32)

        rows, cols = int(self.config.grid_rows), int(self.config.grid_cols)
        for row in range(rows):
            y0, y1 = row * height // rows, (row + 1) * height // rows
            for col in range(cols):
                x0, x1 = col * width // cols, (col + 1) * width // cols
                cell = np.zeros((height, width), dtype=bool)
                cell[y0:y1, x0:x1] = True
                for index, cluster_mask in cluster_masks.items():
                    region = cell & cluster_mask
                    key = f"cell.{row}.{col}.cluster.{index}.occupancy"
                    features[key] = float(region.sum() / max(cell.sum(), 1))
                    masks[key] = region.astype(np.float32)

        for first in range(self.visual_prototypes.shape[0]):
            first_mask = cluster_masks[first]
            first_stats = _mask_geometry(first_mask)
            if first_stats["area"] <= 0.0:
                continue
            for second in range(first + 1, self.visual_prototypes.shape[0]):
                second_mask = cluster_masks[second]
                second_stats = _mask_geometry(second_mask)
                if second_stats["area"] <= 0.0:
                    continue
                prefix = f"relation.{first}.{second}"
                dx = second_stats["cx"] - first_stats["cx"]
                dy = second_stats["cy"] - first_stats["cy"]
                pair_mask = (first_mask | second_mask).astype(np.float32)
                for name, value in (
                    ("dx", dx),
                    ("dy", dy),
                    ("distance", float(np.hypot(dx, dy))),
                ):
                    key = f"{prefix}.{name}"
                    features[key] = float(value)
                    masks[key] = pair_mask
        return _Observation(features=features, masks=masks)

    @staticmethod
    def _feature_family(key: str) -> str:
        if key.startswith("relation."):
            return "relation"
        if key.endswith((".L", ".a", ".b", ".edge")):
            return "appearance"
        return "structure"

    def _margin(self, key: str, values: np.ndarray) -> float:
        if key.endswith((".L", ".a", ".b", ".edge")):
            floor = float(self.config.color_margin)
        elif key.endswith((".cx", ".cy", ".dx", ".dy", ".distance")):
            floor = float(self.config.position_margin)
        elif key.endswith(".components"):
            floor = float(self.config.count_margin)
        else:
            floor = float(self.config.occupancy_margin)
        return max(
            floor,
            abs(float(np.median(values))) * float(self.config.relative_margin),
            1e-4,
        )

    def fit(
        self,
        support_images: Sequence,
        support_patch_features: np.ndarray,
    ) -> "SegmentFirstSymbolicDet":
        patches = np.asarray(support_patch_features, dtype=np.float32)
        if not support_images or patches.ndim != 4 or patches.shape[0] != len(support_images):
            raise ValueError("fit requires aligned normal images and [K,D,H,W] CLIP features")
        self.support_count = len(support_images)
        self.grid_size = (int(patches.shape[2]), int(patches.shape[3]))
        vectors = np.concatenate(
            [_normalise_patches(item).reshape(-1, patches.shape[1]) for item in patches],
            axis=0,
        )
        self.visual_prototypes = _fit_kmeans(vectors, int(self.config.visual_prototypes))
        observations = [
            self._observe(image, patches[index])
            for index, image in enumerate(support_images)
        ]
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
            confidence = float(np.exp(-span / max(2.0 * margin, 1e-6)))
            if confidence < float(self.config.min_expression_confidence):
                continue
            self.expressions.append(
                DynamicExpression(
                    expression_id=f"D{index:03d}",
                    family=self._feature_family(key),
                    feature=key,
                    lower=float(values.min() - margin),
                    upper=float(values.max() + margin),
                    scale=float(margin),
                    confidence=confidence,
                    support_values=[float(value) for value in values],
                )
            )
            self.expected_masks[key] = np.mean(
                np.stack([item.masks[key] for item in observations]), axis=0
            ).astype(np.float32)

        base = {
            "appearance": float(self.config.appearance_weight),
            "structure": float(self.config.structure_weight),
            "relation": float(self.config.relation_weight),
        }
        raw = {}
        for family in self.FAMILIES:
            confidences = [item.confidence for item in self.expressions if item.family == family]
            raw[family] = base[family] * (float(np.mean(confidences)) if confidences else 0.0)
        total = sum(raw.values())
        self.expert_weights = {
            family: (raw[family] / total if total > 0.0 else 0.0)
            for family in self.FAMILIES
        }
        return self

    def _expert_output(self, family: str, observation: _Observation) -> ExpertOutput:
        anomaly_map = np.zeros(self.grid_size, dtype=np.float32)
        violations: List[Dict[str, object]] = []
        for expression in self.expressions:
            if expression.family != family or expression.feature not in observation.features:
                continue
            value = float(observation.features[expression.feature])
            severity = expression.violation(value, float(self.config.violation_temperature))
            severity *= expression.confidence
            if severity <= 0.0:
                continue
            current = observation.masks.get(expression.feature)
            expected = self.expected_masks.get(expression.feature)
            if current is None or expected is None:
                continue
            anomaly_map = np.maximum(anomaly_map, severity * np.maximum(current, expected))
            violations.append(
                {
                    "expression_id": expression.expression_id,
                    "family": family,
                    "expression": expression.text,
                    "feature": expression.feature,
                    "value": value,
                    "severity": float(severity),
                }
            )
        violations.sort(key=lambda item: float(item["severity"]), reverse=True)
        return ExpertOutput(
            family=family,
            score=max((float(item["severity"]) for item in violations), default=0.0),
            anomaly_map=anomaly_map,
            violations=violations[:12],
        )

    def infer(self, image, patch_features: np.ndarray) -> SymbolicDetOutput:
        if not self.expressions or not self.visual_prototypes.size:
            raise RuntimeError("static SymbolicDet program is not loaded")
        observation = self._observe(image, patch_features)
        outputs = {family: self._expert_output(family, observation) for family in self.FAMILIES}
        ranked = sorted(
            self.FAMILIES,
            key=lambda family: (outputs[family].score, self.expert_weights.get(family, 0.0)),
            reverse=True,
        )
        keep = set(ranked[: max(1, int(self.config.expert_top_k))])
        raw = {
            family: (
                self.expert_weights.get(family, 0.0) * (0.25 + 0.75 * outputs[family].score)
                if family in keep
                else 0.0
            )
            for family in self.FAMILIES
        }
        total = sum(raw.values())
        routed = {
            family: (raw[family] / total if total > 0.0 else 0.0)
            for family in self.FAMILIES
        }
        anomaly_map = np.zeros(self.grid_size, dtype=np.float32)
        score = 0.0
        violations: List[Dict[str, object]] = []
        for family, weight in routed.items():
            anomaly_map += weight * outputs[family].anomaly_map
            score += weight * outputs[family].score
            for item in outputs[family].violations:
                violations.append({**item, "expert_weight": float(weight)})
        violations.sort(
            key=lambda item: float(item["severity"]) * float(item["expert_weight"]),
            reverse=True,
        )
        selected_mass = sum(
            self.expert_weights.get(family, 0.0) for family, weight in routed.items() if weight > 0
        )
        return SymbolicDetOutput(
            score=float(np.clip(score, 0.0, 1.0)),
            anomaly_map=np.clip(anomaly_map, 0.0, 1.0),
            confidence=float(np.clip(selected_mass, 0.0, 1.0)),
            expert_weights=routed,
            expert_scores={family: float(item.score) for family, item in outputs.items()},
            violations=violations[:20],
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": STATIC_SYMBOLICDET_VERSION,
            "protocol": "compiled_from_normal_support_clip_features",
            "feature_source": "clip_projected_patch_features",
            "support_count": int(self.support_count),
            "grid_size": list(self.grid_size),
            "expert_weights": dict(self.expert_weights),
            "expressions": [{**asdict(item), "text": item.text} for item in self.expressions],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.expected_masks)
        masks = np.stack([self.expected_masks[key] for key in keys], axis=0)
        np.savez_compressed(
            path,
            manifest=np.asarray(json.dumps(self.to_dict(), ensure_ascii=False)),
            visual_prototypes=self.visual_prototypes.astype(np.float32),
            mask_keys=np.asarray(keys),
            expected_masks=masks.astype(np.float16),
        )
        return path

    @classmethod
    def load(cls, path: str | Path, config: SymbolicDetConfig) -> "SegmentFirstSymbolicDet":
        payload = np.load(Path(path), allow_pickle=False)
        manifest = json.loads(str(payload["manifest"].item()))
        if manifest.get("version") != STATIC_SYMBOLICDET_VERSION:
            raise RuntimeError(
                f"incompatible SymbolicDet program {manifest.get('version')}; recompile"
            )
        model = cls(config)
        model.support_count = int(manifest["support_count"])
        model.grid_size = tuple(int(value) for value in manifest["grid_size"])
        model.expert_weights = {
            key: float(value) for key, value in manifest["expert_weights"].items()
        }
        model.expressions = [
            DynamicExpression(**{key: value for key, value in item.items() if key != "text"})
            for item in manifest["expressions"]
        ]
        model.visual_prototypes = payload["visual_prototypes"].astype(np.float32)
        keys = payload["mask_keys"].astype(str).tolist()
        masks = payload["expected_masks"].astype(np.float32)
        model.expected_masks = {key: masks[index] for index, key in enumerate(keys)}
        return model
