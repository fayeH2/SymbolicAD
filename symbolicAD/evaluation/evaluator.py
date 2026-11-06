"""Accumulate baseline / logic / final metrics per category and group."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

from symbolicAD.config import Config
from symbolicAD.array_store import save_array_bundle

from .image_metrics import image_metrics
from .pixel_metrics import pixel_metrics
from .pro_metric import compute_pro

VARIANTS = ("baseline", "logic", "final")


def _resize_map(values: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] == size and values.shape[1] == size:
        return values
    return cv2.resize(values, (size, size), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape[0] == size and mask.shape[1] == size:
        return mask
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def _group_indices(groups: Sequence[str], target: str) -> np.ndarray:
    groups = np.asarray(groups)
    if target == "all":
        return np.arange(groups.size)
    return np.where(np.isin(groups, [target, "good"]))[0]


@dataclass
class CategoryEvaluator:
    config: Config
    size: int = 0
    labels: List[int] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)
    image_scores: Dict[str, List[float]] = field(default_factory=dict)
    maps: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    masks: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.size <= 0:
            self.size = int(self.config.evaluation.pixel_eval_size)
        for variant in VARIANTS:
            self.image_scores.setdefault(variant, [])
            self.maps.setdefault(variant, [])

    def update(
        self,
        scores: Dict[str, float],
        maps: Dict[str, np.ndarray],
        label: int,
        group: str,
        gt_mask: np.ndarray,
    ) -> None:
        self.labels.append(int(label))
        self.groups.append(str(group))
        self.masks.append(_resize_mask(gt_mask, self.size).astype(np.uint8))
        for variant in VARIANTS:
            self.image_scores[variant].append(float(scores[variant]))
            resized = _resize_map(maps[variant], self.size)
            self.maps[variant].append(resized.astype(np.float16))

    def _variant_metrics(
        self, variant: str, indices: np.ndarray, with_pro: bool
    ) -> Dict[str, object]:
        labels = np.asarray(self.labels, dtype=np.int64)[indices]
        scores = np.asarray(self.image_scores[variant], dtype=np.float64)[indices]
        image = image_metrics(labels, scores)
        predictions = [self.maps[variant][index].astype(np.float32) for index in indices]
        masks = [self.masks[index] for index in indices]
        pixel = pixel_metrics(
            predictions,
            masks,
            max_points=int(self.config.evaluation.pixel_max_points),
        )
        result: Dict[str, object] = {"image": image, "pixel": pixel}
        if with_pro and self.config.evaluation.enable_pro:
            result["pro"] = compute_pro(
                predictions,
                masks,
                fpr_limit=float(self.config.evaluation.pro_fpr_limit),
                thresholds=int(self.config.evaluation.pro_thresholds),
            )
        return result

    def compute(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "n_samples": len(self.labels),
            "groups": {},
            "variants": list(VARIANTS),
            "pixel_eval_size": self.size,
        }
        for group in ("all", "logical", "structural"):
            indices = _group_indices(self.groups, group)
            if indices.size == 0:
                continue
            entry: Dict[str, object] = {"n_samples": int(indices.size)}
            for variant in VARIANTS:
                entry[variant] = self._variant_metrics(
                    variant, indices, with_pro=True
                )
            result["groups"][group] = entry
        return result

    def dump_predictions(self, path: str | Path) -> None:
        """Persist raw scores and maps for offline re-evaluation."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "labels": np.asarray(self.labels, dtype=np.int64),
            "groups": np.asarray(self.groups, dtype=str),
            "masks": np.stack(self.masks).astype(np.uint8)
            if self.masks
            else np.zeros((0,)),
        }
        for variant in VARIANTS:
            payload[f"{variant}_scores"] = np.asarray(
                self.image_scores[variant], dtype=np.float32
            )
            payload[f"{variant}_maps"] = (
                np.stack(self.maps[variant]).astype(np.float16)
                if self.maps[variant]
                else np.zeros((0,))
            )
        save_array_bundle(path, payload)
