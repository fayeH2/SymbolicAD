"""Weakly supervised neural predicates learned from grounded interventions."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

from .grounded_counterfactuals import GroundedCounterfactual
from .transistor_logic import SpatialTransistorGrounder


PREDICATE_GROUPS: Dict[str, str] = {
    "detach_relation": "detach_lead",
    "cut_continuity": "cut_lead",
    "bend_geometry": "bend_lead",
    "extra_part_count": "extra_lead",
}


def transistor_predicate_features(
    image: np.ndarray | str,
    grounder: SpatialTransistorGrounder,
) -> tuple[np.ndarray, Dict[str, Any], Dict[str, np.ndarray]]:
    """Return compact topology features, diagnostics and spatial localizers."""
    facts, fact_maps, diagnostics, prepared = grounder.extract_spatial(image)
    _, _, _, attachments, template, pose, _ = grounder._scene_with_maps(prepared)
    small = cv2.resize(template.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA)
    horizontal = cv2.resize(template.mean(axis=0)[None], (32, 1), interpolation=cv2.INTER_AREA)[0]
    vertical = cv2.resize(template.mean(axis=1)[:, None], (1, 32), interpolation=cv2.INTER_AREA)[:, 0]
    gx = cv2.Sobel(template.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(template.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.resize(np.hypot(gx, gy), (8, 8), interpolation=cv2.INTER_AREA)
    attachment = np.zeros(12, dtype=np.float32)
    for index, row in enumerate(attachments[:4]):
        attachment[3 * index] = float(row["x"])
        attachment[3 * index + 1] = float(row["width"])
        attachment[3 * index + 2] = 1.0
    vector = np.concatenate([
        small.reshape(-1), horizontal, vertical, gradient.reshape(-1),
        np.asarray([len(attachments) / 4.0], dtype=np.float32),
        attachment, np.asarray(pose, dtype=np.float32).reshape(-1),
    ]).astype(np.float64)
    localizers = {
        "detach_relation": np.asarray(fact_maps["lead_count_error"], dtype=np.float32),
        "extra_part_count": np.asarray(fact_maps["lead_count_error"], dtype=np.float32),
        "cut_continuity": np.asarray(fact_maps["lead_geometry_error"], dtype=np.float32),
        "bend_geometry": np.asarray(fact_maps["lead_geometry_error"], dtype=np.float32),
    }
    return vector, {"facts": facts, "diagnostics": diagnostics}, localizers


@dataclass
class PredicateHead:
    name: str
    center: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    normal_upper: float
    prototypes: np.ndarray

    def _representation(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None]
        distances = np.mean(
            (matrix[:, None, :] - self.prototypes[None, :, :]) ** 2, axis=2
        )
        nearest = self.prototypes[np.argmin(distances, axis=1)]
        return np.abs(matrix - nearest)

    def raw_probability(self, features: np.ndarray) -> np.ndarray:
        matrix = self._representation(features)
        normalized = (matrix - self.center) / self.scale
        logits = normalized @ self.coefficient + self.intercept
        logits = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def violation(self, features: np.ndarray) -> np.ndarray:
        probability = self.raw_probability(features)
        return np.clip(
            (probability - self.normal_upper) / max(1.0 - self.normal_upper, 1e-8),
            0.0,
            1.0,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "coefficient": self.coefficient.tolist(),
            "intercept": self.intercept,
            "normal_upper": self.normal_upper,
            "prototypes": self.prototypes.tolist(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "PredicateHead":
        return cls(
            name=str(state["name"]),
            center=np.asarray(state["center"], dtype=np.float64),
            scale=np.asarray(state["scale"], dtype=np.float64),
            coefficient=np.asarray(state["coefficient"], dtype=np.float64),
            intercept=float(state["intercept"]),
            normal_upper=float(state["normal_upper"]),
            prototypes=np.asarray(state["prototypes"], dtype=np.float64),
        )


def _fit_head(name: str, X: np.ndarray, records: Sequence[GroundedCounterfactual]) -> PredicateHead:
    target_group = PREDICATE_GROUPS[name]
    selected = np.asarray([
        record.label == 0 or record.kind == target_group for record in records
    ])
    y = np.asarray([int(record.kind == target_group) for record in records], dtype=np.uint8)[selected]
    raw_matrix = np.asarray(X, dtype=np.float64)
    prototypes = raw_matrix[np.asarray([
        record.kind == "normal_identity" for record in records
    ])]
    if not len(prototypes):
        raise ValueError("predicate training requires normal identity prototypes")
    probe = raw_matrix[selected]
    distances = np.mean(
        (probe[:, None, :] - prototypes[None, :, :]) ** 2, axis=2
    )
    matrix = np.abs(probe - prototypes[np.argmin(distances, axis=1)])
    center = np.mean(matrix, axis=0)
    scale = np.maximum(np.std(matrix, axis=0), 1e-5)
    normalized = (matrix - center) / scale
    model = LogisticRegression(
        C=0.05, class_weight="balanced", max_iter=2000,
        solver="liblinear", random_state=668,
    ).fit(normalized, y)
    raw = 1.0 / (1.0 + np.exp(-np.clip(
        normalized @ model.coef_[0] + float(model.intercept_[0]), -40.0, 40.0
    )))
    normal_upper = float(np.quantile(raw[y == 0], 0.99))
    return PredicateHead(
        name=name, center=center, scale=scale,
        coefficient=model.coef_[0].astype(np.float64),
        intercept=float(model.intercept_[0]), normal_upper=normal_upper,
        prototypes=prototypes,
    )


def _source_fold(source: str, folds: int) -> int:
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % folds


def crossfit_predicate_heads(
    X: np.ndarray,
    records: Sequence[GroundedCounterfactual],
    *,
    folds: int = 5,
) -> tuple[np.ndarray, Dict[str, PredicateHead], Dict[str, Any]]:
    """Create source-disjoint OOF truths and final deployable heads."""
    names = list(PREDICATE_GROUPS)
    source_folds = np.asarray([_source_fold(record.source, folds) for record in records])
    oof_raw = np.zeros((len(records), len(names)), dtype=np.float64)
    fold_audit = []
    for fold in range(folds):
        train_indices = np.flatnonzero(source_folds != fold)
        valid_indices = np.flatnonzero(source_folds == fold)
        if not len(train_indices) or not len(valid_indices):
            continue
        train_records = [records[index] for index in train_indices]
        for column, name in enumerate(names):
            head = _fit_head(name, X[train_indices], train_records)
            oof_raw[valid_indices, column] = head.raw_probability(X[valid_indices])
        fold_audit.append({
            "fold": fold,
            "train_source_count": len(set(record.source for record in train_records)),
            "valid_source_count": len(set(records[index].source for index in valid_indices)),
            "train_record_count": len(train_indices),
            "valid_record_count": len(valid_indices),
        })
    normal_mask = np.asarray([record.label == 0 for record in records])
    oof_violations = np.zeros_like(oof_raw)
    conformal_upper = {}
    for column, name in enumerate(names):
        upper = float(np.max(oof_raw[normal_mask, column]))
        conformal_upper[name] = upper
        oof_violations[:, column] = np.clip(
            (oof_raw[:, column] - upper) / max(1.0 - upper, 1e-8), 0.0, 1.0
        )
    heads = {name: _fit_head(name, X, records) for name in names}
    for name, head in heads.items():
        # Cross-conformal calibration estimates unseen-normal behaviour instead
        # of using optimistically low in-sample probabilities.
        head.normal_upper = max(head.normal_upper, conformal_upper[name])
    return 1.0 - oof_violations, heads, {
        "folds": folds,
        "source_disjoint": True,
        "folds_audit": fold_audit,
        "feature_dimension": int(X.shape[1]),
        "predicate_groups": dict(PREDICATE_GROUPS),
        "training_loss": "balanced logistic loss with L2 regularization",
        "normal_calibration": (
            "maximum source-disjoint out-of-fold raw probability on normal "
            "interventions (zero-training-error conformal boundary)"
        ),
        "cross_conformal_normal_upper": conformal_upper,
    }


def extract_feature_matrix(
    records: Sequence[GroundedCounterfactual],
    grounder: SpatialTransistorGrounder,
) -> tuple[np.ndarray, list[Dict[str, Any]]]:
    rows, audit = [], []
    for record in records:
        features, diagnostics, _ = transistor_predicate_features(record.image, grounder)
        rows.append(features)
        audit.append({"source": record.source, "kind": record.kind, **diagnostics})
    return np.asarray(rows, dtype=np.float64), audit


__all__ = [
    "PREDICATE_GROUPS", "PredicateHead", "crossfit_predicate_heads",
    "extract_feature_matrix", "transistor_predicate_features",
]
