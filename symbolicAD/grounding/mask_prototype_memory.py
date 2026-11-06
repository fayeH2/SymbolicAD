"""Training-free FSAD memory for category-agnostic segmentation masks.

The memory deliberately does not depend on text labels.  Each normal SAM mask
is stored as a visual embedding plus explicit attributes.  Query masks are
scored by nearest-neighbour distance, while thresholds are estimated only from
leave-one-reference-image-out distances.  Optional component groups are an
offline alignment result; their semantic names are metadata, not match keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from symbolicAD.array_store import load_array_bundle, save_array_bundle


MASK_MEMORY_VERSION = "v1_sam_mask_loo_quantile"
ATTRIBUTE_NAMES = (
    "log_area",
    "log_aspect",
    "circularity",
    "centroid_x",
    "centroid_y",
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a matrix, got shape={values.shape}")
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    """Empirical higher quantile, matching ``quantile_q(d_ref)``."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        raise ValueError("cannot calibrate a quantile from zero values")
    return float(np.quantile(values, float(quantile), method="higher"))


def attributes_from_geometry(item: Mapping[str, object]) -> np.ndarray:
    mean_rgb = np.asarray(item.get("mean_rgb", [0.0, 0.0, 0.0]), dtype=np.float32)
    std_rgb = np.asarray(item.get("std_rgb", [0.0, 0.0, 0.0]), dtype=np.float32)
    centroid = np.asarray(item.get("centroid_xy", [0.0, 0.0]), dtype=np.float32)
    return np.asarray(
        [
            np.log(max(float(item.get("area_ratio", 0.0)), 1e-8)),
            np.log(max(float(item.get("aspect_ratio", 1.0)), 1e-8)),
            float(item.get("circularity", 0.0)),
            float(centroid[0]),
            float(centroid[1]),
            *(mean_rgb / 255.0).tolist(),
            *(std_rgb / 255.0).tolist(),
        ],
        dtype=np.float32,
    )


@dataclass
class MaskPrototypeSample:
    image_id: str
    mask_id: int
    visual: np.ndarray
    attributes: np.ndarray
    context: str = "default"
    group_id: str = ""
    semantic_label: str = ""


@dataclass
class ThresholdCalibration:
    quantile: float
    threshold: float
    sample_count: int
    reference_image_count: int
    distances: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    confidence: str = "low"

    def to_dict(self, include_distances: bool = True) -> Dict[str, object]:
        result = {
            "quantile": float(self.quantile),
            "threshold": float(self.threshold),
            "sample_count": int(self.sample_count),
            "reference_image_count": int(self.reference_image_count),
            "confidence": self.confidence,
        }
        if include_distances:
            result["distances"] = self.distances.astype(float).tolist()
        return result


@dataclass
class MaskScore:
    mask_id: int
    distance: float
    threshold: float
    normalized_violation: float
    violated: bool
    nearest_image_id: str
    nearest_mask_id: int
    nearest_group_id: str
    nearest_semantic_label: str
    context_requested: str
    context_used: str
    visual_distance: float
    attribute_distance: float

    def to_dict(self) -> Dict[str, object]:
        return {
            key: (float(value) if isinstance(value, (np.floating, float)) else value)
            for key, value in vars(self).items()
        }


@dataclass
class MaskPrototypeMemory:
    visual: np.ndarray
    attributes: np.ndarray
    attribute_center: np.ndarray
    attribute_scale: np.ndarray
    image_ids: List[str]
    mask_ids: np.ndarray
    contexts: List[str]
    group_ids: List[str]
    semantic_labels: List[str]
    calibration: ThresholdCalibration
    context_calibrations: Dict[str, ThresholdCalibration] = field(default_factory=dict)
    group_calibrations: Dict[str, ThresholdCalibration] = field(default_factory=dict)
    visual_weight: float = 0.70
    attribute_weight: float = 0.30
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.visual = _unit_rows(self.visual)
        self.attributes = np.asarray(self.attributes, dtype=np.float32)
        self.attribute_center = np.asarray(self.attribute_center, dtype=np.float32)
        self.attribute_scale = np.asarray(self.attribute_scale, dtype=np.float32)
        self.mask_ids = np.asarray(self.mask_ids, dtype=np.int64)
        count = self.visual.shape[0]
        if self.attributes.shape != (count, len(ATTRIBUTE_NAMES)):
            raise ValueError(
                f"attribute shape {self.attributes.shape} is incompatible with {count} samples"
            )
        for values in (self.image_ids, self.contexts, self.group_ids, self.semantic_labels):
            if len(values) != count:
                raise ValueError("prototype metadata length mismatch")

    @property
    def standardized_attributes(self) -> np.ndarray:
        return (self.attributes - self.attribute_center) / self.attribute_scale

    def _distances(self, visual: np.ndarray, attributes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query_visual = np.asarray(visual, dtype=np.float32).reshape(1, -1)
        query_visual = _unit_rows(query_visual)[0]
        query_attributes = np.asarray(attributes, dtype=np.float32)
        if query_attributes.shape != (len(ATTRIBUTE_NAMES),):
            raise ValueError(f"unexpected query attribute shape {query_attributes.shape}")
        visual_distance = np.clip(1.0 - self.visual @ query_visual, 0.0, 2.0)
        query_standard = (query_attributes - self.attribute_center) / self.attribute_scale
        delta = self.standardized_attributes - query_standard[None, :]
        # RMS keeps the attribute term comparable when attributes are extended.
        attribute_distance = np.sqrt(np.mean(np.square(delta), axis=1))
        attribute_distance = np.minimum(attribute_distance, 4.0) / 4.0
        combined = self.visual_weight * visual_distance + self.attribute_weight * attribute_distance
        return combined, visual_distance, attribute_distance

    def threshold_for(self, context: str) -> tuple[ThresholdCalibration, str]:
        if context in self.context_calibrations:
            return self.context_calibrations[context], context
        return self.calibration, "global"

    def score_appearance(
        self,
        visual: np.ndarray,
        mask_id: int,
    ) -> MaskScore:
        """Match one query mask to the visually most similar normal prototype.

        This is the appearance-anomaly path.  It deliberately traverses every
        normal prototype and chooses the maximum CLIP cosine similarity.  No
        attribute scale, local prototype radius, semantic group or relation is
        allowed to change which normal prototype is the nearest appearance
        match.
        """
        query = _unit_rows(
            np.asarray(visual, dtype=np.float32).reshape(1, -1)
        )[0]
        similarities = self.visual @ query
        nearest = int(np.argmax(similarities))
        visual_distance = float(np.clip(1.0 - similarities[nearest], 0.0, 2.0))
        threshold = max(
            float(
                self.metadata.get(
                    "appearance_visual_distance_threshold",
                    self.calibration.threshold,
                )
            ),
            1e-8,
        )
        # The appearance heat-map is the direct difference from the most
        # similar normal prototype.  Do not erase sub-threshold differences:
        # high similarity is naturally a low score, while low best similarity
        # is naturally high.  The support-only threshold is retained solely
        # for the explicit confirmed-anomaly flag.
        violation = float(np.clip(visual_distance, 0.0, 1.0))
        return MaskScore(
            mask_id=int(mask_id),
            distance=visual_distance,
            threshold=threshold,
            normalized_violation=violation,
            violated=bool(visual_distance > threshold),
            nearest_image_id=self.image_ids[nearest],
            nearest_mask_id=int(self.mask_ids[nearest]),
            nearest_group_id=self.group_ids[nearest],
            nearest_semantic_label=self.semantic_labels[nearest],
            context_requested="appearance",
            context_used="global_clip_cosine_nearest_normal",
            visual_distance=visual_distance,
            attribute_distance=0.0,
        )

    def score(
        self,
        visual: np.ndarray,
        attributes: np.ndarray,
        mask_id: int,
        context: str = "default",
        candidate_group: str = "",
    ) -> MaskScore:
        combined, visual_distance, attribute_distance = self._distances(visual, attributes)
        candidates = np.arange(len(combined))
        if candidate_group:
            restricted = np.asarray(
                [index for index, value in enumerate(self.group_ids) if value == candidate_group],
                dtype=np.int64,
            )
            if restricted.size:
                candidates = restricted
        calibration, context_used = self.threshold_for(context)
        prototype_thresholds = np.asarray(
            self.metadata.get("prototype_acceptance_thresholds", []),
            dtype=np.float32,
        )
        if prototype_thresholds.shape == combined.shape:
            # A match means entering one concrete normal prototype's local
            # acceptance ball, not merely being below a library-wide radius.
            # Select by normalized distance so a broad prototype cannot hide
            # the absence of a genuinely corresponding normal component.
            ratios = combined[candidates] / np.maximum(
                prototype_thresholds[candidates], 1e-8
            )
            nearest = int(candidates[np.argmin(ratios)])
            threshold = max(float(prototype_thresholds[nearest]), 1e-8)
            context_used = "prototype_local_loo"
        else:
            nearest = int(candidates[np.argmin(combined[candidates])])
            threshold = max(float(calibration.threshold), 1e-8)
        distance = float(combined[nearest])
        # Query-to-normal-library rejection semantics: a query mask that finds
        # a normal prototype within the support-only threshold is normal and
        # must contribute exactly zero anomaly evidence.  Only the excess
        # beyond the acceptance radius is anomalous.  Keep a short continuous
        # ramp outside the radius so rejected masks can still be ranked.
        rejection_margin = max(0.25 * threshold, 0.02)
        violation = float(
            np.clip((distance - threshold) / rejection_margin, 0.0, 1.0)
        )
        return MaskScore(
            mask_id=int(mask_id),
            distance=distance,
            threshold=threshold,
            normalized_violation=violation,
            violated=bool(distance > threshold),
            nearest_image_id=self.image_ids[nearest],
            nearest_mask_id=int(self.mask_ids[nearest]),
            nearest_group_id=self.group_ids[nearest],
            nearest_semantic_label=self.semantic_labels[nearest],
            context_requested=context,
            context_used=context_used,
            visual_distance=float(visual_distance[nearest]),
            attribute_distance=float(attribute_distance[nearest]),
        )

    def score_group(
        self,
        visual: np.ndarray,
        attributes: np.ndarray,
        mask_id: int,
        group_id: str,
        context: str = "default",
    ) -> MaskScore:
        """Score a query against one validated component group.

        Group thresholds are calibrated from cross-reference matches of that
        group.  They are never inferred from its optional semantic label.
        """
        if group_id not in self.group_calibrations:
            raise KeyError(f"group {group_id!r} has no cross-reference calibration")
        result = self.score(
            visual,
            attributes,
            mask_id=mask_id,
            context=context,
            candidate_group=group_id,
        )
        calibration = self.group_calibrations[group_id]
        # Two-shot group estimates can be unrealistically tight.  Apply an
        # explicit safety margin to the group's own normal distance instead of
        # borrowing the global fragment threshold, which may be much too loose.
        threshold = max(
            1.5 * float(calibration.threshold),
            float(calibration.threshold) + 0.02,
        )
        result.threshold = threshold
        rejection_margin = max(0.25 * threshold, 0.02)
        result.normalized_violation = float(
            np.clip((result.distance - threshold) / rejection_margin, 0.0, 1.0)
        )
        result.violated = bool(result.distance > threshold)
        result.context_used = f"group:{group_id}"
        return result

    def audit_reference_images(self) -> Dict[str, object]:
        distances = np.asarray(self.calibration.distances, dtype=np.float64)
        per_image: Dict[str, List[float]] = {}
        loo_image_ids = list(self.metadata.get("loo_image_ids", []))
        for image_id, value in zip(loo_image_ids, distances):
            per_image.setdefault(str(image_id), []).append(float(value))
        medians = {key: float(np.median(value)) for key, value in per_image.items()}
        values = np.asarray(list(medians.values()), dtype=np.float64)
        if values.size < 3:
            return {
                "status": "insufficient_reference_images",
                "per_image_median_loo_distance": medians,
                "flagged_images": [],
            }
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        limit = center + 3.0 * max(1.4826 * mad, 1e-6)
        return {
            "status": "audited",
            "per_image_median_loo_distance": medians,
            "robust_limit": limit,
            "flagged_images": [key for key, value in medians.items() if value > limit],
        }


def _confidence(reference_images: int, sample_count: int) -> str:
    if reference_images >= 8 and sample_count >= 50:
        return "high"
    if reference_images >= 4 and sample_count >= 20:
        return "medium"
    return "low"


def _calibration(
    distances: Sequence[float], quantile: float, reference_images: int
) -> ThresholdCalibration:
    values = np.asarray(distances, dtype=np.float32)
    return ThresholdCalibration(
        quantile=float(quantile),
        threshold=_higher_quantile(values, quantile),
        sample_count=int(values.size),
        reference_image_count=int(reference_images),
        distances=values,
        confidence=_confidence(reference_images, int(values.size)),
    )


def _unique_samples(samples: Sequence[MaskPrototypeSample]) -> List[MaskPrototypeSample]:
    """Remove repeated references to the exact same support mask."""
    unique: List[MaskPrototypeSample] = []
    seen = set()
    for sample in samples:
        key = (str(sample.image_id), int(sample.mask_id))
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def _merge_samples_as_union(
    samples: Sequence[MaskPrototypeSample],
    attribute_center: np.ndarray,
    attribute_scale: np.ndarray,
    visual_similarity: float,
    attribute_distance: float,
) -> tuple[List[List[int]], List[Dict[str, object]]]:
    """Merge shots as a set union with one-to-one mask correspondence.

    Distinct masks within one image always remain distinct prototypes.  For
    each later shot, Hungarian matching lets each mask update at most one
    existing prototype.  This prevents one repeated-looking mask from being
    counted several times while still preserving multiple physical instances.
    """
    image_order = list(dict.fromkeys(str(sample.image_id) for sample in samples))
    by_image = {
        image_id: [index for index, item in enumerate(samples) if item.image_id == image_id]
        for image_id in image_order
    }
    visual = _unit_rows(np.stack([sample.visual for sample in samples], axis=0))
    attributes = np.stack([sample.attributes for sample in samples], axis=0).astype(np.float32)
    standardized = (attributes - attribute_center[None, :]) / attribute_scale[None, :]
    clusters: List[List[int]] = [[index] for index in by_image[image_order[0]]]
    audit: List[Dict[str, object]] = []

    for image_id in image_order[1:]:
        incoming = by_image[image_id]
        if not incoming:
            continue
        cost = np.full((len(clusters), len(incoming)), 1e6, dtype=np.float64)
        for cluster_index, members in enumerate(clusters):
            centroid_visual = _unit_rows(visual[members].mean(axis=0, keepdims=True))[0]
            centroid_attribute = standardized[members].mean(axis=0)
            groups = {samples[index].group_id for index in members if samples[index].group_id}
            for incoming_index, sample_index in enumerate(incoming):
                sample = samples[sample_index]
                if sample.group_id and groups and sample.group_id not in groups:
                    continue
                cosine = float(np.dot(centroid_visual, visual[sample_index]))
                attr_distance = float(
                    np.sqrt(np.mean(np.square(centroid_attribute - standardized[sample_index])))
                )
                same_group = bool(sample.group_id and sample.group_id in groups)
                if not same_group and (
                    cosine < visual_similarity or attr_distance > attribute_distance
                ):
                    continue
                cost[cluster_index, incoming_index] = (1.0 - cosine) + 0.05 * attr_distance
        rows, cols = linear_sum_assignment(cost)
        matched = set()
        for row, col in zip(rows, cols):
            if cost[row, col] >= 1e5:
                continue
            sample_index = incoming[col]
            clusters[row].append(sample_index)
            matched.add(sample_index)
            audit.append(
                {
                    "image_id": image_id,
                    "mask_id": int(samples[sample_index].mask_id),
                    "prototype_index": int(row),
                    "cost": float(cost[row, col]),
                }
            )
        for sample_index in incoming:
            if sample_index not in matched:
                clusters.append([sample_index])
    return clusters, audit


def build_mask_prototype_memory(
    samples: Sequence[MaskPrototypeSample],
    quantile: float = 0.95,
    visual_weight: float = 0.70,
    attribute_weight: float = 0.30,
    dedup_visual_similarity: float = 0.97,
    dedup_attribute_distance: float = 0.75,
) -> MaskPrototypeMemory:
    """Build a deduplicated union of normal support-mask prototypes.

    Calibration always uses the original unique observations.  The runtime
    library uses one centroid per cross-shot correspondence, so adding a shot
    with the same normal components does not duplicate the prototype bank.
    """
    if not samples:
        raise ValueError("normal mask prototype set is empty")
    original_count = len(samples)
    samples = _unique_samples(samples)
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")
    if visual_weight < 0 or attribute_weight < 0 or visual_weight + attribute_weight <= 0:
        raise ValueError("distance weights must be non-negative and not both zero")
    total = visual_weight + attribute_weight
    visual_weight, attribute_weight = visual_weight / total, attribute_weight / total

    raw_visual = _unit_rows(np.stack([sample.visual for sample in samples], axis=0))
    raw_attributes = np.stack([sample.attributes for sample in samples], axis=0).astype(np.float32)
    center = np.median(raw_attributes, axis=0).astype(np.float32)
    mad = np.median(np.abs(raw_attributes - center[None, :]), axis=0).astype(np.float32)
    scale = np.maximum(1.4826 * mad, np.asarray(
        [0.25, 0.25, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.03, 0.03],
        dtype=np.float32,
    ))
    clusters, merge_audit = _merge_samples_as_union(
        samples,
        center,
        scale,
        visual_similarity=float(dedup_visual_similarity),
        attribute_distance=float(dedup_attribute_distance),
    )
    visual = np.stack(
        [_unit_rows(raw_visual[members].mean(axis=0, keepdims=True))[0] for members in clusters],
        axis=0,
    )
    attributes = np.stack(
        [np.median(raw_attributes[members], axis=0) for members in clusters], axis=0
    ).astype(np.float32)

    def first_text(members: Sequence[int], field: str, default: str = "") -> str:
        for index in members:
            value = str(getattr(samples[index], field))
            if value:
                return value
        return default

    memory = MaskPrototypeMemory(
        visual=visual,
        attributes=attributes,
        attribute_center=center,
        attribute_scale=scale,
        image_ids=[str(samples[members[0]].image_id) for members in clusters],
        mask_ids=np.asarray([samples[members[0]].mask_id for members in clusters], dtype=np.int64),
        contexts=[first_text(members, "context", "default") for members in clusters],
        group_ids=[first_text(members, "group_id") for members in clusters],
        semantic_labels=[first_text(members, "semantic_label", "unlabeled_mask") for members in clusters],
        calibration=ThresholdCalibration(quantile, 0.0, 0, 0),
        visual_weight=float(visual_weight),
        attribute_weight=float(attribute_weight),
        metadata={"version": MASK_MEMORY_VERSION, "attribute_names": list(ATTRIBUTE_NAMES)},
    )

    raw_standardized = (raw_attributes - center[None, :]) / scale[None, :]
    raw_visual_distance = np.clip(1.0 - raw_visual @ raw_visual.T, 0.0, 2.0)
    raw_attribute_distance = np.sqrt(
        np.mean(
            np.square(raw_standardized[:, None, :] - raw_standardized[None, :, :]),
            axis=2,
        )
    )
    raw_attribute_distance = np.minimum(raw_attribute_distance, 4.0) / 4.0
    raw_combined = visual_weight * raw_visual_distance + attribute_weight * raw_attribute_distance
    loo_distances: List[float] = []
    appearance_loo_distances: List[float] = []
    loo_image_ids: List[str] = []
    per_context: Dict[str, List[float]] = {}
    per_group: Dict[str, List[float]] = {}
    image_count = len({sample.image_id for sample in samples})
    for index, sample in enumerate(samples):
        candidates = [
            other for other, value in enumerate(samples)
            if other != index and (image_count == 1 or value.image_id != sample.image_id)
        ]
        # A one-mask 1-shot support has no empirical neighbour.  Keep a small,
        # explicit low-confidence radius instead of rejecting the valid setup.
        distance = float(np.min(raw_combined[index, candidates])) if candidates else 0.10
        appearance_distance = (
            float(np.min(raw_visual_distance[index, candidates])) if candidates else 0.10
        )
        loo_distances.append(distance)
        appearance_loo_distances.append(appearance_distance)
        loo_image_ids.append(sample.image_id)
        per_context.setdefault(sample.context, []).append(distance)

        if sample.group_id:
            group_candidates = np.asarray(
                [
                    other
                    for other, value in enumerate(samples)
                    if other != index
                    and value.group_id == sample.group_id
                    and (image_count == 1 or value.image_id != sample.image_id)
                ],
                dtype=np.int64,
            )
            if group_candidates.size:
                per_group.setdefault(sample.group_id, []).append(
                    float(np.min(raw_combined[index, group_candidates]))
                )

    for index, sample in enumerate(samples):
        if sample.group_id and sample.group_id not in per_group:
            per_group[sample.group_id] = [loo_distances[index]]
    memory.calibration = _calibration(loo_distances, quantile, image_count)
    memory.context_calibrations = {
        context: _calibration(values, quantile, len({
            sample.image_id for sample in samples if sample.context == context
        }))
        for context, values in per_context.items()
        if values
    }
    memory.group_calibrations = {
        group: _calibration(values, quantile, len({
            sample.image_id for sample in samples if sample.group_id == group
        }))
        for group, values in per_group.items()
        if values
    }
    radius_cap = _higher_quantile(loo_distances, 0.75)
    prototype_distances = [
        max(loo_distances[index] for index in members) for members in clusters
    ]
    memory.metadata.update(
        {
            "support_image_count": image_count,
            "raw_sample_count": original_count,
            "unique_mask_count": len(samples),
            "prototype_count": len(clusters),
            "deduplicated_mask_references": original_count - len(samples),
            "merged_cross_shot_observations": len(samples) - len(clusters),
            "prototype_union_semantics": "one_to_one_cross_shot_set_union",
            "dedup_visual_similarity": float(dedup_visual_similarity),
            "dedup_attribute_distance": float(dedup_attribute_distance),
            "prototype_members": [
                [
                    {"image_id": str(samples[index].image_id), "mask_id": int(samples[index].mask_id)}
                    for index in members
                ]
                for members in clusters
            ],
            "prototype_merge_audit": merge_audit,
            "quantile_method": "empirical_higher",
            "calibration": (
                "leave_one_reference_image_out" if image_count > 1
                else "single_reference_leave_one_mask_out"
            ),
            "loo_image_ids": loo_image_ids,
            "threshold_confidence": memory.calibration.confidence,
            "appearance_visual_distance_threshold": _higher_quantile(
                appearance_loo_distances, quantile
            ),
            "appearance_min_cosine_similarity": 1.0
            - _higher_quantile(appearance_loo_distances, quantile),
            "appearance_loo_visual_distances": appearance_loo_distances,
            "appearance_calibration": (
                "leave_one_normal_reference_image_out_nearest_raw_clip_"
                "cosine_q95"
            ),
            "appearance_representation": "unit(raw_masked_clip_vector)",
            # One acceptance radius per normal prototype, estimated only from
            # its nearest mask in another normal reference image.  The margin
            # accounts for the very small 2-shot calibration set.
            "prototype_acceptance_thresholds": [
                min(
                    max(1.5 * float(value), float(value) + 0.02),
                    radius_cap,
                )
                for value in prototype_distances
            ],
            "prototype_acceptance_radius_cap": radius_cap,
            "prototype_acceptance_calibration": (
                "nearest_other_normal_image_distance_with_1.5x_or_plus_0.02_margin_"
                "capped_at_support_loo_q75"
            ),
        }
    )
    return memory


def save_mask_prototype_memory(memory: MaskPrototypeMemory, path: str | Path) -> Path:
    manifest = {
        "version": MASK_MEMORY_VERSION,
        "image_ids": memory.image_ids,
        "contexts": memory.contexts,
        "group_ids": memory.group_ids,
        "semantic_labels": memory.semantic_labels,
        "visual_weight": memory.visual_weight,
        "attribute_weight": memory.attribute_weight,
        "calibration": memory.calibration.to_dict(),
        "context_calibrations": {
            key: value.to_dict() for key, value in memory.context_calibrations.items()
        },
        "group_calibrations": {
            key: value.to_dict() for key, value in memory.group_calibrations.items()
        },
        "metadata": memory.metadata,
    }
    return save_array_bundle(
        path,
        {
            "visual": memory.visual.astype(np.float32),
            "attributes": memory.attributes.astype(np.float32),
            "attribute_center": memory.attribute_center.astype(np.float32),
            "attribute_scale": memory.attribute_scale.astype(np.float32),
            "mask_ids": memory.mask_ids.astype(np.int64),
        },
        metadata=manifest,
    )


def _calibration_from_dict(payload: Mapping[str, object]) -> ThresholdCalibration:
    return ThresholdCalibration(
        quantile=float(payload["quantile"]),
        threshold=float(payload["threshold"]),
        sample_count=int(payload["sample_count"]),
        reference_image_count=int(payload["reference_image_count"]),
        distances=np.asarray(payload.get("distances", []), dtype=np.float32),
        confidence=str(payload.get("confidence", "low")),
    )


def load_mask_prototype_memory(path: str | Path) -> MaskPrototypeMemory:
    path = Path(path)
    if path.is_file():  # Read-only compatibility with historical experiment files.
        with np.load(path, allow_pickle=False) as legacy:
            payload = {key: legacy[key] for key in legacy.files if key != "manifest"}
            manifest = json.loads(str(legacy["manifest"].item()))
    else:
        payload, manifest = load_array_bundle(path)
    return MaskPrototypeMemory(
        visual=payload["visual"].astype(np.float32),
        attributes=payload["attributes"].astype(np.float32),
        attribute_center=payload["attribute_center"].astype(np.float32),
        attribute_scale=payload["attribute_scale"].astype(np.float32),
        image_ids=list(manifest["image_ids"]),
        mask_ids=payload["mask_ids"].astype(np.int64),
        contexts=list(manifest["contexts"]),
        group_ids=list(manifest["group_ids"]),
        semantic_labels=list(manifest["semantic_labels"]),
        calibration=_calibration_from_dict(manifest["calibration"]),
        context_calibrations={
            key: _calibration_from_dict(value)
            for key, value in manifest.get("context_calibrations", {}).items()
        },
        group_calibrations={
            key: _calibration_from_dict(value)
            for key, value in manifest.get("group_calibrations", {}).items()
        },
        visual_weight=float(manifest["visual_weight"]),
        attribute_weight=float(manifest["attribute_weight"]),
        metadata=dict(manifest.get("metadata", {})),
    )
