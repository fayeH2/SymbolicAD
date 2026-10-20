"""Executable entity--relation normality rules for the MVTec cable class.

This module is deliberately independent from INP-Former.  Qwen proposes a
symbolic program offline; this deterministic visual grounder measures the
entities and the static engine executes the compiled rules at test time.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score


ROLE_NAMES = ("top", "bottom_left", "bottom_right")
BOOTSTRAP_SLOTS = np.asarray(((0.50, 0.32), (0.34, 0.60), (0.66, 0.60)), dtype=float)


@dataclass
class RelationalCalibrationView:
    image: np.ndarray
    source_index: int
    transform: str
    is_pseudo_anomaly: bool
    family: str


def _as_rgb(image: Union[str, Path, Image.Image, np.ndarray], size: int) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
            array = np.clip(array * scale, 0, 255).astype(np.uint8)
    return cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA)


def _assignment(cost: np.ndarray) -> Tuple[Tuple[int, ...], float, float]:
    """Solve the tiny 3x3 assignment without adding a scipy dependency."""
    best_perm: Tuple[int, ...] = tuple(range(cost.shape[1]))
    best_sum = float("inf")
    best_max = float("inf")
    for permutation in itertools.permutations(range(cost.shape[1]), cost.shape[0]):
        values = np.asarray([cost[row, column] for row, column in enumerate(permutation)])
        total = float(values.mean())
        maximum = float(values.max())
        if (total, maximum) < (best_sum, best_max):
            best_perm, best_sum, best_max = permutation, total, maximum
    return best_perm, best_sum, best_max


def _prototype_to_detected(permutation: Sequence[int]) -> np.ndarray:
    """Invert a detected-row -> prototype-column assignment."""
    return np.argsort(np.asarray(permutation, dtype=int))


class CableSceneGrounder:
    """Ground cable parts using circles and support-derived color identities."""

    def __init__(self, state: Optional[Mapping[str, Any]] = None, image_size: int = 256):
        self.image_size = int(image_size)
        self.slots = BOOTSTRAP_SLOTS.copy()
        self.slot_radii = np.full(3, 0.16, dtype=float)
        self.role_color_prototypes = np.zeros((3, 3), dtype=float)
        self.role_color_exemplars = np.zeros((3, 1, 3), dtype=float)
        self.semantic_roles = {
            "yellow_green": 0,
            "blue": 1,
            "brown": 2,
        }
        self.core_reference = 0.0
        # A hue-aware term is necessary because exposure changes can move a
        # desaturated blue conductor close to brown in raw RGB chromaticity.
        # The term is automatically down-weighted for nearly gray pixels where
        # hue is unstable.
        self.hue_cost_weight = 0.50
        self.anchor_strength_floor = 0.0
        self.fitted = False
        if state is not None:
            self.load_state(state)

    def prepare(self, image) -> np.ndarray:
        return _as_rgb(image, self.image_size)

    def _circle_candidates(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
        n = self.image_size
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=max(24, int(round(0.115 * n))),
            param1=70,
            param2=20,
            minRadius=max(18, int(round(0.098 * n))),
            maxRadius=max(30, int(round(0.192 * n))),
        )
        if circles is None:
            return np.empty((0, 3), dtype=float)
        candidates = np.asarray(circles[0], dtype=float)
        keep = (
            (candidates[:, 0] > 0.17 * n)
            & (candidates[:, 0] < 0.83 * n)
            & (candidates[:, 1] > 0.15 * n)
            & (candidates[:, 1] < 0.82 * n)
        )
        return candidates[keep]

    def _locate(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        candidates = self._circle_candidates(image)
        n = float(self.image_size)
        fallback = np.c_[self.slots * n, self.slot_radii * n]
        if len(candidates) < 3:
            return fallback, np.full(3, 1.0, dtype=float), int(len(candidates))
        center_cost = np.linalg.norm(
            self.slots[:, None, :] * n - candidates[None, :, :2], axis=2
        ) / n
        radius_cost = np.abs(
            self.slot_radii[:, None] - candidates[None, :, 2] / n
        )
        cost = center_cost + 2.0 * radius_cost
        best_columns = None
        best_values = None
        best_key = (float("inf"), float("inf"))
        for columns in itertools.permutations(range(len(candidates)), 3):
            values = np.asarray([cost[row, column] for row, column in enumerate(columns)])
            key = (float(values.mean()), float(values.max()))
            if key < best_key:
                best_key, best_columns, best_values = key, columns, values
        circles = np.asarray([candidates[index] for index in best_columns], dtype=float)
        matched = int(np.sum(best_values < 0.10))
        return circles, np.asarray(best_values), matched

    @staticmethod
    def _chromaticity(image: np.ndarray, circle: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        yy, xx = np.mgrid[:height, :width]
        cx, cy, radius = (float(value) for value in circle)
        distance = np.hypot(xx - cx, yy - cy)
        mask = (distance >= 0.52 * radius) & (distance <= 0.88 * radius)
        pixels = image[mask].astype(np.float64)
        valid = (pixels.min(axis=1) > 12) & (pixels.max(axis=1) < 245)
        if int(valid.sum()) >= 20:
            pixels = pixels[valid]
        if not len(pixels):
            return np.full(3, 1.0 / 3.0)
        chromaticity = pixels / np.maximum(pixels.sum(axis=1, keepdims=True), 1.0)
        return np.median(chromaticity, axis=0)

    @staticmethod
    def _chromaticity_hs(chromaticity: np.ndarray) -> Tuple[float, float]:
        rgb = np.clip(
            chromaticity / max(float(np.max(chromaticity)), 1e-8) * 255.0,
            0,
            255,
        )
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0]
        return float(hsv[0]), float(hsv[1]) / 255.0

    def _identity_distance(self, colors: np.ndarray) -> np.ndarray:
        """Distance from detected colors to every support-derived identity.

        RGB chromaticity remains the primary cue.  Circular hue distance is a
        small auxiliary cue and is weighted by the geometric mean saturation,
        so it helps a dim-but-still-blue conductor without treating the hue of
        a gray/brown conductor as reliable.
        """
        base = np.linalg.norm(
            colors[:, None, None, :]
            - self.role_color_exemplars[None, :, :, :],
            axis=3,
        )
        color_hs = np.asarray([self._chromaticity_hs(item) for item in colors])
        exemplar_hs = np.asarray([
            [self._chromaticity_hs(item) for item in role]
            for role in self.role_color_exemplars
        ])
        hue_delta = np.abs(
            color_hs[:, None, None, 0] - exemplar_hs[None, :, :, 0]
        )
        hue_delta = np.minimum(hue_delta, 180.0 - hue_delta) / 90.0
        saturation_weight = np.sqrt(
            color_hs[:, None, None, 1] * exemplar_hs[None, :, :, 1]
        )
        hybrid = base + self.hue_cost_weight * saturation_weight * hue_delta
        return np.min(hybrid, axis=2)

    @staticmethod
    def _outer_anchor_strength(image: np.ndarray) -> float:
        """Measure the stable outer-sheath annulus, independently of conductors.

        Missing conductors are legitimate anomalies and therefore must not make
        the relation branch abstain.  This anchor only checks that the expected
        large circular cable cross-section is present and reasonably grounded.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.hypot(gx, gy)
        height, width = gray.shape
        yy, xx = np.mgrid[:height, :width]
        radius = np.hypot(
            (xx - (width - 1) / 2.0) / max(width, 1),
            (yy - (height - 1) / 2.0) / max(height, 1),
        )
        annulus = (radius >= 0.36) & (radius <= 0.49)
        if not annulus.any():
            return 0.0
        # A quantile is robust to local scratches while still requiring edge
        # coverage around the large sheath boundary.
        return float(np.quantile(gradient[annulus], 0.70))

    @staticmethod
    def _core_texture(image: np.ndarray, circle: np.ndarray) -> float:
        height, width = image.shape[:2]
        yy, xx = np.mgrid[:height, :width]
        cx, cy, radius = (float(value) for value in circle)
        mask = np.hypot(xx - cx, yy - cy) <= 0.48 * radius
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.hypot(gx, gy)
        return float(np.mean(gradient[mask]) + np.std(gray[mask])) if mask.any() else 0.0

    @staticmethod
    def _identity_name(chromaticity: np.ndarray) -> str:
        rgb = np.clip(chromaticity / max(float(chromaticity.max()), 1e-8) * 255, 0, 255)
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0]
        hue = float(hsv[0])
        if hue >= 75:
            return "blue"
        if hue >= 23:
            return "yellow_green"
        return "brown"

    def fit(self, support_images: Sequence) -> "CableSceneGrounder":
        if len(support_images) < 2:
            raise ValueError("Cable grounding requires at least two normal supports")
        prepared = [self.prepare(image) for image in support_images]
        # Bootstrap from the relation's canonical triangle, then replace every
        # numeric slot/radius with a statistic learned from the supports.
        initial = [self._locate(image)[0] for image in prepared]
        self.slots = np.median(
            np.stack([circles[:, :2] / self.image_size for circles in initial]), axis=0
        )
        self.slot_radii = np.median(
            np.stack([circles[:, 2] / self.image_size for circles in initial]), axis=0
        )
        refined = [self._locate(image)[0] for image in prepared]
        colors = np.stack([
            np.stack([self._chromaticity(image, circle) for circle in circles])
            for image, circles in zip(prepared, refined)
        ])
        self.role_color_prototypes = np.median(colors, axis=0)
        # Keep the four support appearances as a tiny non-parametric identity
        # bank.  Median-only prototypes can merge yellow/brown under a strong
        # specular highlight even though each support shows three distinct roles.
        self.role_color_exemplars = np.transpose(colors, (1, 0, 2))
        names = [self._identity_name(item) for item in self.role_color_prototypes]
        if len(set(names)) == 3:
            self.semantic_roles = {name: index for index, name in enumerate(names)}
        else:
            # The camera pose is stable in this class.  This fallback is only a
            # semantic label assignment; all actual colors come from support pixels.
            self.semantic_roles = {"yellow_green": 0, "blue": 1, "brown": 2}
        core_values = [
            self._core_texture(image, circle)
            for image, circles in zip(prepared, refined)
            for circle in circles
        ]
        self.core_reference = float(np.median(core_values))
        anchor_strengths = [self._outer_anchor_strength(image) for image in prepared]
        self.anchor_strength_floor = float(max(1e-4, 0.35 * min(anchor_strengths)))
        self.fitted = True
        return self

    def _facts_from_prepared(self, image: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if not self.fitted:
            raise RuntimeError("CableSceneGrounder must be fitted before extraction")
        circles, assignment_errors, matched_count = self._locate(image)
        colors = np.stack([self._chromaticity(image, circle) for circle in circles])
        distance = self._identity_distance(colors)
        identity_ids = np.argmin(distance, axis=1)
        permutation, bijection_mean, bijection_max = _assignment(distance)
        n = float(self.image_size)
        centers = circles[:, :2] / n
        radii = circles[:, 2] / n
        pair_distances = np.asarray([
            np.linalg.norm(centers[left] - centers[right])
            for left, right in ((0, 1), (0, 2), (1, 2))
        ])
        first_edge = centers[1] - centers[0]
        second_edge = centers[2] - centers[0]
        triangle_area = abs(float(
            first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
        )) / 2.0
        max_extent = max(
            float(np.linalg.norm(center - np.asarray((0.5, 0.5))) + radius)
            for center, radius in zip(centers, radii)
        )
        core_scores = np.asarray([
            self._core_texture(image, circle) for circle in circles
        ])
        anchor_strength = self._outer_anchor_strength(image)
        anchor_valid = float(
            np.isfinite(anchor_strength)
            and anchor_strength >= self.anchor_strength_floor
        )
        facts: Dict[str, float] = {
            "conductor_count_error": abs(float(matched_count) - 3.0) / 3.0,
            "grounding_assignment_error": float(np.mean(assignment_errors)),
            "color_duplicate_count": float(3 - len(set(identity_ids.tolist()))),
            "color_bijection_cost": float(bijection_mean),
            "color_bijection_max": float(bijection_max),
            "role_color_cost": float(np.mean(np.diag(distance))),
            "triangle_irregularity": float(np.std(pair_distances) / max(np.mean(pair_distances), 1e-8)),
            "triangle_degeneracy": float(max(0.0, 0.025 - triangle_area)),
            "size_irregularity": float(np.std(radii) / max(np.mean(radii), 1e-8)),
            "group_center_offset": float(np.linalg.norm(centers.mean(axis=0) - 0.5)),
            "containment_violation": float(max(0.0, max_extent - 0.49)),
            "core_presence_deficit": float(max(
                0.0,
                (self.core_reference - float(core_scores.min()))
                / max(self.core_reference, 1e-8),
            )),
            "outer_anchor_strength": float(anchor_strength),
            "anchor_valid": anchor_valid,
        }
        # ``permutation`` maps detected row -> prototype column.  Relative
        # relations need the inverse map, prototype identity -> detected row.
        prototype_to_detected = _prototype_to_detected(permutation)
        identity_centers = {
            identity: centers[prototype_to_detected[prototype_index]]
            for identity, prototype_index in self.semantic_roles.items()
        }
        for first, second in itertools.permutations(sorted(identity_centers), 2):
            a, b = identity_centers[first], identity_centers[second]
            facts[f"relative:{first}:{second}:above"] = float(max(0.0, a[1] - b[1]))
            facts[f"relative:{first}:{second}:below"] = float(max(0.0, b[1] - a[1]))
            facts[f"relative:{first}:{second}:left_of"] = float(max(0.0, a[0] - b[0]))
            facts[f"relative:{first}:{second}:right_of"] = float(max(0.0, b[0] - a[0]))
        debug = {
            "circles": circles.tolist(),
            "slot_assignment_errors": assignment_errors.tolist(),
            "matched_conductor_count": matched_count,
            "chromaticities": colors.tolist(),
            "nearest_identity_indices": identity_ids.tolist(),
            "identity_assignment": list(permutation),
            "semantic_roles": dict(self.semantic_roles),
            "outer_anchor_strength": float(anchor_strength),
            "anchor_strength_floor": float(self.anchor_strength_floor),
        }
        return facts, debug

    def extract(self, image) -> Tuple[Dict[str, float], Dict[str, Any]]:
        return self._facts_from_prepared(self.prepare(image))

    def state_dict(self) -> Dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Cannot serialize an unfitted grounder")
        return {
            "kind": "cable_circle_chromaticity_v1",
            "image_size": self.image_size,
            "slots": self.slots.tolist(),
            "slot_radii": self.slot_radii.tolist(),
            "role_color_prototypes": self.role_color_prototypes.tolist(),
            "role_color_exemplars": self.role_color_exemplars.tolist(),
            "semantic_roles": dict(self.semantic_roles),
            "core_reference": self.core_reference,
            "hue_cost_weight": self.hue_cost_weight,
            "anchor_strength_floor": self.anchor_strength_floor,
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        if state.get("kind") != "cable_circle_chromaticity_v1":
            raise ValueError(f"Unsupported cable grounder: {state.get('kind')}")
        self.image_size = int(state["image_size"])
        self.slots = np.asarray(state["slots"], dtype=float)
        self.slot_radii = np.asarray(state["slot_radii"], dtype=float)
        self.role_color_prototypes = np.asarray(state["role_color_prototypes"], dtype=float)
        self.role_color_exemplars = np.asarray(
            state.get("role_color_exemplars", self.role_color_prototypes[:, None, :]),
            dtype=float,
        )
        self.semantic_roles = {str(key): int(value) for key, value in state["semantic_roles"].items()}
        self.core_reference = float(state["core_reference"])
        self.hue_cost_weight = float(state.get("hue_cost_weight", 0.0))
        self.anchor_strength_floor = float(state.get("anchor_strength_floor", 0.0))
        self.fitted = True


def _warp(image: np.ndarray, angle: float = 0.0, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    matrix[:, 2] += (tx * width, ty * height)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _copy_disk(
    image: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    source_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    changed = image.copy()
    source_image = image if source_image is None else source_image
    sx, sy, sr = (int(round(value)) for value in source)
    tx, ty, tr = (int(round(value)) for value in target)
    source_radius = max(4, int(round(sr * 1.02)))
    target_radius = max(4, int(round(tr * 1.02)))
    padded = cv2.copyMakeBorder(
        source_image, source_radius, source_radius, source_radius, source_radius,
        cv2.BORDER_REFLECT_101,
    )
    patch = padded[
        sy:sy + 2 * source_radius + 1,
        sx:sx + 2 * source_radius + 1,
    ]
    diameter = 2 * target_radius + 1
    patch = cv2.resize(patch, (diameter, diameter), interpolation=cv2.INTER_LINEAR)
    yy, xx = np.mgrid[:diameter, :diameter]
    alpha = (np.hypot(xx - target_radius, yy - target_radius) <= target_radius).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)[..., None]
    x0, y0 = tx - target_radius, ty - target_radius
    x1, y1 = x0 + diameter, y0 + diameter
    if x0 < 0 or y0 < 0 or x1 > changed.shape[1] or y1 > changed.shape[0]:
        return changed
    destination = changed[y0:y1, x0:x1].astype(np.float32)
    changed[y0:y1, x0:x1] = np.clip(
        alpha * patch.astype(np.float32) + (1.0 - alpha) * destination, 0, 255
    ).astype(np.uint8)
    return changed


def build_relational_calibration_views(
    support_images: Sequence,
    grounder: CableSceneGrounder,
) -> List[RelationalCalibrationView]:
    """Create relation-preserving nuisances and predicate counterfactuals."""
    views: List[RelationalCalibrationView] = []
    for source_index, source in enumerate(support_images):
        image = grounder.prepare(source)
        normal_variants = [
            ("identity", image),
            ("brightness_082", np.clip(image.astype(float) * 0.82, 0, 255).astype(np.uint8)),
            ("brightness_118", np.clip(image.astype(float) * 1.18, 0, 255).astype(np.uint8)),
            ("contrast_085", np.clip((image.astype(float) - 128) * 0.85 + 128, 0, 255).astype(np.uint8)),
            ("contrast_115", np.clip((image.astype(float) - 128) * 1.15 + 128, 0, 255).astype(np.uint8)),
            ("blur", cv2.GaussianBlur(image, (5, 5), 0.8)),
            ("rotate_m5", _warp(image, angle=-5.0)),
            ("rotate_p5", _warp(image, angle=5.0)),
            ("translate_left", _warp(image, tx=-0.025)),
            ("translate_right", _warp(image, tx=0.025)),
            ("translate_up", _warp(image, ty=-0.025)),
            ("translate_down", _warp(image, ty=0.025)),
        ]
        views.extend(RelationalCalibrationView(item, source_index, name, False, "normal")
                     for name, item in normal_variants)

        circles, _, _ = grounder._locate(image)
        for target in range(3):
            source_role = (target + 1) % 3
            duplicate = _copy_disk(image, circles[source_role], circles[target])
            views.append(RelationalCalibrationView(
                duplicate, source_index, f"duplicate_identity_{source_role}_to_{target}",
                True, "identity",
            ))
        for first, second in ((0, 1), (0, 2), (1, 2)):
            # Both source patches must come from the original image.  Reading
            # the second source from the already modified image would create a
            # duplicate identity rather than a true two-way swap.
            swapped = _copy_disk(
                image, circles[first], circles[second], source_image=image
            )
            swapped = _copy_disk(
                swapped, circles[second], circles[first], source_image=image
            )
            views.append(RelationalCalibrationView(
                swapped, source_index, f"swap_roles_{first}_{second}", True, "position",
            ))
        # Remove one complete conductor using the median filler color between parts.
        yy, xx = np.mgrid[:grounder.image_size, :grounder.image_size]
        filler_mask = np.ones(image.shape[:2], dtype=bool)
        for cx, cy, radius in circles:
            filler_mask &= np.hypot(xx - cx, yy - cy) > 1.05 * radius
        filler_mask &= np.hypot(xx - grounder.image_size / 2, yy - grounder.image_size / 2) < 0.40 * grounder.image_size
        fill = np.median(image[filler_mask], axis=0) if filler_mask.any() else np.median(image.reshape(-1, 3), axis=0)
        for target, (cx, cy, radius) in enumerate(circles):
            removed = image.copy()
            mask = np.hypot(xx - cx, yy - cy) <= 1.02 * radius
            removed[mask] = np.clip(fill, 0, 255).astype(np.uint8)
            views.append(RelationalCalibrationView(
                removed, source_index, f"remove_conductor_{target}", True, "count",
            ))
        for target, (cx, cy, radius) in enumerate(circles):
            removed_core = image.copy()
            ring = (np.hypot(xx - cx, yy - cy) >= 0.55 * radius) & (
                np.hypot(xx - cx, yy - cy) <= 0.82 * radius
            )
            fill_core = np.median(image[ring], axis=0)
            center = np.hypot(xx - cx, yy - cy) <= 0.50 * radius
            removed_core[center] = np.clip(fill_core, 0, 255).astype(np.uint8)
            views.append(RelationalCalibrationView(
                removed_core, source_index, f"remove_core_{target}", True, "core",
            ))
    return views


def _rule_fact(rule: Mapping[str, Any]) -> Tuple[Optional[str], str]:
    predicate = str(rule.get("predicate"))
    subject = rule.get("subject")
    object_name = rule.get("object")
    attribute = rule.get("attribute")
    relation = rule.get("relation")
    target = rule.get("target")
    values = tuple(rule.get("values") or ())
    empty_tail = object_name is None and attribute is None and relation is None and not values

    # Every accepted tuple below has a literal executable meaning.  Rejecting
    # unsupported arguments is essential: silently mapping target=1 to the
    # hard-coded three-conductor fact would only imitate symbolic reasoning.
    if (
        predicate == "count_eq" and subject == "conductor"
        and target == 3 and empty_tail
    ):
        return "conductor_count_error", "count"
    if (
        predicate == "all_inside" and subject == "conductor"
        and object_name == "outer_sheath" and attribute is None
        and relation is None and target is None and not values
    ):
        return "containment_violation", "containment"
    if (
        predicate == "one_to_one_inside" and subject == "copper_core"
        and object_name == "conductor" and attribute is None
        and relation is None and target is None and not values
    ):
        return "core_presence_deficit", "core"
    if (
        predicate == "distinct_attribute_count_eq" and subject == "conductor"
        and object_name is None and attribute == "color_identity"
        and relation is None and target == 3 and not values
    ):
        return "color_duplicate_count", "identity"
    if (
        predicate == "attribute_multiset_eq" and subject == "conductor"
        and object_name is None and attribute == "color_identity"
        and relation is None and target is None
        and tuple(sorted(values)) == ("blue", "brown", "yellow_green")
    ):
        return "color_duplicate_count", "identity"
    shape_rule = (
        subject == "conductor" and object_name is None and attribute is None
        and relation is None and target is None and not values
    )
    if predicate == "triangular_layout" and shape_rule:
        return "triangle_degeneracy", "geometry"
    if predicate == "similar_size" and shape_rule:
        return "size_irregularity", "geometry"
    if predicate == "centered_group" and shape_rule:
        return "group_center_offset", "geometry"
    if (
        predicate == "relative_position"
        and subject in {"blue", "brown", "yellow_green"}
        and object_name in {"blue", "brown", "yellow_green"}
        and subject != object_name
        and attribute is None and relation in {"above", "below", "left_of", "right_of"}
        and target is None and not values
    ):
        # Canonicalize logical inverses so ``A right_of B`` and
        # ``B left_of A`` cannot be counted as independent evidence.
        if relation == "below":
            subject, object_name, relation = object_name, subject, "above"
        elif relation == "right_of":
            subject, object_name, relation = object_name, subject, "left_of"
        key = f"relative:{subject}:{object_name}:{relation}"
        return key, "position"
    return None, "unsupported"


def _safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2 or float(np.ptp(scores)) < 1e-12:
        return 0.5
    return float(roc_auc_score(labels, scores))


def _continuous_constraint(values: np.ndarray, fact: str) -> Dict[str, float]:
    tolerance = {
        "triangle_irregularity": 0.025,
        "triangle_degeneracy": 0.003,
        "size_irregularity": 0.025,
        "group_center_offset": 0.015,
        "containment_violation": 0.005,
        "core_presence_deficit": 0.05,
    }.get(fact, 0.01)
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = max(float((q75 - q25) / 1.349), tolerance, 1e-6)
    return {
        "upper": float(max(values.max(), np.quantile(values, 0.99)) + tolerance),
        "soft_scale": scale,
        "normal_median": median,
        "normal_max": float(values.max()),
    }


def rule_violation(rule: Mapping[str, Any], facts: Mapping[str, float]) -> float:
    value = float(facts.get(str(rule["fact"]), 0.0))
    if rule["fact"] == "color_duplicate_count":
        return float(np.clip(value, 0.0, 1.0))
    excess = max(0.0, value - float(rule["upper"]))
    return float(math.tanh(excess / max(float(rule["soft_scale"]), 1e-8)))


def compile_relational_rules(
    proposal: Mapping[str, Any],
    normal_rows: Sequence[Mapping[str, float]],
    pseudo_rows: Sequence[Mapping[str, float]],
    pseudo_families: Sequence[str],
    normal_groups: Sequence[int],
    pseudo_groups: Sequence[int],
    allowed_families: Sequence[str] = ("identity",),
    minimum_cv_auc: float = 0.75,
    maximum_normal_fpr: float = 0.10,
    crossfit_folds: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compile only Qwen predicates that survive support-only counterfactual tests."""
    allowed = set(allowed_families)
    audits: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    normal_groups = np.asarray(normal_groups, dtype=int)
    pseudo_groups = np.asarray(pseudo_groups, dtype=int)
    pseudo_families = np.asarray(pseudo_families)
    all_groups = sorted(set(normal_groups.tolist()) & set(pseudo_groups.tolist()))

    for proposal_rule in proposal.get("rules", []):
        fact, family = _rule_fact(proposal_rule)
        audit = {
            "proposal_id": proposal_rule.get("id"),
            "predicate": proposal_rule.get("predicate"),
            "fact": fact,
            "family": family,
            "accepted": False,
        }
        if fact is None:
            audit["rejection"] = "predicate has no cable grounder implementation"
            audits.append(audit)
            continue
        if family not in allowed:
            audit["rejection"] = "family disabled by the registered control experiment"
            audits.append(audit)
            continue
        if not normal_rows or fact not in normal_rows[0]:
            audit["rejection"] = "grounder did not emit the required fact"
            audits.append(audit)
            continue
        relevant = np.asarray([
            item == family or (family == "identity" and item == "count")
            for item in pseudo_families
        ])
        if not relevant.any():
            audit["rejection"] = "no matching support-derived counterfactuals"
            audits.append(audit)
            continue
        normal_values = np.asarray([float(row[fact]) for row in normal_rows])
        pseudo_values = np.asarray([float(row[fact]) for row in pseudo_rows])
        if fact.startswith("relative:"):
            semantic_q90 = float(np.quantile(normal_values, 0.90))
            audit["semantic_support_violation_q90"] = semantic_q90
            if semantic_q90 > 0.03:
                audit["rejection"] = "relation contradicted by known-normal support"
                audits.append(audit)
                continue
        full_constraint = (
            {"upper": 0.0, "soft_scale": 0.5, "normal_median": 0.0,
             "normal_max": float(normal_values.max())}
            if fact == "color_duplicate_count"
            else _continuous_constraint(normal_values, fact)
        )
        cv_labels, cv_scores = [], []
        if crossfit_folds:
            for fold in crossfit_folds:
                train_rows = list(fold.get("train_normal_rows", []))
                held_normal_rows = list(fold.get("held_normal_rows", []))
                held_pseudo_rows = list(fold.get("held_pseudo_rows", []))
                held_families = list(fold.get("held_pseudo_families", []))
                if not train_rows or fact not in train_rows[0]:
                    continue
                train = np.asarray([float(row[fact]) for row in train_rows])
                constraint = (
                    {"upper": 0.0, "soft_scale": 0.5}
                    if fact == "color_duplicate_count"
                    else _continuous_constraint(train, fact)
                )
                fold_rule = {"fact": fact, **constraint}
                cv_labels.extend([0] * len(held_normal_rows))
                cv_scores.extend([
                    rule_violation(fold_rule, row) for row in held_normal_rows
                ])
                for row, held_family in zip(held_pseudo_rows, held_families):
                    is_relevant = held_family == family or (
                        family == "identity" and held_family == "count"
                    )
                    if is_relevant:
                        cv_labels.append(1)
                        cv_scores.append(rule_violation(fold_rule, row))
            audit["cv_protocol"] = "leave_one_support_out_refit_grounder"
        else:
            for held_out in all_groups:
                train = normal_values[normal_groups != held_out]
                if not len(train):
                    continue
                constraint = (
                    {"upper": 0.0, "soft_scale": 0.5}
                    if fact == "color_duplicate_count"
                    else _continuous_constraint(train, fact)
                )
                fold_rule = {"fact": fact, **constraint}
                held_normal = np.where(normal_groups == held_out)[0]
                held_pseudo = np.where((pseudo_groups == held_out) & relevant)[0]
                cv_labels.extend([0] * len(held_normal))
                cv_scores.extend([
                    rule_violation(fold_rule, normal_rows[index]) for index in held_normal
                ])
                cv_labels.extend([1] * len(held_pseudo))
                cv_scores.extend([
                    rule_violation(fold_rule, pseudo_rows[index]) for index in held_pseudo
                ])
            audit["cv_protocol"] = "legacy_rows_grouped_by_support"
        cv_labels_array = np.asarray(cv_labels, dtype=int)
        cv_scores_array = np.asarray(cv_scores, dtype=float)
        cv_auc = _safe_auc(cv_labels_array, cv_scores_array)
        normal_cv = cv_scores_array[cv_labels_array == 0]
        pseudo_cv = cv_scores_array[cv_labels_array == 1]
        normal_fpr = float(np.mean(normal_cv > 0.0)) if len(normal_cv) else 1.0
        trigger = float(np.mean(pseudo_cv > 0.0)) if len(pseudo_cv) else 0.0
        audit.update({
            "cv_auc": cv_auc,
            "cv_normal_fpr": normal_fpr,
            "cv_counterfactual_trigger_rate": trigger,
        })
        if cv_auc < minimum_cv_auc:
            audit["rejection"] = f"cv_auc<{minimum_cv_auc}"
        elif normal_fpr > maximum_normal_fpr:
            audit["rejection"] = f"normal_fpr>{maximum_normal_fpr}"
        elif trigger < 0.40:
            audit["rejection"] = "counterfactual trigger rate<0.40"
        else:
            reliability = max(0.0, (cv_auc - 0.5) * 2.0) * (1.0 - normal_fpr)
            semantic = math.sqrt(
                float(proposal_rule.get("importance", 0.8))
                * float(proposal_rule.get("confidence", 0.8))
            )
            compiled = {
                **{key: value for key, value in proposal_rule.items()},
                "fact": fact,
                "family": family,
                **full_constraint,
                "reliability": reliability,
                "weight": reliability * semantic,
            }
            audit["accepted"] = True
            candidates.append(compiled)
        audits.append(audit)

    # Multiple language formulations of the same executable fact must not be
    # counted repeatedly.  Keep the empirically strongest version only.
    selected: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        key = str(candidate["fact"])
        if key not in selected or candidate["weight"] > selected[key]["weight"]:
            selected[key] = candidate
    rules = sorted(selected.values(), key=lambda item: item["weight"], reverse=True)
    total = sum(float(rule["weight"]) for rule in rules)
    if total > 0:
        for rule in rules:
            rule["weight"] = float(rule["weight"] / total)
    return rules, audits


class CableRelationalRuleEngine:
    """Static runtime for a compiled v3 cable rule library."""

    def __init__(self, library: Mapping[str, Any]):
        if str(library.get("schema_version")) != "3.0-relational":
            raise ValueError("Expected a schema_version=3.0-relational library")
        self.library = dict(library)
        self.grounder = CableSceneGrounder(library["grounder"])
        self.rules = list(library.get("rules", []))
        self.fusion = dict(library.get("fusion", {}))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CableRelationalRuleEngine":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: Union[str, Path]) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.library, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output

    def evaluate(self, image) -> Tuple[float, List[Dict[str, Any]], Dict[str, float], Dict[str, Any]]:
        facts, scene = self.grounder.extract(image)
        evidence = []
        for rule in self.rules:
            score = rule_violation(rule, facts)
            evidence.append({
                "rule_id": rule["id"],
                "predicate": rule["predicate"],
                "fact": rule["fact"],
                "value": float(facts[rule["fact"]]),
                "violation": score,
                "weight": float(rule["weight"]),
                "reason": rule.get("reason", ""),
            })
        evidence.sort(key=lambda item: item["violation"] * item["weight"], reverse=True)
        if not evidence or facts.get("anchor_valid", 0.0) <= 0:
            return 0.0, evidence, facts, scene
        weighted = sum(item["weight"] * item["violation"] for item in evidence)
        maximum = max(item["violation"] for item in evidence)
        score = float(0.75 * maximum + 0.25 * weighted)
        return score, evidence, facts, scene

    def decision_score(self, baseline_score: float, relation_score: float) -> float:
        if not self.rules or relation_score <= 0.0 or not self.fusion.get("enabled", False):
            return float(baseline_score)
        return float(baseline_score + float(self.fusion["alpha"]) * relation_score)


def build_relational_library(
    dataset: str,
    class_name: str,
    support_paths: Sequence[str],
    grounder: CableSceneGrounder,
    proposal: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    normal_baseline_scores: Sequence[float],
    pseudo_baseline_scores: Sequence[float],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    normal_scores = np.asarray(normal_baseline_scores, dtype=float)
    pseudo_scores = np.asarray(pseudo_baseline_scores, dtype=float)
    normal_q90 = float(np.quantile(normal_scores, 0.90))
    normal_q10 = float(np.quantile(normal_scores, 0.10))
    pseudo_q10 = float(np.quantile(pseudo_scores, 0.10))
    # This is a support-only score-scale cap, not a learned defect classifier.
    # Taking an absolute gap avoids disabling the symbolic branch precisely when
    # the baseline misses a pseudo anomaly (pseudo_q10 < normal_q90).
    reference_scale = max(
        abs(pseudo_q10 - normal_q90),
        normal_q90 - normal_q10,
        1e-6,
    )
    alpha = float(0.5 * reference_scale)
    return {
        "schema_version": "3.0-relational",
        "dataset": dataset,
        "class_name": class_name,
        "support_paths": list(support_paths),
        "grounder": grounder.state_dict(),
        "qwen": {
            "model": proposal.get("model"),
            "normality_summary": proposal.get("normality_summary", ""),
            "entities": proposal.get("entities", []),
            "candidate_rules": proposal.get("rules", []),
        },
        "rules": [dict(rule) for rule in rules],
        "candidate_audit": [dict(item) for item in audits],
        "fusion": {
            "enabled": bool(rules and alpha > 0),
            "kind": "support_calibrated_symbolic_scale_cap",
            "alpha": alpha,
            "normal_baseline_q10": normal_q10,
            "normal_baseline_q90": normal_q90,
            "pseudo_baseline_q10": pseudo_q10,
            "reference_scale": reference_scale,
            "learned_from_test_labels": False,
            "note": "scale calibration only; rule validity is established by grouped relational CV",
            "formula": "decision = baseline + alpha * relational_violation",
        },
        "provenance": dict(metadata),
        "test_labels_used_for_training_or_selection": False,
    }
