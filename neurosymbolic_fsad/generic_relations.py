"""Generic support-fitted grounding for repeated-part relational rules.

This module contains no LLM dependency.  It compiles a Qwen proposal into a
small deterministic foreground-component program, verifies it by leave-one-
support-out counterfactual tests, and executes the frozen program at runtime.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score


FACT_BY_PREDICATE = {
    "count_eq": "count_error",
    "arrangement_eq": "layout_error",
    "ordered_spacing": "spacing_error",
    "similar_size": "size_error",
    "similar_orientation": "shape_error",
    "appearance_consistency": "appearance_error",
}


def _rgb(image: Union[str, Path, Image.Image, np.ndarray], size: int) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            array = np.asarray(handle.convert("RGB"))
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.shape[-1] == 4:
            array = array[..., :3]
    return cv2.resize(array.astype(np.uint8), (size, size), interpolation=cv2.INTER_AREA)


def _background_lab(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    width = max(3, image.shape[0] // 14)
    border = np.concatenate([
        lab[:width].reshape(-1, 3), lab[-width:].reshape(-1, 3),
        lab[:, :width].reshape(-1, 3), lab[:, -width:].reshape(-1, 3),
    ])
    return np.median(border, axis=0)


def _normalise_map(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    positive = array[array > 0]
    if not positive.size:
        return np.zeros_like(array, dtype=np.float32)
    scale = max(float(np.quantile(positive, 0.99)), 1e-6)
    return np.clip(array / scale, 0.0, 1.0).astype(np.float32)


def _slot_mask(size: int, centroid: np.ndarray, area: float) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.float32)
    center = tuple(np.rint(np.asarray(centroid) * size).astype(int))
    radius = max(3, int(round(np.sqrt(max(float(area), 1e-5) / np.pi) * size)))
    cv2.circle(result, center, radius, 1.0, -1)
    return result


def _component_descriptor(image: np.ndarray, labels: np.ndarray, label: int, stats, centroid):
    mask = labels == label
    area = float(stats[cv2.CC_STAT_AREA]) / float(image.shape[0] * image.shape[1])
    contour_rows = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contour_rows, key=cv2.contourArea) if contour_rows else None
    hu = np.zeros(7, dtype=np.float32)
    if contour is not None and len(contour) >= 3:
        raw = cv2.HuMoments(cv2.moments(contour)).reshape(-1)
        hu = -np.sign(raw) * np.log10(np.abs(raw) + 1e-12)
        hu = np.clip(hu, -20.0, 20.0).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean_lab = np.mean(lab[mask], axis=0) if np.any(mask) else np.zeros(3)
    coordinates = np.column_stack(np.nonzero(mask)).astype(np.float32)
    orientation = 0.0
    if len(coordinates) >= 3:
        covariance = np.cov(coordinates[:, ::-1].T)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        orientation = float(np.arctan2(axis[1], axis[0]))
    orientation_vec = np.asarray(
        [np.cos(2.0 * orientation), np.sin(2.0 * orientation)], dtype=np.float32
    )
    x, y = float(centroid[0]) / image.shape[1], float(centroid[1]) / image.shape[0]
    return {
        "label": int(label),
        "area": area,
        "centroid": np.asarray([x, y], dtype=np.float32),
        "hu": hu,
        "orientation": orientation_vec,
        "lab": np.asarray(mean_lab, dtype=np.float32),
        "bbox": [int(stats[cv2.CC_STAT_LEFT]), int(stats[cv2.CC_STAT_TOP]),
                 int(stats[cv2.CC_STAT_WIDTH]), int(stats[cv2.CC_STAT_HEIGHT])],
        "mask": mask,
    }


class GenericComponentGrounder:
    """Support-fitted foreground component and slot-relation grounder."""

    def __init__(
        self,
        expected_count: int,
        arrangement: str,
        image_size: int = 256,
        state: Optional[Mapping[str, Any]] = None,
    ):
        self.expected_count = int(expected_count)
        self.arrangement = str(arrangement)
        self.image_size = int(image_size)
        self.threshold = 24.0
        self.min_area = 0.002
        self.morph = 3
        self.prototype: Dict[str, np.ndarray] = {}
        self.support_match_rate = 0.0
        if state is not None:
            self.load_state(state)

    def _segment(self, image: np.ndarray, params=None, full_descriptors: bool = True):
        threshold, min_area, morph = params or (self.threshold, self.min_area, self.morph)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        background = _background_lab(image)
        distance = np.linalg.norm(lab - background[None, None, :], axis=2)
        mask = (distance > float(threshold)).astype(np.uint8)
        if int(morph) > 1:
            kernel = np.ones((int(morph), int(morph)), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        components = []
        pixels = float(image.shape[0] * image.shape[1])
        for label in range(1, count):
            fraction = float(stats[label, cv2.CC_STAT_AREA]) / pixels
            if fraction < float(min_area) or fraction > 0.82:
                continue
            if full_descriptors:
                components.append(_component_descriptor(
                    image, labels, label, stats[label], centroids[label]
                ))
            else:
                components.append({
                    "label": int(label),
                    "area": fraction,
                    "centroid": np.asarray([
                        float(centroids[label][0]) / image.shape[1],
                        float(centroids[label][1]) / image.shape[0],
                    ], dtype=np.float32),
                })
        components.sort(key=lambda item: item["area"], reverse=True)
        if len(components) > max(32, self.expected_count * 4):
            components = components[:max(32, self.expected_count * 4)]
        return components, labels, background

    def _ordered(self, components: Sequence[Mapping[str, Any]]):
        if self.arrangement == "column":
            return sorted(components, key=lambda item: (item["centroid"][1], item["centroid"][0]))
        if self.arrangement == "grid":
            return sorted(components, key=lambda item: (round(float(item["centroid"][1]), 1), item["centroid"][0]))
        if self.arrangement == "ring":
            center = np.mean([item["centroid"] for item in components], axis=0)
            return sorted(components, key=lambda item: math.atan2(
                item["centroid"][1] - center[1], item["centroid"][0] - center[0]
            ))
        return sorted(components, key=lambda item: (item["centroid"][0], item["centroid"][1]))

    def fit(self, support_images: Sequence) -> "GenericComponentGrounder":
        images = [_rgb(item, self.image_size) for item in support_images]
        candidates = []
        for threshold in (10, 14, 18, 24, 32, 42, 56, 72):
            for min_area in (0.0005, 0.001, 0.002, 0.004, 0.008, 0.015):
                for morph in (1, 3, 5):
                    rows = [self._segment(
                        image, (threshold, min_area, morph), full_descriptors=False
                    )[0]
                            for image in images]
                    counts = np.asarray([len(row) for row in rows], dtype=float)
                    match = float(np.mean(counts == self.expected_count))
                    count_loss = float(np.mean(np.abs(counts - self.expected_count)))
                    area_cv = 5.0
                    if all(len(row) >= self.expected_count for row in rows):
                        selected = [self._ordered(row[:self.expected_count]) for row in rows]
                        totals = np.asarray([sum(item["area"] for item in row) for row in selected])
                        area_cv = float(np.std(totals) / max(np.mean(totals), 1e-8))
                    objective = 20.0 * (1.0 - match) + count_loss + area_cv
                    candidates.append((objective, -match, threshold, min_area, morph, rows))
        candidates.sort(key=lambda item: item[:5])
        _, neg_match, self.threshold, self.min_area, self.morph, _ = candidates[0]
        self.support_match_rate = float(-neg_match)
        rows = [self._segment(image)[0] for image in images]
        valid = [self._ordered(row[:self.expected_count]) for row in rows
                 if len(row) == self.expected_count]
        if not valid:
            self.prototype = {}
            return self
        initial = valid[0]
        slot_centroids = np.stack([item["centroid"] for item in initial])
        aligned = []
        for row in valid:
            current = np.stack([item["centroid"] for item in row])
            cost = np.linalg.norm(slot_centroids[:, None, :] - current[None, :, :], axis=2)
            source, target = linear_sum_assignment(cost)
            ordered = [None] * self.expected_count
            for src, dst in zip(source, target):
                ordered[int(src)] = row[int(dst)]
            aligned.append(ordered)
        self.prototype = {
            "centroid": np.median(np.stack([[item["centroid"] for item in row] for row in aligned]), axis=0),
            "area": np.median(np.stack([[item["area"] for item in row] for row in aligned]), axis=0),
            "hu": np.median(np.stack([[item["hu"] for item in row] for row in aligned]), axis=0),
            "orientation": np.median(np.stack([[item["orientation"] for item in row] for row in aligned]), axis=0),
            "lab": np.median(np.stack([[item["lab"] for item in row] for row in aligned]), axis=0),
        }
        return self

    def extract_spatial(
        self, image
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, Any]]:
        prepared = _rgb(image, self.image_size)
        components, labels, background = self._segment(prepared)
        count = len(components)
        facts = {
            "count_error": float(abs(count - self.expected_count) / max(self.expected_count, 1)),
            "layout_error": 1.0,
            "spacing_error": 1.0,
            "size_error": 1.0,
            "shape_error": 1.0,
            "appearance_error": 1.0,
        }
        maps = {
            name: np.zeros((self.image_size, self.image_size), dtype=np.float32)
            for name in facts
        }
        assignment = []
        matched = []
        unmatched_components = set(range(count))
        unmatched_slots = set(range(self.expected_count))
        if self.prototype and count:
            current = np.stack([item["centroid"] for item in components])
            proto = self.prototype["centroid"]
            cost = np.linalg.norm(proto[:, None, :] - current[None, :, :], axis=2)
            source, target = linear_sum_assignment(cost)
            for src, dst in zip(source, target):
                assignment.append([int(src), int(dst)])
                unmatched_slots.discard(int(src))
                unmatched_components.discard(int(dst))
                matched.append((int(src), int(dst), components[int(dst)]))

        # Count is localized only by genuinely missing expected slots and
        # genuinely extra current components, not by a union that also paints
        # normal present parts.
        if self.prototype:
            for slot in unmatched_slots:
                maps["count_error"] = np.maximum(
                    maps["count_error"],
                    _slot_mask(
                        self.image_size, self.prototype["centroid"][slot],
                        self.prototype["area"][slot],
                    ),
                )
        for component_index in unmatched_components:
            component = components[component_index]
            maps["count_error"][component["mask"]] = 1.0

        if matched:
            slot_indices = np.asarray([row[0] for row in matched], dtype=int)
            current_indices = np.asarray([row[1] for row in matched], dtype=int)
            ordered = [row[2] for row in matched]
            proto = self.prototype["centroid"][slot_indices]
            centroids = np.stack([item["centroid"] for item in ordered])
            areas = np.asarray([item["area"] for item in ordered])
            hu = np.stack([item["hu"] for item in ordered])
            orientation = np.stack([item["orientation"] for item in ordered])
            lab = np.stack([item["lab"] for item in ordered])
            layout_each = np.linalg.norm(centroids - proto, axis=1)
            facts["layout_error"] = float(np.mean(layout_each))
            for local_index, (slot, _, component) in enumerate(matched):
                weight = float(layout_each[local_index])
                maps["layout_error"][component["mask"]] = max(weight, 1e-6)
                start = tuple(np.rint(component["centroid"] * self.image_size).astype(int))
                stop = tuple(np.rint(self.prototype["centroid"][slot] * self.image_size).astype(int))
                cv2.line(maps["layout_error"], start, stop, max(weight, 1e-6), 2)

            if len(matched) >= 2:
                distance_now = np.linalg.norm(
                    centroids[:, None] - centroids[None, :], axis=2
                )
                distance_ref = np.linalg.norm(proto[:, None] - proto[None, :], axis=2)
                spacing_residual = np.abs(distance_now - distance_ref)
                facts["spacing_error"] = float(np.mean(spacing_residual))
                for left in range(len(matched)):
                    for right in range(left + 1, len(matched)):
                        weight = float(spacing_residual[left, right])
                        p0 = tuple(np.rint(centroids[left] * self.image_size).astype(int))
                        p1 = tuple(np.rint(centroids[right] * self.image_size).astype(int))
                        cv2.line(maps["spacing_error"], p0, p1, max(weight, 1e-6), 3)

            area_ref = self.prototype["area"][slot_indices]
            size_each = np.abs(np.log((areas + 1e-6) / (area_ref + 1e-6)))
            facts["size_error"] = float(np.mean(size_each))
            for value, component in zip(size_each, ordered):
                maps["size_error"][component["mask"]] = max(float(value), 1e-6)
            orientation_ref = self.prototype.get("orientation")
            if orientation_ref is not None:
                orientation_ref = orientation_ref[slot_indices]
                current_unit = orientation / np.maximum(np.linalg.norm(orientation, axis=1, keepdims=True), 1e-8)
                reference_unit = orientation_ref / np.maximum(
                    np.linalg.norm(orientation_ref, axis=1, keepdims=True), 1e-8
                )
                cosine = np.clip(np.sum(current_unit * reference_unit, axis=1), -1.0, 1.0)
                shape_each = np.arccos(cosine) / np.pi
            else:
                shape_each = np.linalg.norm(
                    hu - self.prototype["hu"][slot_indices], axis=1
                ) / 10.0
            facts["shape_error"] = float(np.mean(shape_each))
            appearance_each = np.linalg.norm(
                lab - self.prototype["lab"][slot_indices], axis=1
            ) / 100.0
            facts["appearance_error"] = float(np.mean(appearance_each))
            for shape_value, appearance_value, component in zip(
                shape_each, appearance_each, ordered
            ):
                maps["shape_error"][component["mask"]] = max(float(shape_value), 1e-6)
                maps["appearance_error"][component["mask"]] = max(
                    float(appearance_value), 1e-6
                )

        # When a relation is undefined because too few entities were matched,
        # it remains a relation-level uncertainty, not a duplicate count map.
        # The evolved AST can use the independent count atom to handle absence.
        maps = {
            name: _normalise_map(cv2.GaussianBlur(value, (0, 0), 1.4))
            for name, value in maps.items()
        }
        diagnostics = {
            "detected_count": int(count),
            "expected_count": self.expected_count,
            "assignment": assignment,
            "unmatched_slots": sorted(unmatched_slots),
            "unmatched_components": sorted(unmatched_components),
            "component_centroids": [item["centroid"].tolist() for item in components],
            "component_areas": [float(item["area"]) for item in components],
            "background_lab": background.tolist(),
        }
        return facts, maps, diagnostics

    def extract(self, image) -> Tuple[Dict[str, float], Dict[str, Any]]:
        facts, _, diagnostics = self.extract_spatial(image)
        return facts, diagnostics

    def counterfactual(self, image, predicate: str) -> np.ndarray:
        prepared = _rgb(image, self.image_size)
        components, _, background_lab = self._segment(prepared)
        if not components:
            return prepared
        component = components[0]
        mask = component["mask"]
        output = prepared.copy()
        background_rgb = cv2.cvtColor(
            np.uint8([[np.clip(background_lab, 0, 255)]]), cv2.COLOR_LAB2RGB
        )[0, 0]
        if predicate == "appearance_consistency":
            hsv = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)
            hsv[..., 0][mask] = (hsv[..., 0][mask].astype(int) + 60) % 180
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        if predicate in {"arrangement_eq", "ordered_spacing"}:
            shift = max(12, self.image_size // 8)
            pixels = output.copy()
            output[mask] = background_rgb
            shifted_mask = np.roll(mask, shift, axis=1)
            shifted_pixels = np.roll(pixels, shift, axis=1)
            output[shifted_mask] = shifted_pixels[shifted_mask]
            return output
        if predicate in {"similar_orientation", "similar_size"}:
            x, y, w, h = component["bbox"]
            patch, patch_mask = output[y:y+h, x:x+w].copy(), mask[y:y+h, x:x+w]
            output[mask] = background_rgb
            if predicate == "similar_orientation":
                patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
                patch_mask = cv2.rotate(patch_mask.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE).astype(bool)
            else:
                scale = 0.62
                new_w, new_h = max(2, int(w * scale)), max(2, int(h * scale))
                patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
                patch_mask = cv2.resize(patch_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            ph, pw = patch.shape[:2]
            left = max(0, min(self.image_size - pw, x + (w - pw) // 2))
            top = max(0, min(self.image_size - ph, y + (h - ph) // 2))
            target = output[top:top+ph, left:left+pw]
            target[patch_mask] = patch[patch_mask]
            return output
        output[mask] = background_rgb
        return output

    def state_dict(self) -> Dict[str, Any]:
        return {
            "kind": "generic_component_v1",
            "expected_count": self.expected_count,
            "arrangement": self.arrangement,
            "image_size": self.image_size,
            "threshold": float(self.threshold),
            "min_area": float(self.min_area),
            "morph": int(self.morph),
            "support_match_rate": float(self.support_match_rate),
            "prototype": {key: value.tolist() for key, value in self.prototype.items()},
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        if state.get("kind") != "generic_component_v1":
            raise ValueError(f"Unsupported generic grounder: {state.get('kind')}")
        self.expected_count = int(state["expected_count"])
        self.arrangement = str(state["arrangement"])
        self.image_size = int(state["image_size"])
        self.threshold = float(state["threshold"])
        self.min_area = float(state["min_area"])
        self.morph = int(state["morph"])
        self.support_match_rate = float(state.get("support_match_rate", 0.0))
        self.prototype = {key: np.asarray(value, dtype=np.float32)
                          for key, value in state.get("prototype", {}).items()}


def compile_generic_rules(
    proposal: Mapping[str, Any], support_paths: Sequence[Path], image_size: int = 256,
    minimum_cv_auc: float = 0.75, maximum_normal_fpr: float = 0.25,
) -> Tuple[Optional[GenericComponentGrounder], List[Dict[str, Any]], List[Dict[str, Any]]]:
    entity = proposal.get("primary_entity")
    if not isinstance(entity, Mapping):
        return None, [], []
    expected = int(entity["expected_count"])
    arrangement = str(entity["arrangement"])
    audits = []
    fold_payloads = []
    for held in range(len(support_paths)):
        train = [path for index, path in enumerate(support_paths) if index != held]
        grounder = GenericComponentGrounder(expected, arrangement, image_size).fit(train)
        train_facts = [grounder.extract(path)[0] for path in train]
        held_facts = grounder.extract(support_paths[held])[0]
        fold_payloads.append((grounder, train_facts, held_facts, support_paths[held]))
    accepted = []
    for candidate in proposal.get("rules", []):
        predicate = str(candidate.get("predicate"))
        fact = FACT_BY_PREDICATE.get(predicate)
        audit = {"proposal_id": candidate.get("id"), "predicate": predicate, "fact": fact}
        if fact is None:
            audit.update({"accepted": False, "reason": "unsupported_predicate"})
            audits.append(audit)
            continue
        normal_scores, pseudo_scores = [], []
        for grounder, train_facts, held_facts, held_path in fold_payloads:
            train_values = np.asarray([row[fact] for row in train_facts], dtype=float)
            upper = float(np.max(train_values) + max(np.std(train_values), 0.01))
            scale = max(float(np.std(train_values)), upper * 0.25, 0.01)
            normal_scores.append(max(0.0, (held_facts[fact] - upper) / scale))
            pseudo_image = grounder.counterfactual(held_path, predicate)
            pseudo_value = grounder.extract(pseudo_image)[0][fact]
            pseudo_scores.append(max(0.0, (pseudo_value - upper) / scale))
        labels = np.asarray([0] * len(normal_scores) + [1] * len(pseudo_scores))
        scores = np.asarray(normal_scores + pseudo_scores)
        cv_auc = float(roc_auc_score(labels, scores)) if np.ptp(scores) > 0 else 0.5
        normal_fpr = float(np.mean(np.asarray(normal_scores) > 0))
        pseudo_trigger = float(np.mean(np.asarray(pseudo_scores) > 0))
        passed = bool(
            cv_auc >= minimum_cv_auc
            and normal_fpr <= maximum_normal_fpr
            and pseudo_trigger >= 0.5
        )
        audit.update({
            "accepted": passed,
            "reason": "passed_crossfit" if passed else "failed_crossfit",
            "cv_auc": cv_auc,
            "normal_fpr": normal_fpr,
            "pseudo_trigger_rate": pseudo_trigger,
            "normal_scores": normal_scores,
            "pseudo_scores": pseudo_scores,
        })
        audits.append(audit)
        if passed:
            accepted.append({**dict(candidate), "fact": fact})
    final_grounder = GenericComponentGrounder(expected, arrangement, image_size).fit(support_paths)
    if final_grounder.support_match_rate < 0.75 or not final_grounder.prototype:
        for audit in audits:
            if audit.get("accepted"):
                audit.update({"accepted": False, "reason": "unstable_full_support_grounding"})
        return final_grounder, [], audits
    support_facts = [final_grounder.extract(path)[0] for path in support_paths]
    rules = []
    for candidate in accepted:
        values = np.asarray([row[candidate["fact"]] for row in support_facts], dtype=float)
        upper = float(np.max(values) + max(np.std(values), 0.01))
        scale = max(float(np.std(values)), upper * 0.25, 0.01)
        rules.append({
            **candidate,
            "upper": upper,
            "soft_scale": scale,
            "weight": float(candidate.get("importance", 0.8)) * float(candidate.get("confidence", 0.8)),
        })
    total = sum(item["weight"] for item in rules)
    if total > 0:
        for item in rules:
            item["weight"] /= total
    return final_grounder, rules, audits


class GenericRelationalRuleEngine:
    def __init__(self, library: Mapping[str, Any]):
        self.library = dict(library)
        state = self.library.get("grounder")
        self.grounder = None
        if isinstance(state, Mapping):
            self.grounder = GenericComponentGrounder(
                int(state["expected_count"]), str(state["arrangement"]),
                int(state["image_size"]), state=state,
            )
        self.rules = list(self.library.get("rules", []))

    def evaluate(self, image):
        if self.grounder is None or not self.rules:
            return 0.0, [], {}, {"abstained": True}
        facts, diagnostics = self.grounder.extract(image)
        evidence = []
        for rule in self.rules:
            excess = max(0.0, facts[rule["fact"]] - float(rule["upper"]))
            violation = float(math.tanh(excess / max(float(rule["soft_scale"]), 1e-8)))
            evidence.append({
                "rule_id": rule["id"], "predicate": rule["predicate"],
                "fact": rule["fact"], "value": facts[rule["fact"]],
                "violation": violation, "weight": float(rule["weight"]),
            })
        score = float(sum(row["weight"] * row["violation"] for row in evidence))
        evidence.sort(key=lambda row: row["weight"] * row["violation"], reverse=True)
        return score, evidence, facts, diagnostics

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GenericRelationalRuleEngine":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: Union[str, Path]) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output
