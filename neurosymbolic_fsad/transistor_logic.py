"""Executable normality atoms and spatial evidence for MVTec transistor.

This module is intentionally separate from the historical v5 weighted rule
engine.  It provides a deterministic visual grounder, support-only stability
compiler, and a product-fuzzy logic program in which every accepted rule is
executed.  The image geometry exactly matches INP-Former's
``Resize(448) -> CenterCrop(392) -> map 256`` preprocessing so symbolic maps,
neural maps, and ground-truth masks share one coordinate system.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .class_specific_rules import TransistorGrounder
from .fuzzy_logic import FuzzyLogicProgram, fuzzy_and, fuzzy_not, fuzzy_or


ImageLike = Union[str, Path, Image.Image, np.ndarray]


ATOM_SPECS: Dict[str, Dict[str, Any]] = {
    "lead_count": {
        "fact": "lead_count_error",
        "predicate": "part_count",
        "description": (
            "The LLM proposes three leads and the visual lead-count grounder "
            "must remain inside its normal-support tolerance."
        ),
        "llm_predicates": ("part_count",),
        "localizer": "lead_template_union_or_missing_slots",
    },
    "lead_geometry": {
        "fact": "lead_geometry_error",
        "predicate": "lead_topology",
        "description": (
            "The pose-normalized lead assembly must match normal support "
            "geometry proposed by the LLM relation family."
        ),
        "llm_predicates": (
            "attached_to", "relative_position", "similar_width",
            "similar_orientation", "continuity",
        ),
        "localizer": "pose_normalized_lead_template_residual",
    },
    "lead_symmetry": {
        "fact": "lead_symmetry_error",
        "predicate": "symmetric_spacing",
        "description": "The ordered three lead attachments have symmetric spacing.",
        "llm_predicates": ("ordered_spacing", "mirror_symmetry"),
        "localizer": "attachment_slots_and_intervals",
    },
    "package_pose": {
        "fact": "package_pose_error",
        "predicate": "relative_pose",
        "description": "The package occupies a stable pose relative to the board.",
        "llm_predicates": ("package_pose",),
        "localizer": "current_vs_normal_package_box",
    },
}


FINAL_FUSION_AST: Dict[str, Any] = {
    "op": "OR",
    "id": "final.neuro_symbolic_anomaly",
    "args": [
        {
            "op": "ATOM", "name": "neural_anomaly",
            "id": "final.atom.neural_anomaly",
        },
        {
            "op": "AND", "id": "final.grounded_symbolic_violation",
            "args": [
                {
                    "op": "ATOM", "name": "neural_grounding",
                    "id": "final.atom.neural_grounding",
                },
                {
                    "op": "ATOM", "name": "symbolic_anomaly",
                    "id": "final.atom.symbolic_anomaly",
                },
            ],
        },
    ],
}

_FINAL_SPECIAL_ATOMS = (
    "neural_anomaly", "neural_grounding", "symbolic_anomaly",
)
_FINAL_FUSION_PROGRAM = FuzzyLogicProgram(
    FINAL_FUSION_AST, (), special_atoms=_FINAL_SPECIAL_ATOMS
)
_SYMBOLIC_SPECIAL_ATOMS = ("grounding", "symbolic_abstain")


def preprocess_rgb(
    image: ImageLike,
    *,
    input_size: int = 448,
    crop_size: int = 392,
    map_size: int = 256,
) -> np.ndarray:
    """Return an RGB uint8 image in the INP-Former anomaly-map frame."""
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
    resized = cv2.resize(
        array.astype(np.uint8), (input_size, input_size), interpolation=cv2.INTER_AREA
    )
    start = (input_size - crop_size) // 2
    cropped = resized[start:start + crop_size, start:start + crop_size]
    return cv2.resize(cropped, (map_size, map_size), interpolation=cv2.INTER_AREA)


def preprocess_mask(
    image: ImageLike,
    *,
    input_size: int = 448,
    crop_size: int = 392,
    map_size: int = 256,
) -> np.ndarray:
    """Return a Boolean ground-truth mask in the same map frame."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            array = np.asarray(handle.convert("L"))
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("L"))
    else:
        array = np.asarray(image)
        if array.ndim == 3:
            array = np.max(array[..., :3], axis=-1)
    resized = cv2.resize(
        array.astype(np.uint8), (input_size, input_size), interpolation=cv2.INTER_NEAREST
    )
    start = (input_size - crop_size) // 2
    cropped = resized[start:start + crop_size, start:start + crop_size]
    return cv2.resize(
        cropped, (map_size, map_size), interpolation=cv2.INTER_NEAREST
    ) > 0


def _normalise_map(array: np.ndarray, quantile: float = 0.995) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    positive = values[values > 0]
    if not positive.size:
        return np.zeros_like(values, dtype=np.float32)
    scale = max(float(np.quantile(positive, quantile)), 1e-6)
    return np.clip(values / scale, 0.0, 1.0).astype(np.float32)


def _box_mask(size: int, box: Sequence[int], thickness: int = -1) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.float32)
    x0, y0, x1, y1 = (int(value) for value in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(size - 1, x1), min(size - 1, y1)
    if x1 > x0 and y1 > y0:
        cv2.rectangle(result, (x0, y0), (x1, y1), 1.0, thickness)
    return result


class SpatialTransistorGrounder(TransistorGrounder):
    """Transistor facts plus a spatial localization for every fact.

    The underlying scene program is inherited from the independently tested
    v5 transistor grounder.  Only its input coordinate frame and its outputs
    are extended here; the v5 implementation and historical results remain
    untouched.
    """

    kind = "mvtec_transistor_spatial_logic_v7"

    def __init__(self, state: Mapping[str, Any] | None = None, image_size: int = 256):
        super().__init__(state=None, image_size=image_size)
        self.input_size = 448
        self.crop_size = 392
        if state is not None:
            self.image_size = int(state.get("image_size", image_size))
            self.input_size = int(state.get("input_size", 448))
            self.crop_size = int(state.get("crop_size", 392))
            self.prototype = dict(state["prototype"])

    def _prepared(self, image: ImageLike) -> np.ndarray:
        if isinstance(image, np.ndarray) and image.shape[:2] == (
            self.image_size, self.image_size
        ):
            return image.astype(np.uint8)
        return preprocess_rgb(
            image,
            input_size=self.input_size,
            crop_size=self.crop_size,
            map_size=self.image_size,
        )

    def fit(self, supports: Sequence[ImageLike]):
        prepared = [self._prepared(image) for image in supports]
        super().fit(prepared)
        poses = []
        attachment_rows = []
        for image in prepared:
            _, _, attachments, _, pose = self._lead_scene(image)
            poses.append(pose.tolist())
            attachment_rows.append([dict(row) for row in attachments])
        self.prototype["raw_poses"] = poses
        self.prototype["attachments"] = attachment_rows
        return self

    def _scene_with_maps(self, image: ImageLike):
        prepared = self._prepared(image)
        array, body_box, attachments, template, pose = self._lead_scene(prepared)
        x0, _, x1, y1 = (int(value) for value in body_box)
        lower = min(self.image_size, y1 + int(0.42 * self.image_size))
        crop_box = (x0, max(0, y1 - 2), x1, lower)
        return prepared, array, body_box, attachments, template, pose, crop_box

    def _template_residual(self, template: np.ndarray) -> Tuple[float, np.ndarray]:
        references = np.asarray(self.prototype.get("templates", []), dtype=np.float32)
        if not references.size:
            return 1.0, np.ones_like(template, dtype=np.float32)
        references = references.reshape(len(references), 64, 64)
        distances = np.mean(np.abs(references - template[None]), axis=(1, 2))
        index = int(np.argmin(distances))
        return float(distances[index]), np.abs(template - references[index])

    def extract_spatial(self, image: ImageLike):
        (
            prepared, _, body_box, attachments, template, pose, crop_box,
        ) = self._scene_with_maps(image)
        geometry_error, residual64 = self._template_residual(template)
        from .class_specific_rules import _prototype_distance  # isolated reuse

        pose_error, _ = _prototype_distance([pose], self.prototype["pose"])
        positions = [float(row["x"]) for row in attachments]
        symmetry_error = 1.0
        if len(positions) == 3:
            symmetry_error = abs(
                (positions[1] - positions[0]) - (positions[2] - positions[1])
            )
        facts = {
            "lead_count_error": abs(len(attachments) - 3) / 3.0,
            "lead_geometry_error": float(geometry_error),
            "lead_symmetry_error": float(symmetry_error),
            "package_pose_error": float(pose_error),
        }

        size = self.image_size
        maps: Dict[str, np.ndarray] = {}
        x0, cy0, x1, cy1 = crop_box
        residual = np.zeros((size, size), dtype=np.float32)
        if x1 > x0 and cy1 > cy0:
            residual[cy0:cy1, x0:x1] = cv2.resize(
                residual64.astype(np.float32),
                (x1 - x0, cy1 - cy0), interpolation=cv2.INTER_LINEAR,
            )
        maps["lead_geometry_error"] = _normalise_map(
            cv2.GaussianBlur(residual, (0, 0), 1.2)
        )

        reference_templates = np.asarray(
            self.prototype.get("templates", []), dtype=np.float32
        )
        template_union = np.asarray(template, dtype=np.float32)
        if reference_templates.size:
            reference_templates = reference_templates.reshape(-1, 64, 64)
            template_union = np.maximum(
                template_union, np.median(reference_templates, axis=0)
            )
        count_map = np.zeros((size, size), dtype=np.float32)
        if x1 > x0 and cy1 > cy0:
            count_map[cy0:cy1, x0:x1] = cv2.resize(
                template_union.astype(np.float32),
                (x1 - x0, cy1 - cy0), interpolation=cv2.INTER_LINEAR,
            )
        maps["lead_count_error"] = _normalise_map(
            cv2.GaussianBlur(count_map, (0, 0), 2.0)
        )

        symmetry_map = np.zeros((size, size), dtype=np.float32)
        y_start = max(0, int(body_box[3]) - 3)
        y_stop = min(size - 1, y_start + int(0.36 * size))
        for row in attachments:
            center_x = int(round(float(row["x"]) * size))
            width = max(4, int(round(float(row["width"]) * size)))
            cv2.rectangle(
                symmetry_map,
                (max(0, center_x - width), y_start),
                (min(size - 1, center_x + width), y_stop),
                1.0, -1,
            )
        if not attachments:
            symmetry_map = count_map.copy()
        maps["lead_symmetry_error"] = _normalise_map(
            cv2.GaussianBlur(symmetry_map, (0, 0), 2.0)
        )

        current_box = _box_mask(size, body_box, thickness=4)
        normal_poses = np.asarray(self.prototype.get("raw_poses", []), dtype=float)
        if normal_poses.size:
            median_box = np.rint(
                np.median(normal_poses, axis=0) * size
            ).astype(int).tolist()
            expected_box = _box_mask(size, median_box, thickness=4)
            pose_map = np.maximum(current_box, expected_box)
            cv2.line(
                pose_map,
                tuple(np.rint(np.asarray(body_box)[[0, 1]]).astype(int)),
                tuple(np.rint(np.asarray(median_box)[[0, 1]]).astype(int)),
                1.0, 2,
            )
        else:
            pose_map = current_box
        maps["package_pose_error"] = _normalise_map(
            cv2.GaussianBlur(pose_map, (0, 0), 2.0)
        )
        diagnostics = {
            "detected_lead_count": len(attachments),
            "lead_attachments": attachments,
            "body_box": [int(value) for value in body_box],
            "lead_crop_box": [int(value) for value in crop_box],
            "template_distance": float(geometry_error),
            "package_pose_distance": float(pose_error),
            "coordinate_frame": (
                f"resize{self.input_size}_centercrop{self.crop_size}_map{size}"
            ),
        }
        return facts, maps, diagnostics, prepared

    def state_dict(self):
        return {
            "kind": self.kind,
            "image_size": self.image_size,
            "input_size": self.input_size,
            "crop_size": self.crop_size,
            "prototype": self.prototype,
        }


def _llm_vote_rates(candidate_bank: Mapping[str, Any] | None) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not candidate_bank:
        return result
    candidates = candidate_bank.get("candidates", candidate_bank.get("rules", []))
    if not isinstance(candidates, list):
        return result
    denominator = max(
        int(candidate_bank.get("expected_support_count", 0)),
        int(candidate_bank.get("successful_support_count", 0)),
        1,
    )
    for row in candidates:
        if not isinstance(row, Mapping):
            continue
        semantics = row.get("rule", row.get("semantics", row))
        predicate = str(semantics.get("predicate", row.get("predicate", "")))
        vote = row.get(
            "occurrence_rate_among_expected_supports",
            row.get("support_rate", row.get("vote_rate")),
        )
        if vote is None:
            vote = float(
                row.get("occurrence_count", row.get("support_count", row.get("count", 0)))
            ) / denominator
        try:
            result[predicate] = max(result.get(predicate, 0.0), float(vote))
        except (TypeError, ValueError):
            continue
    return result


def _atom_llm_vote(
    atom_id: str,
    rates: Mapping[str, float],
    candidate_bank: Mapping[str, Any] | None,
) -> float:
    """Return support coverage of a mapped LLM predicate family.

    Rotation-dependent descriptions such as horizontal versus vertical spacing
    remain separate canonical candidates, while proposal support is their
    sample-ID union.  This vote only admits a family for visual verification;
    it is never used as a rule weight.
    """
    predicates = frozenset(ATOM_SPECS[atom_id]["llm_predicates"])
    if candidate_bank:
        sample_ids = set()
        saw_provenance = False
        for candidate in candidate_bank.get("candidates", []):
            if not isinstance(candidate, Mapping):
                continue
            rule = candidate.get("rule", {})
            if str(rule.get("predicate", "")) not in predicates:
                continue
            ids = candidate.get("sample_ids")
            if isinstance(ids, list):
                saw_provenance = True
                sample_ids.update(str(value) for value in ids)
        if saw_provenance:
            denominator = max(
                int(candidate_bank.get("expected_support_count", 0)), 1
            )
            return float(len(sample_ids) / denominator)
    return max((float(rates.get(name, 0.0)) for name in predicates), default=0.0)


def _llm_candidate_audit(candidate_bank: Mapping[str, Any] | None) -> list:
    if not candidate_bank:
        return []
    predicate_to_atoms: Dict[str, list] = {}
    for atom_id, specification in ATOM_SPECS.items():
        for predicate in specification["llm_predicates"]:
            predicate_to_atoms.setdefault(predicate, []).append(atom_id)
    rows = []
    candidates = candidate_bank.get("candidates", [])
    if not isinstance(candidates, list):
        return rows
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        rule = candidate.get("rule", {})
        predicate = str(rule.get("predicate", ""))
        mapped = predicate_to_atoms.get(predicate, [])
        rows.append({
            "canonical_key": candidate.get("canonical_key"),
            "rule": rule,
            "occurrence_count": candidate.get("occurrence_count", 0),
            "occurrence_rate_among_expected_supports": candidate.get(
                "occurrence_rate_among_expected_supports", 0.0
            ),
            "mapped_grounder_atoms": mapped,
            "status": "groundable_candidate" if mapped else "rejected_not_groundable",
            "reason": (
                "mapped into a deterministic executable visual atom"
                if mapped else
                "the current transistor grounder has no validated executable atom"
            ),
        })
    return rows


def _calibrate_rule(values: Sequence[float], support_count: int) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("Cross-fit fact values must be finite and non-empty")
    # Keep one calibration policy for nested k4/k8/k16/k30 controls.  Support
    # size changes observations, not the thresholding algorithm.
    q25, q75, q95 = (
        float(value) for value in np.quantile(array, [0.25, 0.75, 0.95])
    )
    upper = q95 + 0.25 * max(q75 - q25, 0.0)
    scale = max(float(np.std(array)), float(np.quantile(array, 0.75) - np.quantile(array, 0.25)), upper * 0.15, 0.02)
    violations = np.tanh(np.maximum(0.0, array - upper) / scale)
    return {
        "upper": float(upper),
        "scale": float(scale),
        "crossfit_normal_fpr": float(np.mean(violations > 1e-8)),
        "crossfit_max_violation": float(np.max(violations)),
        "crossfit_mean": float(np.mean(array)),
        "crossfit_std": float(np.std(array)),
        "crossfit_q95": float(np.quantile(array, 0.95)),
    }


def _bootstrap_threshold_stability(
    values: Sequence[float], support_count: int, *, repeats: int = 200, seed: int = 668,
) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    thresholds = []
    for _ in range(repeats):
        sample = rng.choice(array, size=len(array), replace=True)
        thresholds.append(_calibrate_rule(sample, support_count)["upper"])
    thresholds = np.asarray(thresholds, dtype=float)
    median = float(np.median(thresholds))
    q10, q90 = (float(value) for value in np.quantile(thresholds, [0.10, 0.90]))
    return {
        "repeats": int(repeats),
        "threshold_median": median,
        "threshold_q10": q10,
        "threshold_q90": q90,
        "relative_interval_width": float((q90 - q10) / max(abs(median), 0.02)),
    }


def _logic_ast(active: Sequence[str]) -> Dict[str, Any]:
    active = list(active)
    if not active:
        return {
            "op": "ATOM",
            "name": "symbolic_abstain",
            "id": "symbolic.abstain_zero",
        }
    count = "lead_count" if "lead_count" in active else None
    dependent = [name for name in active if name != count]
    children = []
    if count is not None:
        children.append({"op": "ATOM", "name": count, "id": "atom.lead_count"})
    if dependent:
        dependent_node: Dict[str, Any]
        if len(dependent) == 1:
            dependent_node = {
                "op": "ATOM", "name": dependent[0], "id": f"atom.{dependent[0]}"
            }
        else:
            dependent_node = {
                "op": "AND", "id": "relations.all", "args": [
                    {"op": "ATOM", "name": name, "id": f"atom.{name}"}
                    for name in dependent
                ],
            }
        if count is not None:
            children.append({
                "op": "IMPLIES", "id": "relations.when_grounded",
                "args": [
                    {"op": "ATOM", "name": "grounding", "id": "atom.grounding"},
                    dependent_node,
                ],
            })
        else:
            children.append(dependent_node)
    if len(children) == 1:
        normal = children[0]
    else:
        normal = {"op": "AND", "id": "normal.all_rules", "args": children}
    return {"op": "NOT", "id": "symbolic.anomaly", "arg": normal}


def compile_transistor_logic_library(
    support_paths: Sequence[Union[str, Path]],
    *,
    candidate_bank: Mapping[str, Any] | None = None,
    dataset: str = "mvtec",
    class_name: str = "transistor",
) -> Dict[str, Any]:
    """Compile stable static rules using normal supports only."""
    paths = [Path(path).expanduser().resolve() for path in support_paths]
    if len(paths) < 4:
        raise ValueError("At least four normal supports are required")
    if candidate_bank:
        expected = int(candidate_bank.get("expected_support_count", len(paths)))
        if expected != len(paths):
            raise ValueError(
                "Qwen candidate bank/support mismatch: "
                f"bank={expected}, compiler={len(paths)}"
            )
        bank_class = str(candidate_bank.get("class_name", class_name))
        if bank_class != class_name:
            raise ValueError(
                f"Qwen candidate bank class mismatch: {bank_class!r} != {class_name!r}"
            )
        bank_support = candidate_bank.get("support")
        if isinstance(bank_support, list):
            bank_paths = [
                Path(str(row["image_path"])).expanduser().resolve()
                for row in sorted(
                    bank_support, key=lambda row: int(row["sample_index"])
                )
            ]
            if bank_paths != paths:
                raise ValueError(
                    "Qwen candidate bank paths do not exactly match compiler supports"
                )
    prepared = [preprocess_rgb(path) for path in paths]
    normal_values: Dict[str, list] = {
        specification["fact"]: [] for specification in ATOM_SPECS.values()
    }
    fold_rows = []
    for held in range(len(prepared)):
        train = [image for index, image in enumerate(prepared) if index != held]
        grounder = SpatialTransistorGrounder().fit(train)
        facts, _, diagnostics, _ = grounder.extract_spatial(prepared[held])
        for fact in normal_values:
            normal_values[fact].append(float(facts[fact]))
        fold_rows.append({
            "held_support": str(paths[held]),
            "train_support_count": len(train),
            "facts": facts,
            "diagnostics": diagnostics,
        })

    rates = _llm_vote_rates(candidate_bank)
    support_count = len(paths)
    required_vote = 0.70
    maximum_fpr = 0.10
    maximum_bootstrap_relative_width = 0.50
    audits = []
    provisional = []
    for atom_id, specification in ATOM_SPECS.items():
        fact = specification["fact"]
        values = normal_values[fact]
        calibration = _calibrate_rule(values, support_count)
        bootstrap = _bootstrap_threshold_stability(values, support_count)
        vote = _atom_llm_vote(atom_id, rates, candidate_bank)
        # A missing bank is an explicit fallback for deterministic unit/smoke
        # tests; production runs record the absence and do not fabricate votes.
        llm_ok = vote >= required_vote if candidate_bank else True
        dynamic_headroom = not (
            atom_id == "lead_symmetry" and calibration["upper"] >= 0.98
        )
        bootstrap_stable = (
            bootstrap["relative_interval_width"]
            <= maximum_bootstrap_relative_width
        )
        stable = (
            llm_ok
            and calibration["crossfit_normal_fpr"] <= maximum_fpr
            and dynamic_headroom
            and bootstrap_stable
        )
        reasons = []
        if not llm_ok:
            reasons.append("insufficient_llm_support_vote")
        if calibration["crossfit_normal_fpr"] > maximum_fpr:
            reasons.append("crossfit_normal_fpr_too_high")
        if not dynamic_headroom:
            reasons.append("threshold_exhausts_predicate_range")
        if not bootstrap_stable:
            reasons.append("bootstrap_threshold_interval_too_wide")
        row = {
            "id": atom_id,
            "fact": fact,
            "predicate": specification["predicate"],
            "description": specification["description"],
            "localizer": specification["localizer"],
            "llm_source_predicates": list(specification["llm_predicates"]),
            "llm_vote_rate": vote,
            "required_llm_vote_rate": required_vote,
            "leave_one_out_values": [float(value) for value in values],
            **calibration,
            "bootstrap": bootstrap,
            "status": "accepted_provisional" if stable and support_count < 10 else (
                "accepted_stable" if stable else "rejected"
            ),
            "rejection_reasons": reasons,
        }
        audits.append(row)
        if stable:
            provisional.append(row)

    # Remove redundant atoms using only normal-support cross-fit vectors.  A
    # deterministic priority keeps semantically specific atoms before pose.
    priority = {"lead_count": 0, "lead_geometry": 1, "lead_symmetry": 2, "package_pose": 3}
    accepted = []
    for row in sorted(provisional, key=lambda item: priority[item["id"]]):
        vector = np.asarray(row["leave_one_out_values"], dtype=float)
        redundant_with = None
        for kept in accepted:
            other = np.asarray(kept["leave_one_out_values"], dtype=float)
            if np.std(vector) <= 1e-12 or np.std(other) <= 1e-12:
                correlation = 1.0 if np.allclose(vector, other) else 0.0
            else:
                correlation = float(np.corrcoef(vector, other)[0, 1])
            if np.isfinite(correlation) and abs(correlation) > 0.95:
                redundant_with = kept["id"]
                break
        if redundant_with is None:
            accepted.append(row)
        else:
            row["status"] = "rejected_redundant"
            row["rejection_reasons"].append(f"redundant_with:{redundant_with}")

    active_ids = [row["id"] for row in accepted]
    ast = _logic_ast(active_ids)
    program = FuzzyLogicProgram(
        ast, active_ids, special_atoms=_SYMBOLIC_SPECIAL_ATOMS
    )
    grounder = SpatialTransistorGrounder().fit(prepared)
    return {
        "schema_version": "7.0-executable-product-fuzzy-logic",
        "dataset": dataset,
        "class_name": class_name,
        "rule_support_count": support_count,
        "support": [str(path) for path in paths],
        "candidate_rule_count": len(candidate_bank.get("candidates", [])) if candidate_bank else len(ATOM_SPECS),
        "groundable_rule_count": len(ATOM_SPECS),
        "accepted_rule_count": len(accepted),
        "accepted_rule_ids": active_ids,
        "rules": accepted,
        "candidate_audit": audits,
        "selection": {
            "uses_test_images_or_labels": False,
            "policy": "fixed-v7.1-nested-support-policy",
            "required_llm_vote_rate": required_vote,
            "maximum_crossfit_normal_fpr": maximum_fpr,
            "maximum_bootstrap_relative_interval_width": (
                maximum_bootstrap_relative_width
            ),
            "cross_validation": "leave_one-normal-support-out",
            "threshold_bootstrap_repeats": 200,
            "redundancy_cutoff_abs_pearson": 0.95,
        },
        "llm": {
            "model": "/home/hjh/AgentModels/Qwen3-VL-8B-Thinking",
            "predicate_vote_rates": rates,
            "candidate_bank_supplied": candidate_bank is not None,
            "role": "candidate proposal only; never a runtime weight",
            "vote_semantics": (
                "union of per-support predicate-family mentions; proposal "
                "coverage is not treated as rule satisfaction"
            ),
            "candidate_audit": _llm_candidate_audit(candidate_bank),
        },
        "grounder": grounder.state_dict(),
        "logic": {
            "semantics": "stable_product_logic",
            "operators": {
                "NOT": "1-a",
                "AND": "product(args)",
                "OR": "1-product(1-args)",
                "IMPLIES": "1-a*(1-b)",
            },
            "symbolic_anomaly_ast": ast,
            "formula": (
                (
                    "NOT(AND(all accepted normality atoms))"
                    if active_ids else "SYMBOLIC_ABSTAIN(0)"
                )
                + "; final = "
                "OR(INPFormerAnomaly, AND(NeuralGroundingConfidence, "
                "SymbolicAnomaly))"
            ),
            "runtime_validation": {
                "active_atom_counts": program.validation.active_atom_counts,
                "execution_atoms": list(program.validation.execution_atoms),
                "node_count": program.validation.node_count,
            },
        },
        "crossfit": fold_rows,
        "runtime_requires_llm": False,
    }


@dataclass
class LogicInference:
    symbolic_anomaly: float
    symbolic_map: np.ndarray
    facts: Dict[str, float]
    atoms: Dict[str, Dict[str, Any]]
    scalar_trace: list
    diagnostics: Dict[str, Any]
    explanation: str
    rule_maps: Dict[str, np.ndarray]
    # Spatial support of each executable predicate before it is multiplied by
    # the predicate's scalar violation.  Keeping this separate is essential:
    # a satisfied rule has an all-one truth map, but its geometric scope is
    # still needed to ask whether INP-Former sees evidence in the same place.
    localization_maps: Dict[str, np.ndarray]


class TransistorLogicEngine:
    """Execute every accepted atom and the frozen symbolic AST."""

    def __init__(self, library: Mapping[str, Any]):
        self.library = dict(library)
        self.rules = list(self.library["rules"])
        self.active_ids = [str(row["id"]) for row in self.rules]
        self.rule_by_id = {str(row["id"]): row for row in self.rules}
        self.grounder = SpatialTransistorGrounder(state=self.library["grounder"])
        self.program = FuzzyLogicProgram(
            self.library["logic"]["symbolic_anomaly_ast"], self.active_ids,
            special_atoms=_SYMBOLIC_SPECIAL_ATOMS,
        )

    @staticmethod
    def _violation(value: float, upper: float, scale: float) -> float:
        excess = max(0.0, float(value) - float(upper))
        return float(math.tanh(excess / max(float(scale), 1e-8)))

    def evaluate(self, image: ImageLike) -> LogicInference:
        facts, fact_maps, diagnostics, _ = self.grounder.extract_spatial(image)
        atoms: Dict[str, Dict[str, Any]] = {}
        scalar_values: Dict[str, Any] = {}
        map_values: Dict[str, Any] = {}
        rule_maps: Dict[str, np.ndarray] = {}
        localization_maps: Dict[str, np.ndarray] = {}
        for rule in self.rules:
            atom_id = str(rule["id"])
            fact = str(rule["fact"])
            value = float(facts[fact])
            violation = self._violation(value, rule["upper"], rule["scale"])
            truth = 1.0 - violation
            localization = np.asarray(fact_maps[fact], dtype=np.float32)
            violation_map = np.clip(violation * localization, 0.0, 1.0)
            truth_map = 1.0 - violation_map
            atoms[atom_id] = {
                "rule_id": atom_id,
                "predicate": rule["predicate"],
                "fact": fact,
                "fact_value": value,
                "upper": float(rule["upper"]),
                "scale": float(rule["scale"]),
                "truth": truth,
                "violation": violation,
                "defined": True,
            }
            scalar_values[atom_id] = truth
            map_values[atom_id] = truth_map
            rule_maps[atom_id] = truth_map
            localization_maps[atom_id] = np.clip(localization, 0.0, 1.0)
        # Relations are meaningful when the lead count atom is reasonably
        # grounded.  This truth participates through IMPLIES rather than by
        # multiplying a hand-written score.
        count_truth = float(scalar_values.get("lead_count", 1.0))
        scalar_values["grounding"] = count_truth
        map_values["grounding"] = np.full(
            (self.grounder.image_size, self.grounder.image_size),
            count_truth, dtype=np.float32,
        )
        scalar_values["symbolic_abstain"] = 0.0
        map_values["symbolic_abstain"] = np.zeros(
            (self.grounder.image_size, self.grounder.image_size),
            dtype=np.float32,
        )
        symbolic, trace = self.program.evaluate(scalar_values)
        symbolic_map, _ = self.program.evaluate(map_values)
        symbolic_map = np.asarray(symbolic_map, dtype=np.float32)
        for atom_id in self.active_ids:
            counterfactual = dict(scalar_values)
            counterfactual[atom_id] = 1.0
            without_atom, _ = self.program.evaluate(counterfactual)
            atoms[atom_id]["logical_counterfactual_contribution"] = float(
                max(0.0, float(symbolic) - float(without_atom))
            )
        violated = sorted(
            (row for row in atoms.values() if row["violation"] > 1e-8),
            key=lambda row: row["violation"], reverse=True,
        )
        if violated:
            clauses = [
                f"{row['rule_id']} violated: {row['fact']}={row['fact_value']:.4f} "
                f"> normal upper {row['upper']:.4f}"
                for row in violated
            ]
            explanation = "; ".join(clauses)
        elif self.active_ids:
            explanation = "All accepted static normality rules are satisfied."
        else:
            explanation = (
                "No candidate survived normal-only stability checks; "
                "the symbolic branch abstains."
            )
        return LogicInference(
            symbolic_anomaly=float(symbolic),
            symbolic_map=np.clip(symbolic_map, 0.0, 1.0),
            facts={key: float(value) for key, value in facts.items()},
            atoms=atoms,
            scalar_trace=trace,
            diagnostics=diagnostics,
            explanation=explanation,
            rule_maps=rule_maps,
            localization_maps=localization_maps,
        )

    @classmethod
    def load(cls, path: Union[str, Path]):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def neural_symbolic_or(neural: Any, symbolic: Any):
    """Explicit fuzzy disjunction used for both image scores and pixel maps."""
    return fuzzy_or(neural, symbolic)


def grounded_neural_symbolic(neural: Any, symbolic: Any):
    """Execute ``OR(neural, AND(neural_grounding, symbolic))``.

    The second occurrence has a different predicate meaning: it is perceptual
    grounding confidence for a symbolic violation.  At pixels where the
    detector supplies no visual evidence, a brittle symbolic localizer cannot
    create an unsupported hot region.  No coefficient is learned or searched.
    """
    value, _ = execute_grounded_neural_symbolic(neural, symbolic)
    return value


def execute_grounded_neural_symbolic(neural: Any, symbolic: Any):
    """Execute the final fusion AST and return its complete node trace."""
    return _FINAL_FUSION_PROGRAM.evaluate({
        "neural_anomaly": neural,
        "neural_grounding": neural,
        "symbolic_anomaly": symbolic,
    })
