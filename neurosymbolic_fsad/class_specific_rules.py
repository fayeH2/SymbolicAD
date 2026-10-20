"""Category-specific static visual grounders for relational FSAD experiments.

The classes in this module deliberately do not share a foreground detector.
They share only serialization and rule-scoring mechanics.  Each grounder is a
small executable scene program induced/calibrated from four normal supports.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks


ImageLike = Union[str, Path, Image.Image, np.ndarray]


def _rgb(image: ImageLike, size: int = 256) -> np.ndarray:
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


def _components(mask: np.ndarray, minimum: int, maximum: int | None = None):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    rows = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum or (maximum is not None and area > maximum):
            continue
        rows.append({
            "label": label,
            "mask": labels == label,
            "area": area,
            "bbox": [int(value) for value in stats[label, :4]],
            "centroid": np.asarray(centroids[label], dtype=np.float32),
        })
    return sorted(rows, key=lambda row: row["area"], reverse=True)


def _shape_descriptor(mask: np.ndarray) -> np.ndarray:
    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if not contours:
        return np.zeros(13, dtype=np.float32)
    contour = max(contours, key=cv2.contourArea)
    area = max(float(cv2.contourArea(contour)), 1.0)
    perimeter = max(float(cv2.arcLength(contour, True)), 1.0)
    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    x, y, width, height = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).reshape(-1)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    hu = np.clip(hu, -12.0, 12.0)
    covariance_ratio = 0.0
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) >= 3:
        values = np.linalg.eigvalsh(np.cov(points.T))
        covariance_ratio = float(values[0] / max(values[-1], 1e-6))
    return np.asarray([
        *hu.tolist(),
        area / hull_area,
        area / max(float(width * height), 1.0),
        4.0 * math.pi * area / (perimeter * perimeter),
        perimeter / math.sqrt(area),
        covariance_ratio,
        math.log(area + 1.0),
    ], dtype=np.float32)


def _prototype_state(rows: Sequence[np.ndarray]) -> Dict[str, Any]:
    matrix = np.asarray(rows, dtype=np.float32)
    if not len(matrix):
        return {"rows": [], "center": [], "scale": []}
    center = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - center), axis=0)
    scale = np.maximum(1.4826 * mad, np.std(matrix, axis=0) * 0.25)
    scale = np.maximum(scale, 0.03)
    return {"rows": matrix.tolist(), "center": center.tolist(), "scale": scale.tolist()}


def _prototype_distance(features: Sequence[np.ndarray], state: Mapping[str, Any]) -> Tuple[float, list]:
    reference = np.asarray(state.get("rows", []), dtype=np.float32)
    if not len(features) or reference.size == 0:
        return 50.0, []
    center = np.asarray(state["center"], dtype=np.float32)
    scale = np.asarray(state["scale"], dtype=np.float32)
    ref = (reference - center) / scale
    values = []
    for feature in features:
        row = (np.asarray(feature, dtype=np.float32) - center) / scale
        values.append(float(np.min(np.linalg.norm(ref - row[None], axis=1)) / math.sqrt(len(row))))
    return float(max(values)), values


def _template_distance(template: np.ndarray, references: Sequence[Sequence[float]]) -> float:
    """Return the closest mean absolute template distance.

    The templates are deliberately stored in the JSON rule library.  This
    makes the detector fully static at test time and keeps the four normal
    support images out of the runtime dependency chain.
    """
    rows = np.asarray(references, dtype=np.float32)
    if rows.size == 0:
        return 1.0
    probe = np.asarray(template, dtype=np.float32).reshape(1, -1)
    rows = rows.reshape(len(rows), -1)
    return float(np.min(np.mean(np.abs(rows - probe), axis=1)))


class Macaroni2Grounder:
    kind = "visa_macaroni2_shape_v1"

    def __init__(self, state: Mapping[str, Any] | None = None, image_size: int = 256):
        self.image_size = image_size
        self.prototype: Dict[str, Any] = {}
        if state is not None:
            self.image_size = int(state["image_size"])
            self.prototype = dict(state["prototype"])

    def _instances(self, image: ImageLike):
        array = _rgb(image, self.image_size)
        hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
        # Yellow pasta has a hue far below the green acquisition mat.
        mask = ((hsv[..., 0] >= 8) & (hsv[..., 0] <= 45) &
                (hsv[..., 1] >= 45) & (hsv[..., 2] >= 75)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        rows = _components(mask, minimum=int(self.image_size * self.image_size * 0.006),
                           maximum=int(self.image_size * self.image_size * 0.22))
        return array, rows[:8]

    def fit(self, supports: Sequence[ImageLike]):
        descriptors = []
        for image in supports:
            _, rows = self._instances(image)
            descriptors.extend(_shape_descriptor(row["mask"]) for row in rows[:4])
        self.prototype = _prototype_state(descriptors)
        return self

    def extract(self, image: ImageLike):
        _, rows = self._instances(image)
        selected = rows[:4]
        descriptors = [_shape_descriptor(row["mask"]) for row in selected]
        distance, per_instance = _prototype_distance(descriptors, self.prototype)
        spread = 0.0
        if len(descriptors) >= 2:
            center = np.asarray(self.prototype.get("center", []), dtype=np.float32)
            scale = np.asarray(self.prototype.get("scale", []), dtype=np.float32)
            normalized = [(row - center) / scale for row in descriptors]
            spread = max(float(np.linalg.norm(a - b) / math.sqrt(len(a)))
                         for i, a in enumerate(normalized) for b in normalized[i + 1:])
        facts = {
            "piece_count_error": abs(len(rows) - 4) / 4.0,
            "piece_shape_error": distance,
            "within_scene_shape_inconsistency": spread,
        }
        return facts, {
            "detected_piece_count": len(rows),
            "areas": [row["area"] / float(self.image_size ** 2) for row in rows],
            "per_instance_shape_distance": per_instance,
        }

    def state_dict(self):
        return {"kind": self.kind, "image_size": self.image_size, "prototype": self.prototype}


class PCB1Grounder:
    """Ground the four-pin row proposed consistently by Qwen on VisA pcb1."""

    kind = "visa_pcb1_four_pin_row_v1"

    def __init__(self, state: Mapping[str, Any] | None = None, image_size: int = 512):
        self.image_size = image_size
        self.prototype: Dict[str, Any] = {}
        if state is not None:
            self.image_size = int(state["image_size"])
            self.prototype = dict(state["prototype"])

    def _pin_scene(self, image: ImageLike):
        array = _rgb(image, self.image_size)
        hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
        blue = (
            (hsv[..., 0] > 75) & (hsv[..., 0] < 115) & (hsv[..., 1] > 70)
        ).astype(np.uint8)
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        candidates = _components(
            blue, minimum=int(0.025 * self.image_size ** 2),
            maximum=int(0.55 * self.image_size ** 2),
        )
        if not candidates:
            return [], np.zeros(128, dtype=np.float32), [0, 0, 0, 0], "missing_board", []
        board = candidates[0]
        x, y, width, height = board["bbox"]
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        margin_near = int(0.04 * height)
        margin_far = int(0.37 * height)
        bands = {
            "upper": (max(0, y - margin_far), max(1, y - margin_near)),
            "lower": (
                min(self.image_size - 1, y + height + margin_near),
                min(self.image_size, y + height + margin_far),
            ),
        }
        options = []
        for side, (start, stop) in bands.items():
            if stop <= start:
                continue
            profile = gray[start:stop, x:x + width].mean(axis=0).astype(np.float32)
            profile = cv2.GaussianBlur(profile[None], (9, 1), 2.0)[0]
            peaks, properties = find_peaks(
                profile, distance=max(6, int(0.035 * width)),
                prominence=10.0, width=(2, max(5, int(0.045 * width))),
            )
            prominences = properties.get("prominences", np.asarray([], dtype=float))
            options.append((len(peaks), float(np.sum(prominences)), side, peaks, prominences, profile))
        if not options:
            return [], np.zeros(128, dtype=np.float32), board["bbox"], "missing_band", []
        _, _, side, peaks, prominences, profile = max(options, key=lambda row: (row[0], row[1]))
        positions = (peaks.astype(np.float32) / max(width, 1)).tolist()
        normalized_profile = cv2.resize(profile[None], (128, 1), interpolation=cv2.INTER_AREA)[0]
        normalized_profile = (normalized_profile - np.median(normalized_profile)) / max(
            float(np.std(normalized_profile)), 1.0
        )
        return positions, normalized_profile, board["bbox"], side, prominences.tolist()

    def fit(self, supports: Sequence[ImageLike]):
        profiles = []
        for image in supports:
            _, profile, _, _, _ = self._pin_scene(image)
            profiles.append(profile.reshape(-1).tolist())
        self.prototype = {"profiles": profiles}
        return self

    def extract(self, image: ImageLike):
        positions, profile, board_box, side, prominences = self._pin_scene(image)
        spacing_error = 1.0
        if len(positions) == 4:
            gaps = np.diff(np.asarray(positions, dtype=float))
            spacing_error = float(np.std(gaps) / max(np.mean(gaps), 1e-6))
        profile_error = _template_distance(profile, self.prototype.get("profiles", []))
        facts = {
            "pin_count_error": abs(len(positions) - 4) / 4.0,
            "pin_spacing_error": spacing_error,
            "pin_profile_error": profile_error,
        }
        return facts, {
            "detected_pin_count": len(positions),
            "pin_positions": positions,
            "pin_prominences": prominences,
            "pin_side": side,
            "board_box": list(board_box),
            "profile_distance": profile_error,
        }

    def state_dict(self):
        return {"kind": self.kind, "image_size": self.image_size, "prototype": self.prototype}


class TransistorGrounder:
    kind = "mvtec_transistor_lead_topology_v2"

    def __init__(self, state: Mapping[str, Any] | None = None, image_size: int = 256):
        self.image_size = image_size
        self.prototype: Dict[str, Any] = {}
        if state is not None:
            self.image_size = int(state["image_size"])
            self.prototype = dict(state["prototype"])

    @staticmethod
    def _body_box(hsv: np.ndarray) -> Tuple[int, int, int, int]:
        """Locate the package body without confusing it with PCB rails/holes."""
        size = hsv.shape[0]
        dark = hsv[..., 2] < 85
        lo, hi = int(0.16 * size), int(0.84 * size)
        row_counts = dark[:, lo:hi].sum(axis=1)
        candidates = np.flatnonzero(
            (row_counts > 0.47 * size) & (np.arange(size) < 0.63 * size)
        )
        if not len(candidates):
            return int(0.25 * size), int(0.10 * size), int(0.75 * size), int(0.52 * size)
        y0, y1 = int(candidates.min()), int(candidates.max())
        occupied = (dark[y0:y1 + 1].mean(axis=0) > 0.65).astype(np.uint8)
        occupied = cv2.morphologyEx(
            occupied[None], cv2.MORPH_CLOSE,
            np.ones((1, max(3, int(0.035 * size) | 1)), np.uint8),
        )[0]
        count, _, stats, centroids = cv2.connectedComponentsWithStats(occupied[None], 8)
        valid = [
            index for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] > 0.25 * size
            and 0.25 * size < centroids[index, 0] < 0.75 * size
        ]
        if not valid:
            return int(0.25 * size), y0, int(0.75 * size), y1
        label = max(valid, key=lambda index: stats[index, cv2.CC_STAT_AREA])
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        x1 = x0 + int(stats[label, cv2.CC_STAT_WIDTH])
        return x0, y0, x1, y1

    def _lead_scene(self, image: ImageLike):
        array = _rgb(image, self.image_size)
        hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
        x0, y0, x1, y1 = self._body_box(hsv)
        width = max(x1 - x0, 1)
        # Immediately below the body every normal transistor has exactly three
        # low-saturation metal attachments.  PCB holes and vertical rails lie
        # farther down and cannot create an attachment component here.
        metal = ((hsv[..., 1] < 80) & (hsv[..., 2] > 50)).astype(np.uint8)
        strip_y0 = max(0, y1)
        strip_y1 = min(self.image_size, y1 + int(0.075 * self.image_size))
        projection = (metal[strip_y0:strip_y1, x0:x1].mean(axis=0) > 0.25).astype(np.uint8)
        projection = cv2.morphologyEx(
            projection[None], cv2.MORPH_CLOSE,
            np.ones((1, max(3, int(0.02 * self.image_size) | 1)), np.uint8),
        )[0]
        count, _, stats, centroids = cv2.connectedComponentsWithStats(projection[None], 8)
        attachments = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < max(3, int(0.02 * self.image_size)):
                continue
            attachments.append({
                "x": float((x0 + centroids[label, 0]) / self.image_size),
                "width": float(stats[label, cv2.CC_STAT_WIDTH] / self.image_size),
            })
        attachments.sort(key=lambda row: row["x"])

        # A pose-normalised symbolic lead-zone template records the geometry of
        # the three conductive paths.  Blurring grants a few pixels of
        # registration tolerance, while cut/bent paths remain structurally far.
        lower = min(self.image_size, y1 + int(0.42 * self.image_size))
        crop = metal[max(0, y1 - 2):lower, x0:x1]
        template = cv2.resize(crop.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)
        template = cv2.GaussianBlur(template, (5, 5), 1.0)
        pose = np.asarray([x0, y0, x1, y1], dtype=np.float32) / self.image_size
        return array, (x0, y0, x1, y1), attachments, template, pose

    def fit(self, supports: Sequence[ImageLike]):
        templates, poses = [], []
        for image in supports:
            _, _, _, template, pose = self._lead_scene(image)
            templates.append(template.reshape(-1).tolist())
            poses.append(pose)
        self.prototype = {
            "templates": templates,
            "pose": _prototype_state(poses),
        }
        return self

    def extract(self, image: ImageLike):
        _, body_box, attachments, template, pose = self._lead_scene(image)
        geometry_error = _template_distance(template, self.prototype.get("templates", []))
        pose_error, _ = _prototype_distance([pose], self.prototype["pose"])
        positions = [row["x"] for row in attachments]
        symmetry_error = 1.0
        if len(positions) == 3:
            symmetry_error = abs((positions[1] - positions[0]) - (positions[2] - positions[1]))
        facts = {
            "lead_count_error": abs(len(attachments) - 3) / 3.0,
            "lead_geometry_error": float(geometry_error),
            "lead_symmetry_error": float(symmetry_error),
            "package_pose_error": float(pose_error),
        }
        return facts, {
            "detected_lead_count": len(attachments),
            "lead_attachments": attachments,
            "body_box": list(body_box),
            "template_distance": float(geometry_error),
            "package_pose_distance": float(pose_error),
        }

    def state_dict(self):
        return {"kind": self.kind, "image_size": self.image_size, "prototype": self.prototype}


class BracketBrownGrounder:
    kind = "mpdd_bracket_brown_assembly_v2"

    def __init__(self, state: Mapping[str, Any] | None = None, image_size: int = 512):
        self.image_size = image_size
        self.prototype: Dict[str, Any] = {}
        if state is not None:
            self.image_size = int(state["image_size"])
            self.prototype = dict(state["prototype"])

    def _scene(self, image: ImageLike):
        array = _rgb(image, self.image_size)
        hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
        # Brown bracket is substantially darker than the bright green backdrop.
        dark = ((hsv[..., 2] < 155) & (hsv[..., 1] < 210)).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        candidates = _components(dark, minimum=250, maximum=int(0.45 * self.image_size ** 2))
        object_row = None
        if candidates:
            central = [row for row in candidates
                       if 0.15 * self.image_size < row["centroid"][0] < 0.85 * self.image_size
                       and 0.12 * self.image_size < row["centroid"][1] < 0.90 * self.image_size]
            object_row = max(central or candidates, key=lambda row: row["area"])
        object_mask = object_row["mask"].astype(np.uint8) if object_row is not None else np.zeros_like(dark)
        filled = np.zeros_like(object_mask)
        contours = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if contours:
            outer = max(contours, key=cv2.contourArea)
            cv2.drawContours(filled, [outer], -1, 1, -1)

        # The two functional apertures are green, round regions roughly
        # two-thirds down the vertical leg.  The suspension aperture at the top
        # is intentionally excluded by this object-relative relation.
        holes = []
        if object_row is not None:
            x0, y0, width, height = object_row["bbox"]
            green = (
                (hsv[..., 0] > 30) & (hsv[..., 0] < 60)
                & (hsv[..., 1] > 80) & (hsv[..., 2] > 100) & (filled > 0)
            ).astype(np.uint8)
            for row in _components(
                green,
                minimum=max(5, int(self.image_size * self.image_size * 0.00006)),
                maximum=max(30, int(self.image_size * self.image_size * 0.00060)),
            ):
                bx, by, bw, bh = row["bbox"]
                relative_y = (row["centroid"][1] - y0) / max(height, 1)
                ratio = max(bw, bh) / max(min(bw, bh), 1)
                if 0.50 <= relative_y <= 0.70 and ratio <= 2.3:
                    holes.append((
                        float(row["centroid"][0]), float(row["centroid"][1]),
                        float(row["area"]),
                    ))

        # Instead of trusting unstable highlight component counts, describe the
        # fastener configuration by two support-calibrated symbolic occupancies:
        # silver evidence around the joint and around the distal tip.
        whiteness = (
            hsv[..., 2].astype(np.float32) / 255.0
            * (1.0 - hsv[..., 1].astype(np.float32) / 255.0)
            * filled.astype(np.float32)
        )
        joint_strength = distal_strength = 0.0
        if object_row is not None:
            _, y0, _, height = object_row["bbox"]
            relative_y = (np.arange(self.image_size)[:, None] - y0) / max(height, 1)
            joint_zone = (relative_y >= 0.70) & (relative_y < 0.88) & (filled > 0)
            distal_zone = (relative_y >= 0.88) & (relative_y <= 1.02) & (filled > 0)
            # Upper quantiles are insensitive to the amount of dark bracket in
            # each pose and respond to missing silver caps/bolts.
            if np.any(joint_zone):
                joint_strength = float(np.quantile(whiteness[joint_zone], 0.985))
            if np.any(distal_zone):
                distal_strength = float(np.quantile(whiteness[distal_zone], 0.985))

        # Normalise the silhouette so the static prototype compares bracket
        # geometry rather than global scale/translation.
        silhouette = np.zeros((64, 64), dtype=np.float32)
        if object_row is not None:
            x0, y0, width, height = object_row["bbox"]
            crop = object_mask[y0:y0 + height, x0:x0 + width]
            silhouette = cv2.resize(crop.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)
            silhouette = cv2.GaussianBlur(silhouette, (5, 5), 1.0)
        return array, object_row, object_mask, holes, joint_strength, distal_strength, silhouette

    def fit(self, supports: Sequence[ImageLike]):
        shapes, assembly = [], []
        for image in supports:
            _, object_row, object_mask, _, joint, distal, silhouette = self._scene(image)
            if object_row is not None:
                shapes.append(silhouette.reshape(-1).tolist())
                assembly.append(np.asarray([joint, distal], dtype=np.float32))
        self.prototype = {
            "shape_templates": shapes,
            "assembly": _prototype_state(assembly),
        }
        return self

    def extract(self, image: ImageLike):
        _, object_row, _, holes, joint, distal, silhouette = self._scene(image)
        shape_error = _template_distance(silhouette, self.prototype.get("shape_templates", []))
        assembly_feature = np.asarray([joint, distal], dtype=np.float32)
        assembly_error, per_fastener = _prototype_distance(
            [assembly_feature], self.prototype["assembly"]
        )
        # Missing evidence is directional and should not penalise a harmless
        # specular highlight that is brighter than the four normal supports.
        normal = np.asarray(self.prototype["assembly"].get("rows", []), dtype=float)
        missing_error = 10.0
        if normal.size:
            lower = np.min(normal, axis=0)
            scale = np.maximum(np.std(normal, axis=0), 0.025)
            missing_error = float(np.max(np.maximum(0.0, lower - assembly_feature) / scale))
        facts = {
            "hole_count_error": abs(len(holes) - 2) / 2.0,
            "fastener_count_error": missing_error,
            "fastener_relation_error": float(assembly_error),
            "bracket_geometry_error": shape_error,
        }
        return facts, {
            "detected_hole_count": len(holes),
            "hole_centroids": [list(row[:2]) for row in holes],
            "joint_fastener_strength": joint,
            "distal_fastener_strength": distal,
            "fastener_prototype_distance": per_fastener,
            "object_area": object_row["area"] / float(self.image_size ** 2) if object_row else 0.0,
        }

    def state_dict(self):
        return {"kind": self.kind, "image_size": self.image_size, "prototype": self.prototype}


GROUNDERS = {
    Macaroni2Grounder.kind: Macaroni2Grounder,
    PCB1Grounder.kind: PCB1Grounder,
    TransistorGrounder.kind: TransistorGrounder,
    BracketBrownGrounder.kind: BracketBrownGrounder,
}

CLASS_PROGRAMS = {
    ("mvtec", "transistor"): {
        "grounder": TransistorGrounder,
        "rules": [
            ("lead_count", "lead_count_error", "count_eq"),
            ("lead_geometry", "lead_geometry_error", "shape_consistency"),
            ("lead_symmetry", "lead_symmetry_error", "symmetric_spacing"),
            ("package_pose", "package_pose_error", "relative_position"),
        ],
    },
    ("visa", "macaroni2"): {
        "grounder": Macaroni2Grounder,
        "rules": [
            ("piece_count", "piece_count_error", "count_eq"),
            ("piece_shape", "piece_shape_error", "shape_consistency"),
            ("piece_consistency", "within_scene_shape_inconsistency", "attribute_consistency"),
        ],
    },
    ("visa", "pcb1"): {
        "grounder": PCB1Grounder,
        "rules": [
            ("pin_count", "pin_count_error", "count_eq"),
            ("pin_spacing", "pin_spacing_error", "ordered_spacing"),
            ("pin_profile", "pin_profile_error", "appearance_consistency"),
        ],
    },
    ("mpdd", "bracket_brown"): {
        "grounder": BracketBrownGrounder,
        "rules": [
            ("hole_count", "hole_count_error", "count_eq"),
            ("fastener_presence", "fastener_count_error", "count_eq"),
            ("fastener_configuration", "fastener_relation_error", "relative_position"),
            ("bracket_geometry", "bracket_geometry_error", "shape_consistency"),
        ],
    },
}


def build_class_specific_library(
    dataset: str,
    class_name: str,
    support_paths: Sequence[Union[str, Path]],
    qwen_proposal: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    program = CLASS_PROGRAMS[(dataset, class_name)]
    paths = [Path(path).resolve() for path in support_paths]
    normal_values: Dict[str, list] = {fact: [] for _, fact, _ in program["rules"]}
    fold_diagnostics = []
    for held in range(len(paths)):
        train = [path for index, path in enumerate(paths) if index != held]
        fold_grounder = program["grounder"]().fit(train)
        facts, diagnostics = fold_grounder.extract(paths[held])
        for fact in normal_values:
            normal_values[fact].append(float(facts[fact]))
        fold_diagnostics.append({
            "held_support": str(paths[held]),
            "train_support": [str(path) for path in train],
            "facts": facts,
            "diagnostics": diagnostics,
        })
    rules = []
    for rule_id, fact, predicate in program["rules"]:
        values = np.asarray(normal_values[fact], dtype=float)
        floor = 0.02 if "count" not in fact else 0.0
        margin = max(float(np.std(values)), floor)
        upper = float(np.max(values) + margin)
        scale = max(float(np.std(values)), upper * 0.25, 0.05)
        rules.append({
            "id": rule_id,
            "predicate": predicate,
            "fact": fact,
            "upper": upper,
            "soft_scale": scale,
            "weight": 1.0 / len(program["rules"]),
            "leave_one_support_out_values": values.tolist(),
            "leave_one_support_out_normal_fpr": 0.0,
        })
    grounder = program["grounder"]().fit(paths)
    return {
        "schema_version": "5.0-class-specific-static",
        "dataset": dataset,
        "class_name": class_name,
        "support": [str(path) for path in paths],
        "qwen": dict(qwen_proposal or {}),
        "grounder": grounder.state_dict(),
        "rules": rules,
        "crossfit": fold_diagnostics,
        "runtime_requires_llm": False,
        "test_labels_used_for_training_or_selection": False,
    }


class ClassSpecificRuleEngine:
    def __init__(self, library: Mapping[str, Any]):
        self.library = dict(library)
        state = self.library["grounder"]
        grounder_type = GROUNDERS[state["kind"]]
        self.grounder = grounder_type(state=state)
        self.rules = list(self.library["rules"])

    def evaluate(self, image: ImageLike):
        facts, diagnostics = self.grounder.extract(image)
        evidence = []
        for rule in self.rules:
            excess = max(0.0, float(facts[rule["fact"]]) - float(rule["upper"]))
            violation = float(math.tanh(excess / max(float(rule["soft_scale"]), 1e-8)))
            evidence.append({
                "rule_id": rule["id"], "predicate": rule["predicate"],
                "fact": rule["fact"], "value": float(facts[rule["fact"]]),
                "violation": violation, "weight": float(rule["weight"]),
            })
        score = float(sum(row["weight"] * row["violation"] for row in evidence))
        evidence.sort(key=lambda row: row["weight"] * row["violation"], reverse=True)
        return score, evidence, facts, diagnostics

    @classmethod
    def load(cls, path: Union[str, Path]):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
