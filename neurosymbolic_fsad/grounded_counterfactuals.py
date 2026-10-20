"""Image-grounded counterfactual supervision for normal-only FSAD.

Unlike atom-space counterfactuals, these interventions alter a normal support
image and then run the visual grounder again.  Labels therefore describe an
image-level causal intervention; they do not prescribe which rule atom must
change or which Boolean expression should win.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import cv2
import numpy as np

from .transistor_logic import ATOM_SPECS, SpatialTransistorGrounder


@dataclass(frozen=True)
class GroundedCounterfactual:
    source: str
    kind: str
    label: int
    image: np.ndarray
    intervention_mask: np.ndarray
    parameters: Dict[str, Any]


def _inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if not np.any(mask8):
        return image.copy()
    return cv2.inpaint(image.astype(np.uint8), mask8, 5, cv2.INPAINT_TELEA)


def _erase_with_board_color(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Erase conductive material with a locally plausible non-metal colour.

    OpenCV inpainting tends to copy neighbouring silver pixels back into a
    narrow lead mask, leaving the downstream metal detector unchanged.  This
    fill is estimated from saturated board pixels and is therefore explicitly
    outside the grounder's low-saturation metal predicate.
    """
    result = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    board = image[(hsv[..., 1] > 105) & (hsv[..., 2] > 45)]
    colour = np.median(board, axis=0).astype(np.uint8) if len(board) else np.asarray(
        [110, 65, 55], dtype=np.uint8
    )
    result[np.asarray(mask).astype(bool)] = colour
    return result


def _lead_x(diagnostics: Mapping[str, Any], index: int, size: int) -> int:
    rows = list(diagnostics.get("lead_attachments", []))
    if rows:
        row = rows[index % len(rows)]
        return int(round(float(row["x"]) * size))
    x0, _, x1, _ = (int(value) for value in diagnostics["body_box"])
    return int(round(x0 + (index + 1) * (x1 - x0) / 4.0))


def _rectangle(size: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(
        result,
        (max(0, x0), max(0, y0)),
        (min(size - 1, x1), min(size - 1, y1)),
        1,
        -1,
    )
    return result


def generate_transistor_counterfactuals(
    support_paths: Sequence[str | Path],
    grounder: SpatialTransistorGrounder,
    *,
    seed: int = 668,
) -> list[GroundedCounterfactual]:
    """Generate nuisance-preserving normals and causal lead violations.

    The intervention families are a closed, auditable compiler target for LLM
    failure-mode proposals: ``detach_lead``, ``cut_lead``, ``bend_lead`` and
    ``extra_lead``.  The compiler never edits atom truth values directly.
    """
    rng = np.random.default_rng(seed)
    records: list[GroundedCounterfactual] = []
    for support_index, path_like in enumerate(support_paths):
        path = Path(path_like).resolve()
        _, _, diagnostics, prepared = grounder.extract_spatial(path)
        size = int(prepared.shape[0])
        _, _, _, body_y1 = (int(value) for value in diagnostics["body_box"])
        lead_index = support_index % 3
        center_x = _lead_x(diagnostics, lead_index, size)

        zero = np.zeros((size, size), dtype=np.uint8)
        records.append(GroundedCounterfactual(
            str(path), "normal_identity", 0, prepared.copy(), zero.copy(), {},
        ))

        # Normal-preserving nuisance transforms discourage lighting/blur atoms
        # and give the normal false-positive term more than memorised identities.
        gain = 0.90 if support_index % 2 == 0 else 1.10
        brightness = np.clip(prepared.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        records.append(GroundedCounterfactual(
            str(path), "normal_brightness", 0, brightness, zero.copy(), {"gain": gain},
        ))
        blur = cv2.GaussianBlur(prepared, (5, 5), 0.8)
        records.append(GroundedCounterfactual(
            str(path), "normal_blur", 0, blur, zero.copy(), {"sigma": 0.8},
        ))

        # Remove the metal immediately below the package.  This may affect
        # count, geometry, both, or neither; only the grounder's observation is
        # supplied to evolution.
        detach_mask = _rectangle(
            size, center_x - 13, body_y1 - 2,
            center_x + 13, body_y1 + int(0.13 * size),
        )
        records.append(GroundedCounterfactual(
            str(path), "detach_lead", 1,
            _erase_with_board_color(prepared, detach_mask), detach_mask,
            {"lead_index": lead_index},
        ))

        # Break continuity farther from the attachment while leaving the
        # attachment count mostly intact.
        cut_y = body_y1 + int((0.13 + 0.02 * (support_index % 3)) * size)
        cut_mask = _rectangle(
            size, center_x - 13, cut_y - 6,
            center_x + 13, min(size - 1, body_y1 + int(0.40 * size)),
        )
        records.append(GroundedCounterfactual(
            str(path), "cut_lead", 1,
            _erase_with_board_color(prepared, cut_mask), cut_mask,
            {"lead_index": lead_index, "cut_y": cut_y},
        ))

        # Move a lead segment sideways.  Inpaint the source first, then paste
        # only low-saturation bright metal pixels into the shifted location.
        y0 = body_y1 + int(0.08 * size)
        y1 = min(size, body_y1 + int(0.34 * size))
        x0, x1 = max(0, center_x - 13), min(size, center_x + 14)
        roi = prepared[y0:y1, x0:x1].copy()
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        metal = ((hsv[..., 1] < 90) & (hsv[..., 2] > 55)).astype(np.uint8)
        source_mask = np.zeros((size, size), dtype=np.uint8)
        source_mask[y0:y1, x0:x1] = metal
        bent = _erase_with_board_color(prepared, source_mask)
        shift = (-1 if support_index % 2 else 1) * int(0.105 * size)
        target_x0, target_x1 = x0 + shift, x1 + shift
        valid0, valid1 = max(0, target_x0), min(size, target_x1)
        if valid1 > valid0:
            source0 = valid0 - target_x0
            source1 = source0 + valid1 - valid0
            patch = bent[y0:y1, valid0:valid1]
            probe = roi[:, source0:source1]
            probe_mask = metal[:, source0:source1].astype(bool)
            patch[probe_mask] = probe[probe_mask]
            bent[y0:y1, valid0:valid1] = patch
        bend_mask = np.maximum(
            source_mask,
            _rectangle(size, target_x0, y0, target_x1, y1),
        )
        records.append(GroundedCounterfactual(
            str(path), "bend_lead", 1, bent, bend_mask,
            {"lead_index": lead_index, "horizontal_shift": shift},
        ))

        # Duplicate an existing conductive path into the widest neighbouring
        # gap.  This supplies a count/topology violation without naming the atom.
        body_x0, _, body_x1, _ = (int(value) for value in diagnostics["body_box"])
        extra_x = body_x0 + 14 if lead_index != 0 else body_x1 - 14
        extra = prepared.copy()
        source = prepared[body_y1:min(size, body_y1 + int(0.34 * size)), x0:x1]
        source_hsv = cv2.cvtColor(source, cv2.COLOR_RGB2HSV)
        source_metal = ((source_hsv[..., 1] < 90) & (source_hsv[..., 2] > 55))
        tx0 = max(0, extra_x - source.shape[1] // 2)
        tx1 = min(size, tx0 + source.shape[1])
        source = source[:, :tx1 - tx0]
        source_metal = source_metal[:, :tx1 - tx0]
        target = extra[body_y1:body_y1 + source.shape[0], tx0:tx1]
        target[source_metal] = source[source_metal]
        extra[body_y1:body_y1 + source.shape[0], tx0:tx1] = target
        extra_mask = np.zeros((size, size), dtype=np.uint8)
        extra_mask[body_y1:body_y1 + source.shape[0], tx0:tx1] = source_metal
        records.append(GroundedCounterfactual(
            str(path), "extra_lead", 1, extra, extra_mask,
            {"source_lead_index": lead_index, "target_x": extra_x},
        ))
    rng.shuffle(records)
    return records


def ground_counterfactual_truths(
    records: Sequence[GroundedCounterfactual],
    grounder: SpatialTransistorGrounder,
    rules: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Dict[str, Any]]]:
    """Re-observe interventions and return atom normality truths."""
    by_id = {str(rule["id"]): rule for rule in rules}
    atom_order = [str(rule["id"]) for rule in rules]
    rows, labels, groups, audit = [], [], [], []
    for record in records:
        facts, _, diagnostics, _ = grounder.extract_spatial(record.image)
        truths, violations = [], {}
        for atom in atom_order:
            rule = by_id[atom]
            fact_name = str(rule.get("fact", ATOM_SPECS[atom]["fact"]))
            value = float(facts[fact_name])
            excess = max(0.0, value - float(rule["upper"]))
            violation = float(np.tanh(excess / max(float(rule["scale"]), 1e-8)))
            truths.append(1.0 - violation)
            violations[atom] = violation
        rows.append(truths)
        labels.append(record.label)
        groups.append(record.kind)
        audit.append({
            "source": record.source,
            "kind": record.kind,
            "label": record.label,
            "parameters": record.parameters,
            "facts": facts,
            "violations": violations,
            "diagnostics": diagnostics,
        })
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.uint8),
        np.asarray(groups, dtype=str),
        audit,
    )


__all__ = [
    "GroundedCounterfactual",
    "generate_transistor_counterfactuals",
    "ground_counterfactual_truths",
]
