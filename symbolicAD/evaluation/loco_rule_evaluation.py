"""Held-out MVTec LOCO evaluation for executable rule sets.

The search/validation layer never imports this module.  Test labels and GT
masks enter only here, after candidate admission and rule-set selection are
frozen.  Image metrics compare the raw executable Qwen set with the evolved
set; localization evaluates each logic map in its declared full-image frame.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from ..reasoning.constraint_executor import ConstraintOperator, combine_statuses
from ..reasoning.predicate_executor import ExecutionContext
from ..reasoning.violation_result import RuleStatus
from ..rules.rule_registry import RuleRegistry, default_rule_registry
from ..rules.rule_schema import RuleSpec
from .spro import (
    LocoEvaluationSample,
    MapFrame,
    ScoreMapGeometry,
    evaluate_spro,
)


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("binary metrics require good and anomalous samples")
    finite = np.isfinite(scores)
    evaluated_labels = labels[finite]
    evaluated_scores = scores[finite]
    result = {
        "count": int(labels.size),
        "normal_count": int(np.sum(labels == 0)),
        "anomaly_count": int(np.sum(labels == 1)),
        "evaluated_count": int(np.sum(finite)),
        "abstention_count": int(np.sum(~finite)),
        "evaluation_coverage": float(np.mean(finite)),
        "evaluated_normal_count": int(np.sum(evaluated_labels == 0)),
        "evaluated_anomaly_count": int(np.sum(evaluated_labels == 1)),
    }
    if set(np.unique(evaluated_labels)) != {0, 1}:
        result.update({
            "image_auroc": None,
            "image_ap": None,
            "metric_error": "decided subset does not contain both classes",
        })
        return result
    result.update({
        "image_auroc": float(roc_auc_score(evaluated_labels, evaluated_scores)),
        "image_ap": float(average_precision_score(evaluated_labels, evaluated_scores)),
    })
    return result


def _groups(defect_types: Sequence[str]) -> Dict[str, np.ndarray]:
    values = np.asarray(defect_types)
    good = values == "good"
    return {
        "all": np.arange(len(values), dtype=np.int64),
        "logical": np.flatnonzero(good | (values == "logical_anomalies")),
        "structural": np.flatnonzero(good | (values == "structural_anomalies")),
    }


def _ground_truth(path: Path, defect_type: str) -> np.ndarray:
    if defect_type == "good":
        with Image.open(path) as image:
            width, height = image.size
        return np.zeros((height, width), dtype=np.uint8)
    class_root = path.parents[2]
    mask = class_root / "ground_truth" / defect_type / f"{path.stem}_mask.png"
    if not mask.is_file():
        raise FileNotFoundError(mask)
    return np.asarray(Image.open(mask).convert("L"), dtype=np.uint8)


def _execute_set(
    rules: Sequence[RuleSpec],
    context: ExecutionContext,
    registry: RuleRegistry,
) -> tuple[float, np.ndarray, float, Dict[str, np.ndarray], RuleStatus]:
    maps = {}
    results = []
    known = 0
    for rule in rules:
        capability = registry.resolve(rule.canonical_predicate)
        if capability is None:
            maps[rule.rule_id] = np.zeros(
                (context.map_size, context.map_size), dtype=np.float32
            )
            continue
        result = capability.executor.execute(rule, context)
        results.append(result)
        if result.status != RuleStatus.UNKNOWN:
            known += 1
        maps[rule.rule_id] = (
            np.asarray(result.anomaly_map, dtype=np.float32)
            if result.status == RuleStatus.VIOLATED and result.anomaly_map is not None
            else np.zeros((context.map_size, context.map_size), dtype=np.float32)
        )
    status_values = [row.status for row in results]
    status_values.extend(
        RuleStatus.UNKNOWN for _ in range(len(rules) - len(results))
    )
    status = combine_statuses(status_values, ConstraintOperator.AND)
    logic = np.zeros((context.map_size, context.map_size), dtype=np.float32)
    if status == RuleStatus.VIOLATED:
        for values in maps.values():
            logic = np.maximum(logic, values)
    if status == RuleStatus.VIOLATED:
        score = max(
            row.violation_score for row in results
            if row.status == RuleStatus.VIOLATED
        )
    elif status == RuleStatus.SATISFIED:
        score = 0.0
    else:
        # NaN means explicit abstention.  It is excluded from selective AUROC
        # and counted in coverage; UNKNOWN is never silently imputed as normal.
        score = float("nan")
    return (
        float(score),
        logic,
        known / len(rules) if rules else 0.0,
        maps,
        status,
    )


def evaluate_rule_sets(
    *,
    category: str,
    split_facts: Mapping[str, Mapping[str, ExecutionContext]],
    rule_sets: Mapping[str, Sequence[RuleSpec]],
    output_dir: str | Path,
    registry: RuleRegistry | None = None,
    threshold_count: int = 1000,
    minimum_evaluation_coverage: float = 0.95,
) -> Dict[str, Any]:
    """Evaluate frozen rule sets and persist predictions, maps and sPRO curves."""
    registry = registry or default_rule_registry()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ordered = []
    for defect_type in ("good", "logical_anomalies", "structural_anomalies"):
        for path, context in sorted(split_facts.get(defect_type, {}).items()):
            ordered.append((Path(path), defect_type, context))
    if not ordered:
        raise ValueError("test facts are empty")

    paths = [str(path.resolve()) for path, _, _ in ordered]
    defect_types = [defect for _, defect, _ in ordered]
    labels = np.asarray([defect != "good" for defect in defect_types], dtype=np.uint8)
    ground_truths = [_ground_truth(path, defect) for path, defect, _ in ordered]
    map_geometry = ScoreMapGeometry(MapFrame.FULL_IMAGE_NORMALIZED)

    result: Dict[str, Any] = {
        "schema": "loco-frozen-rule-evaluation-v1",
        "category": category,
        "test_only_after_rule_freeze": True,
        "map_frame": map_geometry.frame.value,
        "rule_sets": {},
        "quality_gate": {
            "metric": "all.image_auroc", "threshold": 0.80,
            "minimum_evaluation_coverage": minimum_evaluation_coverage,
            "note": (
                "The 0.80 gate requires the declared decision coverage and "
                "does not apply to sPRO or pixel AP."
            ),
        },
    }
    prediction_rows = [
        {"path": path, "defect_type": defect, "label": int(label)}
        for path, defect, label in zip(paths, defect_types, labels)
    ]
    saved_arrays: Dict[str, np.ndarray] = {
        "paths": np.asarray(paths),
        "defect_types": np.asarray(defect_types),
        "labels": labels,
    }
    spro_curve_arrays: Dict[str, np.ndarray] = {}

    for set_name, rules in rule_sets.items():
        rules = tuple(rules)
        image_scores, logic_maps, known_fractions, statuses = [], [], [], []
        # Raw-Qwen is a set-level control; only the compact evolved set gets
        # expensive per-rule localization curves.
        selected_rule_maps: Dict[str, list[np.ndarray]] = (
            {rule.rule_id: [] for rule in rules}
            if set_name == "evolved" else {}
        )
        for row_index, (_, _, context) in enumerate(ordered):
            score, logic, known, maps, status = _execute_set(
                rules, context, registry
            )
            image_scores.append(score)
            logic_maps.append(logic)
            known_fractions.append(known)
            statuses.append(status)
            for rule_id in selected_rule_maps:
                selected_rule_maps[rule_id].append(maps[rule_id])
            prediction_rows[row_index][f"{set_name}_score"] = score
            prediction_rows[row_index][f"{set_name}_known_fraction"] = known
            prediction_rows[row_index][f"{set_name}_status"] = status.value
        scores = np.asarray(image_scores, dtype=np.float32)
        logic_array = np.stack(logic_maps).astype(np.float32)
        saved_arrays[f"{set_name}__logic_total"] = logic_array
        for rule_id, maps in selected_rule_maps.items():
            saved_arrays[f"{set_name}__rule__{rule_id}"] = np.stack(maps).astype(np.float32)

        metrics = {}
        for group, indices in _groups(defect_types).items():
            metrics[group] = _binary_metrics(labels[indices], scores[indices])
        metrics["all"]["passes_0_80_gate"] = bool(
            metrics["all"]["image_auroc"] is not None
            and metrics["all"]["image_auroc"] >= 0.80
            and metrics["all"]["evaluation_coverage"]
            >= float(minimum_evaluation_coverage)
        )

        # Pixel AUROC/AP at the executor resolution is a transparent
        # diagnostic.  Formal LOCO localization is the frame-aware sPRO below.
        small_gt = np.stack([
            cv2.resize(
                (gt > 0).astype(np.uint8), logic_array.shape[-2:][::-1],
                interpolation=cv2.INTER_NEAREST,
            )
            for gt in ground_truths
        ]).astype(bool)
        if np.any(small_gt) and np.any(~small_gt):
            metrics["diagnostic_pixel"] = {
                "auroc": float(roc_auc_score(small_gt.ravel(), logic_array.ravel())),
                "ap": float(average_precision_score(small_gt.ravel(), logic_array.ravel())),
                "resolution": list(logic_array.shape[-2:]),
            }

        map_families = {"logic_total": logic_array}
        # Per-rule localization is required for attribution diagnosis.  It is
        # tractable because evolution is expected to return a small set.
        map_families.update({
            f"rule::{rule_id}": np.stack(maps).astype(np.float32)
            for rule_id, maps in selected_rule_maps.items()
        })
        spro = {}
        for map_name, map_values in map_families.items():
            samples = [
                LocoEvaluationSample(
                    sample_id=paths[index], category=category,
                    anomaly_type=(
                        "good" if defect_types[index] == "good"
                        else "logical" if defect_types[index] == "logical_anomalies"
                        else "structural"
                    ),
                    ground_truth=ground_truths[index],
                    maps={map_name: map_values[index]},
                )
                for index in range(len(paths))
            ]
            map_result = {}
            for anomaly_type in ("logical", "structural"):
                curve = evaluate_spro(
                    samples, map_name, anomaly_type,
                    threshold_count=threshold_count, geometry=map_geometry,
                )
                map_result[f"{anomaly_type}_spro_5fpr"] = curve.normalized_area
                map_result[f"{anomaly_type}_region_count"] = curve.region_count
                prefix = f"{set_name}__{map_name}__{anomaly_type}"
                spro_curve_arrays[f"{prefix}__fpr"] = curve.false_positive_rates
                spro_curve_arrays[f"{prefix}__spro"] = curve.saturated_overlaps
            map_result["mean_spro_5fpr"] = float(np.mean([
                map_result["logical_spro_5fpr"],
                map_result["structural_spro_5fpr"],
            ]))
            spro[map_name] = map_result

        result["rule_sets"][set_name] = {
            "rule_ids": [rule.rule_id for rule in rules],
            "rules": [rule.to_dict() for rule in rules],
            "mean_known_fraction": float(np.mean(known_fractions)),
            "decision_coverage": float(np.mean([
                status != RuleStatus.UNKNOWN for status in statuses
            ])),
            "status_counts": {
                status.value: int(sum(row == status for row in statuses))
                for status in RuleStatus
            },
            "metrics": metrics,
            "spro": spro,
        }

    result["quality_gate"]["per_rule_set_pass"] = {
        name: row["metrics"]["all"]["passes_0_80_gate"]
        for name, row in result["rule_sets"].items()
    }
    np.savez_compressed(output / "rule_maps.npz", **saved_arrays)
    np.savez_compressed(output / "spro_curves.npz", **spro_curve_arrays)
    with (output / "rule_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    (output / "rule_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


__all__ = ["evaluate_rule_sets"]
