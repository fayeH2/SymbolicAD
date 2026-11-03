#!/usr/bin/env python3
"""Leakage-free image-level evaluation for a trained MVTec LOCO checkpoint.

This command deliberately owns only image AUROC/AP.  Localization is evaluated
by the frame-aware sPRO pipeline, because INP center-crop maps and full-image
rule maps must first be projected into a common original-image frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from symbolicAD.data import collect_test_samples
from symbolicAD.inpformer_adapter import (
    DEFAULT_INPFORMER_ROOT,
    INPFormerAdapter,
    file_sha256,
    load_prediction_cache,
    save_prediction_cache,
)
from symbolicAD.evaluation.spro import (
    LocoEvaluationSample, MapFrame, ScoreMapGeometry, evaluate_spro,
)


LOCO_CLASSES = (
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
)


def defect_group_indices(
    samples: Sequence[Mapping[str, object]], group: str
) -> np.ndarray:
    """Return a valid binary-evaluation subset: good plus one anomaly family."""
    if group not in {"all", "logical", "structural"}:
        raise ValueError(f"Unknown LOCO group: {group}")
    selected = []
    for index, sample in enumerate(samples):
        defect = str(sample["defect_type"]).lower()
        is_good = defect in {"good", "ok"}
        if group == "all" or is_good or group in defect:
            selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("AUROC/AP require both good and anomalous samples")
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_ap": float(average_precision_score(labels, scores)),
        "count": int(len(labels)),
        "normal_count": int(np.sum(labels == 0)),
        "anomaly_count": int(np.sum(labels == 1)),
    }


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _loco_ground_truth(image_path: Path, defect_type: str) -> np.ndarray:
    if defect_type.lower() in {"good", "ok"}:
        with Image.open(image_path) as image:
            width, height = image.size
        return np.zeros((height, width), dtype=np.uint8)
    candidate = (
        image_path.parents[2] / "ground_truth" / defect_type
        / f"{image_path.stem}_mask.png"
    )
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return np.asarray(Image.open(candidate).convert("L"), dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inpformer-root", default=DEFAULT_INPFORMER_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--map-size", type=int, default=256)
    parser.add_argument(
        "--classes", nargs="+", choices=LOCO_CLASSES, default=list(LOCO_CLASSES),
        help="Evaluate only the classes on which this checkpoint was trained.",
    )
    parser.add_argument("--compute-spro", action="store_true")
    parser.add_argument("--spro-thresholds", type=int, default=1000)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    adapter = INPFormerAdapter(
        checkpoint, args.device, args.inpformer_root, args.encoder, args.inp_num,
        args.input_size, args.crop_size, args.map_size,
    )
    checkpoint_metadata = adapter.cache_metadata()
    class_metrics: Dict[str, object] = {}
    localization_metrics: Dict[str, object] = {}
    spro_curve_arrays: Dict[str, np.ndarray] = {}
    detail_rows: List[Dict[str, object]] = []
    selected_classes = tuple(args.classes)
    for class_name in selected_classes:
        samples = collect_test_samples(data_root / class_name)
        paths = [Path(row["path"]) for row in samples]
        identifiers = [str(path.resolve()) for path in paths]
        metadata = {
            **checkpoint_metadata,
            "schema": "loco-image-eval-v1",
            "dataset": "mvtec_loco",
            "class_name": class_name,
        }
        cache_path = output / "cache" / f"{class_name}.npz"
        predictions = load_prediction_cache(
            cache_path, identifiers, expected_metadata=metadata
        )
        if predictions is None:
            predictions = adapter.infer(paths, batch_size=args.batch_size)
            save_prediction_cache(cache_path, predictions, metadata)
        labels = np.asarray([row["label"] for row in samples], dtype=np.uint8)
        scores = np.asarray(predictions["scores"], dtype=float)
        groups = {}
        for group in ("all", "logical", "structural"):
            indices = defect_group_indices(samples, group)
            groups[group] = binary_metrics(labels[indices], scores[indices])
        class_metrics[class_name] = groups
        if args.compute_spro:
            evaluation_samples = []
            for sample, score_map in zip(samples, predictions["maps"]):
                path = Path(sample["path"]).resolve()
                defect = str(sample["defect_type"])
                family = (
                    "good" if defect.lower() in {"good", "ok"}
                    else "logical" if "logical" in defect.lower()
                    else "structural"
                )
                evaluation_samples.append(LocoEvaluationSample(
                    sample_id=str(path), category=class_name,
                    anomaly_type=family,
                    ground_truth=_loco_ground_truth(path, defect),
                    maps={"inp": np.asarray(score_map, dtype=np.float32)},
                ))
            geometry = ScoreMapGeometry(
                MapFrame.MODEL_CENTER_CROP,
                input_size=args.input_size, crop_size=args.crop_size,
            )
            class_localization = {}
            for family in ("logical", "structural"):
                curve = evaluate_spro(
                    evaluation_samples, "inp", family,
                    threshold_count=args.spro_thresholds, geometry=geometry,
                )
                class_localization[f"{family}_spro_5fpr"] = curve.normalized_area
                class_localization[f"{family}_region_count"] = curve.region_count
                spro_curve_arrays[f"{class_name}__{family}__fpr"] = (
                    curve.false_positive_rates
                )
                spro_curve_arrays[f"{class_name}__{family}__spro"] = (
                    curve.saturated_overlaps
                )
            class_localization["mean_spro_5fpr"] = float(np.mean([
                class_localization["logical_spro_5fpr"],
                class_localization["structural_spro_5fpr"],
            ]))
            localization_metrics[class_name] = class_localization
        for sample, score in zip(samples, scores):
            detail_rows.append({
                "class_name": class_name,
                "path": str(Path(sample["path"]).resolve()),
                "defect_type": sample["defect_type"],
                "label": int(sample["label"]),
                "score": float(score),
            })
    adapter.close()

    macro = {
        group: {
            metric: float(np.mean([
                class_metrics[name][group][metric] for name in selected_classes
            ]))
            for metric in ("image_auroc", "image_ap")
        }
        for group in ("all", "logical", "structural")
    }
    result = {
        "schema": "mvtec-loco-image-evaluation-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "evaluation_config": {
            "dataset": "mvtec_loco",
            "classes": list(selected_classes),
            "encoder": args.encoder,
            "inp_num": args.inp_num,
            "input_size": args.input_size,
            "crop_size": args.crop_size,
            "map_size": args.map_size,
            "batch_size": args.batch_size,
            "compute_spro": bool(args.compute_spro),
            "spro_threshold_count": args.spro_thresholds,
            "inp_map_frame": MapFrame.MODEL_CENTER_CROP.value,
            "maximum_false_positive_rate": 0.05,
        },
        "metric_scope": (
            "image AUROC/AP plus project frame-aware paper-table sPRO reproduction"
            if args.compute_spro else "image-level only"
        ),
        "official_spro_computed": False,
        "official_spro_reason": (
            "project implementation uses the published saturation tables and "
            "5% FPR protocol; numerical parity with the official evaluation "
            "package has not yet been certified"
        ),
        "subset_policy": "each anomaly family plus all good test images",
        "per_class": class_metrics,
        "macro": macro,
        "localization": localization_metrics,
        "quality_gate": {
            "metric": "all.image_auroc",
            "threshold": 0.80,
            "per_class_pass": {
                name: bool(class_metrics[name]["all"]["image_auroc"] >= 0.80)
                for name in selected_classes
            },
            "all_classes_pass": bool(all(
                class_metrics[name]["all"]["image_auroc"] >= 0.80
                for name in selected_classes
            )),
            "note": (
                "This gate applies only to image AUROC; pixel AP and "
                "sPRO@5%FPR are reported separately and are not rescaled."
            ),
        },
    }
    _dump(output / "metrics.json", result)
    if spro_curve_arrays:
        np.savez_compressed(output / "spro_curves.npz", **spro_curve_arrays)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    print(json.dumps(result["macro"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
