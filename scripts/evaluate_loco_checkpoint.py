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
        "metric_scope": "image-level only",
        "official_spro_computed": False,
        "official_spro_reason": "computed separately after explicit map-frame projection",
        "subset_policy": "each anomaly family plus all good test images",
        "per_class": class_metrics,
        "macro": macro,
    }
    _dump(output / "metrics.json", result)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    print(json.dumps(result["macro"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
