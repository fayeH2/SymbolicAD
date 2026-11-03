#!/usr/bin/env python3
"""Aggregate independently trained MVTec LOCO class runs.

The report deliberately keeps the 0.80 gate attached to per-class *overall
image AUROC*.  Logical/structural image metrics and sPRO@5% FPR remain visible
diagnostics; they are never averaged into the pass/fail decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOCO_CLASSES = (
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
)

EXPECTED_TEST_COUNTS = {
    "breakfast_box": {"good": 102, "logical": 83, "structural": 90},
    "juice_bottle": {"good": 94, "logical": 142, "structural": 94},
    "pushpins": {"good": 138, "logical": 91, "structural": 81},
    "screw_bag": {"good": 122, "logical": 137, "structural": 82},
    "splicing_connectors": {"good": 119, "logical": 108, "structural": 85},
}

TRAINING_PROTOCOL_FIELDS = (
    "dataset", "shot", "support_seed", "augmentation_policy",
    "support_selector_schema", "support_split_policy", "epochs", "encoder",
    "input_size", "crop_size", "inp_num",
)

EVALUATION_PROTOCOL_FIELDS = (
    "dataset", "encoder", "inp_num", "input_size", "crop_size", "map_size",
    "compute_spro", "spro_threshold_count", "inp_map_frame",
    "maximum_false_positive_rate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_run(
    entry: str,
) -> tuple[str, Path, Mapping[str, Any], Mapping[str, Any], Path]:
    category, separator, raw_path = entry.partition("=")
    if not separator or category not in LOCO_CLASSES or not raw_path:
        raise ValueError(
            "--run must be one of CLASS=RUN_DIR for the five LOCO classes"
        )
    run_dir = Path(raw_path).expanduser().resolve()
    metrics_path = run_dir / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "mvtec-loco-image-evaluation-v1":
        raise ValueError(f"unsupported metrics schema in {metrics_path}")
    if set(payload.get("per_class", {})) != {category}:
        raise ValueError(
            f"{metrics_path} must contain exactly the declared class {category}"
        )
    if category not in payload.get("localization", {}):
        raise ValueError(f"{metrics_path} has no localization evaluation")
    checkpoint = Path(payload["checkpoint"])
    if not checkpoint.is_file() or _sha256(checkpoint) != payload["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch for {category}")
    manifest_path = checkpoint.parent / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("classes") != [category]:
        raise ValueError(f"training manifest is not class-specific for {category}")
    return category, run_dir, payload, manifest, manifest_path


def _validate_counts(category: str, metrics: Mapping[str, Any]) -> None:
    expected = EXPECTED_TEST_COUNTS[category]
    all_row = metrics["all"]
    logical = metrics["logical"]
    structural = metrics["structural"]
    observed = {
        "good": int(all_row["normal_count"]),
        "logical": int(logical["anomaly_count"]),
        "structural": int(structural["anomaly_count"]),
    }
    if observed != expected:
        raise ValueError(
            f"test sample count mismatch for {category}: {observed} != {expected}"
        )
    if int(all_row["count"]) != sum(expected.values()):
        raise ValueError(f"overall test count mismatch for {category}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _plot(summary: Mapping[str, Any], target: Path) -> None:
    categories = list(summary["class_order"])
    short_names = [name.replace("_", "\n") for name in categories]
    x = np.arange(len(categories), dtype=float)
    overall = np.asarray([
        summary["per_class"][name]["image_metrics"]["all"]["image_auroc"]
        for name in categories
    ])
    logical = np.asarray([
        summary["per_class"][name]["image_metrics"]["logical"]["image_auroc"]
        for name in categories
    ])
    structural = np.asarray([
        summary["per_class"][name]["image_metrics"]["structural"]["image_auroc"]
        for name in categories
    ])
    logical_spro = np.asarray([
        summary["per_class"][name]["localization"]["logical_spro_5fpr"]
        for name in categories
    ])
    structural_spro = np.asarray([
        summary["per_class"][name]["localization"]["structural_spro_5fpr"]
        for name in categories
    ])

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.1), constrained_layout=True)
    colours = np.where(overall >= summary["quality_gate"]["threshold"],
                       "#3F7D5A", "#B34A43")
    bars = axes[0].bar(x, overall, color=colours, edgecolor="#27323A", linewidth=.5)
    axes[0].bar_label(bars, labels=[f"{value:.3f}" for value in overall],
                      padding=3, fontsize=8)
    axes[0].axhline(summary["quality_gate"]["threshold"], color="#A33E3E",
                    linestyle="--", linewidth=1.2, label="0.80 gate")
    axes[0].set_title("Per-class overall detection", loc="left", weight="bold")
    axes[0].set_ylabel("Image AUROC")
    axes[0].legend(frameon=False, loc="lower right")

    width = .36
    bars_l = axes[1].bar(x - width / 2, logical, width, color="#D08A36",
                         label="Logical")
    bars_s = axes[1].bar(x + width / 2, structural, width, color="#356A8A",
                         label="Structural")
    axes[1].bar_label(bars_l, labels=[f"{value:.2f}" for value in logical],
                      padding=2, fontsize=7)
    axes[1].bar_label(bars_s, labels=[f"{value:.2f}" for value in structural],
                      padding=2, fontsize=7)
    axes[1].axhline(.80, color="#A33E3E", linestyle="--", linewidth=1.0)
    axes[1].set_title("Detection by anomaly family", loc="left", weight="bold")
    axes[1].set_ylabel("Image AUROC")
    axes[1].legend(frameon=False, ncols=2, loc="lower right")

    bars_l = axes[2].bar(x - width / 2, logical_spro, width, color="#D08A36",
                         label="Logical")
    bars_s = axes[2].bar(x + width / 2, structural_spro, width, color="#356A8A",
                         label="Structural")
    axes[2].bar_label(bars_l, labels=[f"{value:.2f}" for value in logical_spro],
                      padding=2, fontsize=7)
    axes[2].bar_label(bars_s, labels=[f"{value:.2f}" for value in structural_spro],
                      padding=2, fontsize=7)
    axes[2].set_title("Frame-aware localization", loc="left", weight="bold")
    axes[2].set_ylabel("sPRO@5% FPR")
    axes[2].legend(frameon=False, ncols=2, loc="upper right")

    for axis in axes:
        axis.set_xticks(x, short_names)
        axis.set_ylim(0.0, 1.08)
        axis.grid(axis="y", color="#D7DEE3", linewidth=.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[2].text(
        .0, -0.23, "Project sPRO reproduction; official-package parity not certified",
        transform=axes[2].transAxes, fontsize=8, color="#4A555C",
    )
    fig.suptitle(
        "MVTec LOCO · independent K=4 checkpoints · seed 1 · single-seed exploratory",
        weight="bold",
    )
    fig.savefig(target, dpi=190, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", required=True,
        help="CLASS=directory containing metrics.json; repeat once per class",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gate", type=float, default=.80)
    args = parser.parse_args()

    loaded: Dict[
        str, tuple[Path, Mapping[str, Any], Mapping[str, Any], Path]
    ] = {}
    for entry in args.run:
        category, run_dir, payload, manifest, manifest_path = _load_run(entry)
        if category in loaded:
            raise ValueError(f"duplicate run for {category}")
        loaded[category] = (run_dir, payload, manifest, manifest_path)
    missing = [name for name in LOCO_CLASSES if name not in loaded]
    if missing:
        raise ValueError(f"all five classes are required; missing {missing}")

    protocol_signatures = {
        tuple((field, manifest.get(field)) for field in TRAINING_PROTOCOL_FIELDS)
        for _, _, manifest, _ in loaded.values()
    }
    if len(protocol_signatures) != 1:
        raise ValueError("training protocol differs between class runs")
    protocol = dict(next(iter(protocol_signatures)))
    if protocol["augmentation_policy"] != "official22":
        raise ValueError(
            "the five-class primary table is restricted to the predeclared official22 policy"
        )
    evaluation_signatures = {
        tuple(
            (field, payload.get("evaluation_config", {}).get(field))
            for field in EVALUATION_PROTOCOL_FIELDS
        )
        for _, payload, _, _ in loaded.values()
    }
    if len(evaluation_signatures) != 1:
        raise ValueError("evaluation protocol differs between class runs")
    evaluation_protocol = dict(next(iter(evaluation_signatures)))
    if (
        not evaluation_protocol["compute_spro"]
        or evaluation_protocol["inp_map_frame"] != "model_center_crop"
        or evaluation_protocol["maximum_false_positive_rate"] != 0.05
    ):
        raise ValueError("formal frame-aware localization protocol is incomplete")

    per_class: Dict[str, Any] = {}
    csv_rows = []
    for category in LOCO_CLASSES:
        run_dir, payload, manifest, manifest_path = loaded[category]
        image_metrics = payload["per_class"][category]
        localization = payload["localization"][category]
        _validate_counts(category, image_metrics)
        per_class[category] = {
            "run_dir": str(run_dir),
            "metrics_json": str((run_dir / "metrics.json").resolve()),
            "metrics_json_sha256": _sha256(run_dir / "metrics.json"),
            "checkpoint": payload["checkpoint"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "training_manifest": str(manifest_path.resolve()),
            "training_manifest_sha256": _sha256(manifest_path),
            "support": manifest["support"],
            "convergence": manifest["convergence"],
            "image_metrics": image_metrics,
            "localization": localization,
            "passes_overall_image_auroc_gate": bool(
                image_metrics["all"]["image_auroc"] >= args.gate
            ),
        }
        csv_rows.append({
            "class_name": category,
            "shot": manifest["shot"],
            "support_seed": manifest["support_seed"],
            "augmentation_policy": manifest["augmentation_policy"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "all_image_auroc": image_metrics["all"]["image_auroc"],
            "all_image_ap": image_metrics["all"]["image_ap"],
            "all_count": image_metrics["all"]["count"],
            "normal_count": image_metrics["all"]["normal_count"],
            "anomaly_count": image_metrics["all"]["anomaly_count"],
            "logical_image_auroc": image_metrics["logical"]["image_auroc"],
            "logical_image_ap": image_metrics["logical"]["image_ap"],
            "logical_count": image_metrics["logical"]["count"],
            "logical_anomaly_count": image_metrics["logical"]["anomaly_count"],
            "structural_image_auroc": image_metrics["structural"]["image_auroc"],
            "structural_image_ap": image_metrics["structural"]["image_ap"],
            "structural_count": image_metrics["structural"]["count"],
            "structural_anomaly_count": image_metrics["structural"]["anomaly_count"],
            "logical_spro_5fpr": localization["logical_spro_5fpr"],
            "logical_region_count": localization["logical_region_count"],
            "structural_spro_5fpr": localization["structural_spro_5fpr"],
            "structural_region_count": localization["structural_region_count"],
            "mean_spro_5fpr": localization["mean_spro_5fpr"],
            "gate_threshold": args.gate,
            "gate_pass": image_metrics["all"]["image_auroc"] >= args.gate,
            "official_spro_computed": payload["official_spro_computed"],
        })

    macro = {
        group: {
            metric: float(np.mean([
                per_class[name]["image_metrics"][group][metric]
                for name in LOCO_CLASSES
            ]))
            for metric in ("image_auroc", "image_ap")
        }
        for group in ("all", "logical", "structural")
    }
    macro["localization"] = {
        key: float(np.mean([
            per_class[name]["localization"][key] for name in LOCO_CLASSES
        ]))
        for key in (
            "logical_spro_5fpr", "structural_spro_5fpr", "mean_spro_5fpr"
        )
    }
    pass_map = {
        name: per_class[name]["passes_overall_image_auroc_gate"]
        for name in LOCO_CLASSES
    }
    summary = {
        "schema": "mvtec-loco-independent-class-summary-v1",
        "class_order": list(LOCO_CLASSES),
        "protocol": {
            "checkpoint_scope": "one independently trained checkpoint per class",
            "training": protocol,
            "evaluation": evaluation_protocol,
            "image_subset": "all good plus the named anomaly family",
            "localization": "project frame-aware sPRO@5% FPR reproduction",
            "official_spro_parity_certified": False,
        },
        "per_class": per_class,
        "macro": macro,
        "quality_gate": {
            "metric": "all.image_auroc",
            "threshold": args.gate,
            "per_class_pass": pass_map,
            "all_classes_pass": bool(all(pass_map.values())),
            "failed_classes": [name for name, passed in pass_map.items() if not passed],
            "note": "The gate does not apply to AP or sPRO.",
        },
    }

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", summary)
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    _plot(summary, output / "five_class_metrics.png")
    print(json.dumps(summary["quality_gate"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
