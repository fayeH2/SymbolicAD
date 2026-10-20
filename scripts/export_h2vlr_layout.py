#!/usr/bin/env python3
"""Export multi-dataset runs in the H2VLR result layout.

The image-level metric is the compiled baseline-plus-rules score.  Pixel AUROC
and heatmaps continue to use the INP-Former anomaly map because the current
symbolic rules operate at image level and do not synthesize a pixel mask.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT / "configs" / "multidataset_v6.json"
NORMAL_NAMES = {"good", "ok"}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _classes(entry: dict[str, Any]) -> list[str]:
    configured = entry.get("classes", [])
    if configured != "discover_mvt_like":
        return list(configured)
    root = Path(entry["data_root"])
    return sorted(
        child.name for child in root.iterdir()
        if child.is_dir() and (child / "train" / "good").is_dir()
        and (child / entry.get("test_subdir", "test")).is_dir()
    )


def _safe_name(value: str) -> str:
    value = value.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^0-9A-Za-z_.()\-]+", "_", value).strip("_")


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _resize_center_crop(
    image: np.ndarray, input_size: int, crop_size: int, interpolation: int,
) -> np.ndarray:
    resized = cv2.resize(image, (input_size, input_size), interpolation=interpolation)
    start = (input_size - crop_size) // 2
    return resized[start:start + crop_size, start:start + crop_size]


def _mask_index(class_root: Path, test_subdir: str) -> tuple[dict[tuple[str, str], Path], dict[str, list[Path]]]:
    by_defect: dict[tuple[str, str], Path] = {}
    by_stem: dict[str, list[Path]] = {}
    roots = [
        class_root / "ground_truth",
        class_root / test_subdir / "ground_truth",
        class_root / "masks",
        class_root / test_subdir / "masks",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(root)
            defect = relative.parts[0].lower() if len(relative.parts) > 1 else ""
            stem = path.stem
            if stem.lower().endswith("_mask"):
                stem = stem[:-5]
            stem = stem.lower()
            by_defect.setdefault((defect, stem), path)
            by_stem.setdefault(stem, []).append(path)
    return by_defect, by_stem


def _find_mask(
    image_path: Path, class_root: Path, test_subdir: str,
    index: tuple[dict[tuple[str, str], Path], dict[str, list[Path]]],
) -> Path | None:
    try:
        relative = image_path.resolve().relative_to((class_root / test_subdir).resolve())
    except ValueError:
        return None
    if len(relative.parts) < 2 or relative.parts[0].lower() in NORMAL_NAMES:
        return None
    defect = relative.parts[0].lower()
    stem = image_path.stem.lower()
    by_defect, by_stem = index
    matched = by_defect.get((defect, stem))
    if matched is not None:
        return matched
    candidates = by_stem.get(stem, [])
    return candidates[0] if len(candidates) == 1 else None


def _load_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item())) \
            if "metadata_json" in payload else {}
        return {
            "paths": payload["paths"].astype(str),
            "maps": payload["maps"].astype(np.float32),
            "scores": payload["scores"].astype(np.float32),
            "metadata": metadata,
        }


def _prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _histogram_auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    pos_total, neg_total = int(positive.sum()), int(negative.sum())
    if not pos_total or not neg_total:
        return None
    tpr = np.concatenate(([0.0], np.cumsum(positive[::-1]) / pos_total, [1.0]))
    fpr = np.concatenate(([0.0], np.cumsum(negative[::-1]) / neg_total, [1.0]))
    # NumPy 2 removed the long-deprecated ``trapz`` alias.
    return float(np.trapezoid(tpr, fpr))


def _ensure_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink checkpoint export: {link}")
    link.symlink_to(target.resolve())


def _metric_paths(output_root: Path, dataset: str, class_name: str, shot: int) -> tuple[Path, Path, Path]:
    run = output_root / "generic" / dataset / class_name / f"k{shot}"
    cache = output_root / "cache" / dataset / class_name / f"k{shot}" / "cache" / "test_inp_v4.npz"
    return run / "metrics_v4_generic.json", run / "predictions_v4_generic.csv", cache


def _dataset_map_range(
    output_root: Path, dataset: str, classes: Iterable[str], shot: int,
) -> tuple[float, float] | None:
    minimum, maximum = np.inf, -np.inf
    for class_name in classes:
        _, _, cache_path = _metric_paths(output_root, dataset, class_name, shot)
        if not cache_path.is_file():
            continue
        with np.load(cache_path, allow_pickle=False) as payload:
            maps = payload["maps"]
            if maps.size:
                minimum = min(minimum, float(np.min(maps)))
                maximum = max(maximum, float(np.max(maps)))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return None
    return minimum, maximum


def _export_class(
    entry: dict[str, Any], class_name: str, output_root: Path, destination: Path,
    shot: int, seed: int, visual_size: int, map_range: tuple[float, float],
    max_visualizations: int, histogram_bins: int,
) -> dict[str, Any]:
    dataset = entry["id"]
    metric_path, prediction_path, cache_path = _metric_paths(
        output_root, dataset, class_name, shot
    )
    missing = [str(path) for path in (metric_path, prediction_path, cache_path) if not path.is_file()]
    if missing:
        return {"dataset": dataset, "class_name": class_name, "missing": missing}

    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    rows = _prediction_rows(prediction_path)
    cached = _load_cache(cache_path)
    cached_paths = [str(Path(path).resolve()) for path in cached["paths"]]
    row_paths = [str(Path(row["path"]).resolve()) for row in rows]
    if row_paths != cached_paths:
        raise ValueError(f"Prediction/cache path mismatch for {dataset}/{class_name}")

    metadata = cached["metadata"]
    input_size = int(metadata.get("input_size", 448))
    crop_size = int(metadata.get("crop_size", 392))
    maps = cached["maps"]
    map_size = int(maps.shape[-1])
    class_root = Path(entry["data_root"]) / class_name
    test_subdir = entry.get("test_subdir", "test")
    mask_index = _mask_index(class_root, test_subdir)
    positive_hist = np.zeros(histogram_bins, dtype=np.int64)
    negative_hist = np.zeros(histogram_bins, dtype=np.int64)
    anomaly_mask_count = 0
    missing_anomaly_masks = 0
    visualized = 0
    map_min, map_max = map_range
    map_scale = max(map_max - map_min, 1e-12)
    imgs_dir = destination / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    for index, (row, score_map) in enumerate(zip(rows, maps)):
        image_path = Path(row["path"])
        label = int(row["label"])
        mask_path = _find_mask(image_path, class_root, test_subdir, mask_index)
        mask_crop: np.ndarray | None = None
        if label == 0:
            mask_map = np.zeros((map_size, map_size), dtype=bool)
        elif mask_path is not None:
            with Image.open(mask_path) as image:
                raw_mask = np.asarray(image.convert("L"))
            mask_crop = _resize_center_crop(
                raw_mask, input_size, crop_size, cv2.INTER_NEAREST
            )
            mask_map = cv2.resize(
                mask_crop, (map_size, map_size), interpolation=cv2.INTER_NEAREST
            ) > 0
            anomaly_mask_count += 1
        else:
            mask_map = None
            missing_anomaly_masks += 1

        if mask_map is not None:
            local_min, local_max = float(np.min(score_map)), float(np.max(score_map))
            denom = max(map_max - map_min, local_max - local_min, 1e-12)
            bins = np.clip(
                ((score_map - map_min) / denom * (histogram_bins - 1)).astype(np.int64),
                0, histogram_bins - 1,
            )
            positive_hist += np.bincount(bins[mask_map], minlength=histogram_bins)
            negative_hist += np.bincount(bins[~mask_map], minlength=histogram_bins)

        if max_visualizations and visualized >= max_visualizations:
            continue
        original = _resize_center_crop(
            _read_rgb(image_path), input_size, crop_size, cv2.INTER_LINEAR
        )
        original = cv2.resize(
            original, (visual_size, visual_size), interpolation=cv2.INTER_LINEAR
        )
        if mask_crop is None:
            display_mask = np.zeros((visual_size, visual_size), dtype=bool)
        else:
            display_mask = cv2.resize(
                mask_crop, (visual_size, visual_size), interpolation=cv2.INTER_NEAREST
            ) > 0
        gt = original.copy()
        gt[display_mask] = np.asarray([255, 0, 0], dtype=np.uint8)
        normalized = np.clip((score_map - map_min) / map_scale * 255.0, 0, 255).astype(np.uint8)
        normalized = cv2.resize(
            normalized, (visual_size, visual_size), interpolation=cv2.INTER_LINEAR
        )
        heat_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original, 0.5, heat_rgb, 0.5, 0.0)
        defect = _safe_name(str(row["defect_type"]))
        base = f"{_safe_name(class_name)}-{defect}-{_safe_name(image_path.stem)}"
        Image.fromarray(original).save(imgs_dir / f"{base}_ori.jpg", quality=95)
        Image.fromarray(gt).save(imgs_dir / f"{base}_gt.jpg", quality=95)
        Image.fromarray(overlay).save(
            imgs_dir / f"{base}_NeuroSymbolicFSAD.jpg", quality=95
        )
        visualized += 1

    pixel_auc = None if missing_anomaly_masks else _histogram_auc(positive_hist, negative_hist)
    checkpoint = Path(entry["checkpoint"].format(shot=shot)).expanduser().resolve()
    rule_library = Path(metric["rule_library"]).expanduser().resolve()
    checkpoint_dir = destination / "checkpoint"
    checkpoint_link = checkpoint_dir / f"CLS-Seed_{seed}-{_safe_name(class_name)}-check_point.pth"
    rule_link = checkpoint_dir / f"RULE-Seed_{seed}-{_safe_name(class_name)}.json"
    if checkpoint.is_file():
        _ensure_symlink(checkpoint_link, checkpoint)
    if rule_library.is_file():
        _ensure_symlink(rule_link, rule_library)

    exported = {
        "dataset": dataset,
        "class_name": class_name,
        "shot": shot,
        "support_seed": seed,
        "i_roc": float(metric["baseline_plus_rules"]["i_auroc"]),
        "p_roc": pixel_auc,
        "pixel_metric_protocol": "streaming_65536_bin_roc_auc_on_inpformer_map"
        if pixel_auc is not None else "unavailable",
        "baseline_i_roc": float(metric["baseline"]["i_auroc"]),
        "delta_i_roc": float(metric["delta_i_auroc"]),
        "baseline_i_aupr": float(metric["baseline"]["i_aupr"]),
        "rules_i_aupr": float(metric["baseline_plus_rules"]["i_aupr"]),
        "delta_i_aupr": float(metric["delta_i_aupr"]),
        "selected_rule_count": int(metric["selected_rule_count"]),
        "fusion_enabled": bool(metric["fusion"]["enabled"]),
        "normal_trigger_rate": float(metric["normal_trigger_rate"]),
        "anomaly_trigger_rate": float(metric["anomaly_trigger_rate"]),
        "test_count": len(rows),
        "anomaly_masks_found": anomaly_mask_count,
        "anomaly_masks_missing": missing_anomaly_masks,
        "visualizations": visualized,
        "heatmap_source": "INP-Former pixel anomaly map; symbolic fusion is image-level",
        "metric_source": str(metric_path),
        "cache_source": str(cache_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
        "rule_library": str(rule_library),
        "visual_map_range": [map_min, map_max],
    }
    _dump(
        checkpoint_dir / f"Seed_{seed}-{_safe_name(class_name)}-manifest.json",
        exported,
    )
    return exported


def export_results(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve()
    registry = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root).expanduser().resolve()
    wanted = set(args.datasets or [])
    entries = [
        dict(entry) for entry in registry["datasets"]
        if entry.get("status") != "incompatible" and (not wanted or entry["id"] in wanted)
    ]
    report = {
        "schema_version": "1.0-h2vlr-layout-export",
        "shot": args.shot,
        "support_seed": args.support_seed,
        "method": "NeuroSymbolicFSAD",
        "datasets": [],
    }
    for entry in entries:
        class_names = _classes(entry)
        map_range = _dataset_map_range(output_root, entry["id"], class_names, args.shot)
        if map_range is None:
            report["datasets"].append({
                "dataset": entry["id"], "classes": [],
                "missing": "no prediction caches found",
            })
            continue
        destination = output_root / entry["id"] / f"k_{args.shot}"
        class_rows = [
            _export_class(
                entry, class_name, output_root, destination, args.shot,
                args.support_seed, args.visual_size, map_range,
                args.max_visualizations_per_class, args.histogram_bins,
            )
            for class_name in class_names
        ]
        complete = [row for row in class_rows if "i_roc" in row]
        csv_dir = destination / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        main_csv = csv_dir / f"Seed_{args.support_seed}-results.csv"
        with main_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "i_roc", "p_roc"])
            for row in complete:
                writer.writerow([
                    f"{entry['id']}-{row['class_name']}",
                    f"{100.0 * row['i_roc']:.2f}",
                    f"{100.0 * row['p_roc']:.2f}" if row["p_roc"] is not None else "nan",
                ])
        detailed_csv = csv_dir / f"Seed_{args.support_seed}-results-detailed.csv"
        if complete:
            fields = [
                "dataset", "class_name", "shot", "support_seed", "i_roc", "p_roc",
                "baseline_i_roc", "delta_i_roc", "baseline_i_aupr", "rules_i_aupr",
                "delta_i_aupr", "selected_rule_count", "fusion_enabled",
                "normal_trigger_rate", "anomaly_trigger_rate", "test_count",
                "anomaly_masks_found", "anomaly_masks_missing", "visualizations",
                "pixel_metric_protocol", "heatmap_source", "metric_source", "cache_source",
                "checkpoint", "checkpoint_sha256", "rule_library", "visual_map_range",
            ]
            with detailed_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(complete)
        dataset_row = {
            "dataset": entry["id"],
            "output": str(destination),
            "main_csv": str(main_csv),
            "detailed_csv": str(detailed_csv),
            "completed_classes": len(complete),
            "missing_classes": len(class_rows) - len(complete),
            "classes": class_rows,
        }
        _dump(destination / "export_manifest.json", dataset_row)
        report["datasets"].append(dataset_row)
    report_path = output_root / "summary" / f"h2vlr_export_k{args.shot}.json"
    _dump(report_path, report)
    report["report"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--support-seed", type=int, default=1)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--visual-size", type=int, default=400)
    parser.add_argument(
        "--max-visualizations-per-class", type=int, default=0,
        help="0 exports every test image.",
    )
    parser.add_argument("--histogram-bins", type=int, default=65536)
    return parser


def main() -> None:
    result = export_results(build_parser().parse_args())
    print(json.dumps({
        "report": result["report"],
        "datasets": len(result["datasets"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
