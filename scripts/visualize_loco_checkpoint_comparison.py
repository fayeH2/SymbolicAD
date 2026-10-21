#!/usr/bin/env python3
"""Create reproducible quantitative and qualitative MVTec LOCO comparisons.

The qualitative panel deliberately selects the median run-A score in each
normal/logical/structural family.  This avoids selecting examples by visual
appeal or by the size of the difference between two checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from PIL import Image


GROUPS = ("all", "logical", "structural")


def _load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pretty(name: str) -> str:
    return name.replace("_", " ").title()


def _plot_metrics(
    metrics_a: Mapping[str, object],
    metrics_b: Mapping[str, object],
    label_a: str,
    label_b: str,
    output: Path,
) -> None:
    classes = list(metrics_a["per_class"])
    if classes != list(metrics_b["per_class"]):
        raise ValueError("The two metric files contain different class orders")

    palette = {label_a: "#315A7D", label_b: "#D08A36"}
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 12), constrained_layout=True)
    x = np.arange(len(classes), dtype=float)
    width = 0.34
    for axis, group in zip(axes, GROUPS):
        values_a = np.asarray([
            metrics_a["per_class"][name][group]["image_auroc"]
            for name in classes
        ])
        values_b = np.asarray([
            metrics_b["per_class"][name][group]["image_auroc"]
            for name in classes
        ])
        axis.bar(
            x - width / 2, values_a, width, color=palette[label_a],
            edgecolor="#1F2933", linewidth=0.5, label=label_a,
        )
        axis.bar(
            x + width / 2, values_b, width, color=palette[label_b],
            edgecolor="#1F2933", linewidth=0.5, label=label_b,
        )
        for index, (left, right) in enumerate(zip(values_a, values_b)):
            delta = right - left
            axis.text(
                index, max(left, right) + 0.025, f"{delta:+.3f}",
                ha="center", va="bottom", fontsize=9,
                color="#8C4A13" if delta < 0 else "#1F4E6D",
            )
        macro_a = metrics_a["macro"][group]["image_auroc"]
        macro_b = metrics_b["macro"][group]["image_auroc"]
        axis.set_title(
            f"{_pretty(group)} image AUROC  |  macro "
            f"{label_a}={macro_a:.3f}, {label_b}={macro_b:.3f}, "
            f"delta={macro_b - macro_a:+.3f}",
            loc="left", fontsize=12, fontweight="bold",
        )
        axis.set_ylim(0.0, 1.08)
        axis.set_ylabel("AUROC")
        axis.set_xticks(x, [_pretty(name) for name in classes])
        axis.grid(axis="y", color="#D8DEE4", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncols=2, loc="upper right")
    fig.suptitle(
        "MVTec LOCO checkpoint comparison (K=4, one support seed)",
        fontsize=16, fontweight="bold",
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _load_cache(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "paths": np.asarray(payload["paths"]),
            "maps": np.asarray(payload["maps"], dtype=np.float32),
            "scores": np.asarray(payload["scores"], dtype=np.float32),
        }


def _family(path: str) -> str:
    defect = Path(path).parent.name.lower()
    if defect in {"good", "ok"}:
        return "good"
    if "logical" in defect:
        return "logical"
    if "structural" in defect:
        return "structural"
    return "other"


def _median_index(paths: Sequence[str], scores: np.ndarray, family: str) -> int:
    indices = np.asarray([
        index for index, path in enumerate(paths) if _family(str(path)) == family
    ], dtype=np.int64)
    if not len(indices):
        raise ValueError(f"No samples for family {family!r}")
    ordered = indices[np.argsort(scores[indices], kind="stable")]
    return int(ordered[(len(ordered) - 1) // 2])


def _model_frame(
    image: Image.Image,
    size: int,
    input_size: int = 448,
    crop_size: int = 392,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> np.ndarray:
    """Apply the baseline Resize -> CenterCrop -> map-resize geometry."""
    resized = image.resize((input_size, input_size), resample=resample)
    offset = (input_size - crop_size) // 2
    cropped = resized.crop((offset, offset, offset + crop_size, offset + crop_size))
    return np.asarray(cropped.resize((size, size), resample=resample))


def _read_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as handle:
        return _model_frame(handle.convert("RGB"), size)


def _resize_map(score_map: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(score_map, (size, size), interpolation=cv2.INTER_LINEAR)


def _mask_for_image(image_path: Path, size: int) -> np.ndarray:
    if image_path.parent.name.lower() in {"good", "ok"}:
        return np.zeros((size, size), dtype=np.uint8)
    class_root = image_path.parents[2]
    candidate = (
        class_root / "ground_truth" / image_path.parent.name
        / f"{image_path.stem}_mask.png"
    )
    if not candidate.is_file():
        return np.zeros((size, size), dtype=np.uint8)
    with Image.open(candidate) as handle:
        return _model_frame(
            handle.convert("L"), size, resample=Image.Resampling.NEAREST
        )


def _outlined(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    contours = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )[0]
    cv2.drawContours(result, contours, -1, (228, 55, 55), 2)
    return result


def _overlay(
    image: np.ndarray, score_map: np.ndarray, maximum: float,
) -> np.ndarray:
    normalized = np.clip(score_map / max(maximum, 1e-12), 0.0, 1.0)
    heat = plt.get_cmap("magma")(normalized)[..., :3]
    mixed = 0.55 * (image.astype(np.float32) / 255.0) + 0.45 * heat
    return np.clip(mixed, 0.0, 1.0)


def _plot_examples(
    run_a: Path,
    run_b: Path,
    label_a: str,
    label_b: str,
    classes: Iterable[str],
    output: Path,
    manifest_path: Path,
    size: int = 256,
) -> None:
    rows = []
    pooled_maps = []
    caches = {}
    for class_name in classes:
        cache_a = _load_cache(run_a / "cache" / f"{class_name}.npz")
        cache_b = _load_cache(run_b / "cache" / f"{class_name}.npz")
        if not np.array_equal(cache_a["paths"], cache_b["paths"]):
            raise ValueError(f"Cache path mismatch for {class_name}")
        caches[class_name] = (cache_a, cache_b)
        pooled_maps.extend([cache_a["maps"], cache_b["maps"]])
        for family in ("good", "logical", "structural"):
            index = _median_index(cache_a["paths"], cache_a["scores"], family)
            rows.append((class_name, family, index))
    common_maximum = float(np.quantile(
        np.concatenate([item.reshape(-1) for item in pooled_maps]), 0.995
    ))
    pooled_differences = []
    for class_name in classes:
        cache_a, cache_b = caches[class_name]
        pooled_differences.append(cache_b["maps"] - cache_a["maps"])
    difference_limit = float(np.quantile(
        np.abs(np.concatenate([item.reshape(-1) for item in pooled_differences])),
        0.995,
    ))
    difference_limit = max(difference_limit, 1e-6)

    fig, axes = plt.subplots(
        len(rows), 5, figsize=(15, 3.0 * len(rows)), constrained_layout=True,
    )
    manifest = {
        "selection": "median run-A image score within each class/family",
        "heatmap_scale": {"minimum": 0.0, "maximum_pooled_q995": common_maximum},
        "difference_scale": {
            "minimum": -difference_limit,
            "maximum": difference_limit,
        },
        "geometry": "Resize(448,448) -> CenterCrop(392,392) -> Resize(256,256)",
        "rows": [],
    }
    for row_index, (class_name, family, index) in enumerate(rows):
        cache_a, cache_b = caches[class_name]
        image_path = Path(str(cache_a["paths"][index]))
        image = _read_rgb(image_path, size)
        mask = _mask_for_image(image_path, size)
        map_a = _resize_map(cache_a["maps"][index], size)
        map_b = _resize_map(cache_b["maps"][index], size)
        panels = (
            image,
            _outlined(image, mask),
            _overlay(image, map_a, common_maximum),
            _overlay(image, map_b, common_maximum),
            map_b - map_a,
        )
        for column, panel in enumerate(panels):
            if column == 4:
                axes[row_index, column].imshow(
                    panel, cmap="RdBu_r", vmin=-difference_limit,
                    vmax=difference_limit,
                )
            else:
                axes[row_index, column].imshow(panel)
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            for spine in axes[row_index, column].spines.values():
                spine.set_visible(False)
        axes[row_index, 0].set_ylabel(
            f"{_pretty(class_name)}\n{family}\n{image_path.stem}",
            rotation=0, ha="right", va="center", fontsize=10,
        )
        axes[row_index, 2].set_xlabel(
            f"score {cache_a['scores'][index]:.3f}", fontsize=9,
        )
        axes[row_index, 3].set_xlabel(
            f"score {cache_b['scores'][index]:.3f}", fontsize=9,
        )
        manifest["rows"].append({
            "class_name": class_name,
            "family": family,
            "path": str(image_path),
            "index": index,
            "score_a": float(cache_a["scores"][index]),
            "score_b": float(cache_b["scores"][index]),
            "mask_nonzero": int(np.count_nonzero(mask)),
        })
    titles = (
        "Original", "Pixel GT outline", f"{label_a} overlay",
        f"{label_b} overlay", f"Delta: {label_b} - {label_a}",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=12, fontweight="bold")
    heat_scalar = plt.cm.ScalarMappable(
        norm=colors.Normalize(vmin=0.0, vmax=common_maximum), cmap="magma"
    )
    diff_scalar = plt.cm.ScalarMappable(
        norm=colors.Normalize(vmin=-difference_limit, vmax=difference_limit),
        cmap="RdBu_r",
    )
    fig.colorbar(
        heat_scalar, ax=axes[:, 2:4], location="bottom", shrink=0.45,
        label="Raw anomaly-map value; common pooled 0–99.5% scale",
    )
    fig.colorbar(
        diff_scalar, ax=axes[:, 4], location="bottom", shrink=0.75,
        label="Raw anomaly-map difference",
    )
    fig.suptitle(
        "Representative LOCO anomaly-map comparison",
        fontsize=16, fontweight="bold", y=1.006,
    )
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True, type=Path)
    parser.add_argument("--run-b", required=True, type=Path)
    parser.add_argument("--label-a", default="Official22")
    parser.add_argument("--label-b", default="LOCO-safe22")
    parser.add_argument("--classes", nargs="+", default=["pushpins", "screw_bag"])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_a = _load_json(args.run_a / "metrics.json")
    metrics_b = _load_json(args.run_b / "metrics.json")
    _plot_metrics(
        metrics_a, metrics_b, args.label_a, args.label_b,
        args.output_dir / "auroc_comparison.png",
    )
    _plot_examples(
        args.run_a, args.run_b, args.label_a, args.label_b, args.classes,
        args.output_dir / "representative_heatmaps.png",
        args.output_dir / "representative_heatmaps_manifest.json",
    )
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
