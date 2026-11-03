#!/usr/bin/env python3
"""Compare INP score distributions before/after removing a logical anomaly.

The script takes one MVTec LOCO logical-anomaly image, removes the pixels
covered by its ground-truth anomaly mask with OpenCV inpainting, runs the same
INP-Former baseline on both images, and plots the resulting anomaly-score
distributions against a normal reference distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ks_2samp, kurtosis, norm, skew

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from symbolicAD.evaluation.spro import (
    MapFrame, ScoreMapGeometry, project_score_map_to_original,
)
from symbolicAD.inpformer_adapter import INPFormerAdapter


DATA_ROOT = Path("/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO")
DEFAULT_CHECKPOINT = (
    PROJECT / "artifacts/baselines/mvtec_loco/breakfast_box"
    "/k4_seed1_official22/model.pth"
)
DEFAULT_NORMAL_CACHE = (
    PROJECT / "results/inp_baseline/breakfast_box"
    "/k4_seed1_official22/cache/breakfast_box.npz"
)
COLORS = {
    "normal": "#6B7C8C",
    "original": "#2F6F9F",
    "removed": "#C2643A",
    "inside": "#B34A43",
    "outside": "#4E8A6B",
}


def _load_pair(
    class_root: Path, sample: str, defect_dir: str
) -> Tuple[np.ndarray, np.ndarray, Path, Path]:
    image_path = class_root / "test" / defect_dir / f"{sample}.png"
    mask_path = class_root / "ground_truth" / defect_dir / f"{sample}_mask.png"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path).convert("L"))
    if rgb.shape[:2] != mask.shape[:2]:
        raise ValueError("image and ground-truth mask shapes differ")
    return rgb, mask, image_path, mask_path


def _remove_region(
    rgb: np.ndarray, mask: np.ndarray, *, dilation: int, radius: int
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if dilation > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1)
        )
        binary = cv2.dilate(binary, kernel)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, binary, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)


def _mask_to_model_frame(mask: np.ndarray, size: int, input_size: int, crop_size: int) -> np.ndarray:
    square = np.asarray(
        Image.fromarray(mask).resize((input_size, input_size), Image.NEAREST)
    )
    offset = (input_size - crop_size) // 2
    crop = square[offset:offset + crop_size, offset:offset + crop_size]
    resized = np.asarray(
        Image.fromarray(crop).resize((size, size), Image.NEAREST)
    )
    return resized > 0


def _density(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    histogram, _ = np.histogram(values, bins=bins, density=True)
    width = float(bins[1] - bins[0])
    smoothed = gaussian_filter1d(histogram.astype(np.float64), sigma=2.0)
    return smoothed / max(float(smoothed.sum() * width), 1e-12)


def _reference_band(
    maps: Sequence[np.ndarray], bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = np.stack([_density(np.asarray(m).reshape(-1), bins) for m in maps])
    return (
        curves.mean(axis=0),
        np.percentile(curves, 5, axis=0),
        np.percentile(curves, 95, axis=0),
    )


def _cdf(curve: np.ndarray, bins: np.ndarray) -> np.ndarray:
    width = float(bins[1] - bins[0])
    cumulative = np.cumsum(curve) * width
    return np.clip(cumulative / max(float(cumulative[-1]), 1e-12), 0.0, 1.0)


def _outline(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = rgb.copy()
    contours = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    cv2.drawContours(output, contours, -1, (225, 55, 50), 4)
    return output


def _overlay(rgb: np.ndarray, values: np.ndarray, maximum: float) -> np.ndarray:
    normalized = np.clip(values / max(maximum, 1e-12), 0.0, 1.0)
    heat = plt.get_cmap("magma")(normalized)[..., :3]
    return np.clip(
        0.56 * rgb.astype(np.float32) / 255.0 + 0.44 * heat, 0.0, 1.0
    )


def _describe(values: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(flat.max()),
    }


def _cdf_panel(
    axis, bins: np.ndarray, curves: Mapping[str, np.ndarray],
    band: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    centres = 0.5 * (bins[:-1] + bins[1:])
    normal_mean, normal_low, normal_high = band
    axis.fill_between(
        centres, _cdf(normal_low, bins), _cdf(normal_high, bins),
        color=COLORS["normal"], alpha=0.18, linewidth=0,
        label="normal 5-95% band",
    )
    axis.plot(
        centres, _cdf(normal_mean, bins), color=COLORS["normal"],
        linewidth=1.4, linestyle="--", label="normal mean",
    )
    for name, curve in curves.items():
        axis.plot(
            centres, _cdf(curve, bins), color=COLORS[name], linewidth=2.0,
            label=name,
        )
    axis.set_xlim(float(bins[0]), float(bins[-1]))
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("INP anomaly score")
    axis.set_ylabel("ECDF")
    axis.grid(color="#D7DEE3", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8, loc="lower right")


def _gaussian_summary(values: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(ddof=0)),
        "skewness": float(skew(flat)),
        "excess_kurtosis": float(kurtosis(flat)),
    }


def _save_gaussian_figure(
    original_map: np.ndarray,
    removed_map: np.ndarray,
    normal_maps: np.ndarray,
    target: Path,
) -> Dict[str, Dict[str, float]]:
    grouped = {
        "normal": normal_maps.reshape(-1),
        "original": original_map.reshape(-1),
        "removed": removed_map.reshape(-1),
    }
    summaries = {name: _gaussian_summary(values) for name, values in grouped.items()}
    upper = max(
        summary["mean"] + 4.0 * summary["std"]
        for summary in summaries.values()
    )
    x_values = np.linspace(0.0, upper, 800)

    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), constrained_layout=True)
    for name in ("normal", "original", "removed"):
        summary = summaries[name]
        curve = norm.pdf(x_values, summary["mean"], summary["std"])
        axes[0].plot(
            x_values, curve, color=COLORS[name], linewidth=2.2,
            label=(
                f"{name} N(mu={summary['mean']:.3f}, "
                f"sigma={summary['std']:.3f})"
            ),
        )
        axes[0].axvline(
            summary["mean"], color=COLORS[name], linewidth=1.0, linestyle=":"
        )
        inside = (
            (x_values >= summary["mean"] - summary["std"])
            & (x_values <= summary["mean"] + summary["std"])
        )
        axes[0].fill_between(
            x_values[inside], curve[inside], color=COLORS[name], alpha=0.08,
        )
    axes[0].set_xlim(0.0, upper)
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_xlabel("INP anomaly score")
    axes[0].set_ylabel("Gaussian density")
    axes[0].set_title("Fitted normal distributions", loc="left", weight="bold")
    axes[0].grid(color="#D7DEE3", linewidth=0.7)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)

    bins = np.linspace(0.0, upper, 140)
    centres = 0.5 * (bins[:-1] + bins[1:])
    for name in ("original", "removed"):
        summary = summaries[name]
        histogram, _ = np.histogram(grouped[name], bins=bins, density=True)
        axes[1].step(
            centres, histogram, where="mid", color=COLORS[name],
            linewidth=1.1, alpha=0.55, label=f"{name} empirical",
        )
        axes[1].plot(
            x_values, norm.pdf(x_values, summary["mean"], summary["std"]),
            color=COLORS[name], linewidth=2.0, label=f"{name} Gaussian fit",
        )
    axes[1].set_xlim(0.0, upper)
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_xlabel("INP anomaly score")
    axes[1].set_ylabel("density")
    axes[1].set_title("Empirical vs Gaussian fit", loc="left", weight="bold")
    axes[1].grid(color="#D7DEE3", linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8)

    text = "\n".join(
        f"{name}: mu={summary['mean']:.3f}, sigma={summary['std']:.3f}, "
        f"skew={summary['skewness']:.2f}, ex-kurt={summary['excess_kurtosis']:.2f}"
        for name, summary in summaries.items()
    )
    figure.text(0.5, -0.02, text, ha="center", va="top", fontsize=8.5)
    figure.savefig(target, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return summaries


def _save_pixel_intensity_figure(
    original: np.ndarray, removed: np.ndarray, target: Path
) -> Dict[str, float]:
    def luminance(rgb: np.ndarray) -> np.ndarray:
        values = rgb.astype(np.float64) / 255.0
        return (
            0.2126 * values[..., 0]
            + 0.7152 * values[..., 1]
            + 0.0722 * values[..., 2]
        )

    left = luminance(original).reshape(-1)
    right = luminance(removed).reshape(-1)
    bins = np.linspace(0.0, 1.0, 256)
    original_curve = _density(left, bins)
    removed_curve = _density(right, bins)
    statistics = ks_2samp(left, right)
    centres = 0.5 * (bins[:-1] + bins[1:])

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), constrained_layout=True)
    axes[0].plot(centres, original_curve, color=COLORS["original"], linewidth=2.0, label="original")
    axes[0].plot(centres, removed_curve, color=COLORS["removed"], linewidth=2.0, label="removed")
    axes[0].set_xlabel("luminance")
    axes[0].set_ylabel("density")
    axes[0].set_title("Pixel luminance density", loc="left", weight="bold")
    axes[0].legend(frameon=False)
    axes[1].plot(centres, _cdf(original_curve, bins), color=COLORS["original"], linewidth=2.0, label="original")
    axes[1].plot(centres, _cdf(removed_curve, bins), color=COLORS["removed"], linewidth=2.0, label="removed")
    axes[1].set_xlabel("luminance")
    axes[1].set_ylabel("ECDF")
    axes[1].set_title(
        f"Pixel luminance ECDF (KS={statistics.statistic:.4f})",
        loc="left", weight="bold",
    )
    for axis in axes:
        axis.grid(color="#D7DEE3", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(target, dpi=190, facecolor="white")
    plt.close(figure)
    return {
        "ks_statistic": float(statistics.statistic),
        "ks_pvalue": float(statistics.pvalue),
        "original_mean_luminance": float(left.mean()),
        "removed_mean_luminance": float(right.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-name", default="breakfast_box")
    parser.add_argument("--sample", default="073")
    parser.add_argument("--defect-dir", default="logical_anomalies")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--normal-cache", type=Path, default=DEFAULT_NORMAL_CACHE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dilation", type=int, default=3)
    parser.add_argument("--radius", type=int, default=7)
    parser.add_argument("--bins", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    class_root = args.data_root / args.class_name
    rgb, mask, image_path, mask_path = _load_pair(
        class_root, args.sample, args.defect_dir
    )
    removed_rgb = _remove_region(
        rgb, mask, dilation=args.dilation, radius=args.radius
    )

    adapter = INPFormerAdapter(args.checkpoint, args.device)
    input_size = adapter.input_size
    crop_size = adapter.crop_size
    predictions = adapter.infer_images(
        [rgb, removed_rgb],
        identifiers=[str(image_path), str(image_path) + "::removed"],
    )
    adapter.close()
    original_map = np.asarray(predictions["maps"][0], dtype=np.float32)
    removed_map = np.asarray(predictions["maps"][1], dtype=np.float32)
    original_score = float(predictions["scores"][0])
    removed_score = float(predictions["scores"][1])

    with np.load(args.normal_cache, allow_pickle=False) as cache:
        paths = [str(value) for value in cache["paths"]]
        maps = np.asarray(cache["maps"], dtype=np.float32)
        scores = np.asarray(cache["scores"], dtype=np.float64)
    normal_indices = [
        index for index, path in enumerate(paths) if "/test/good/" in path
    ]
    if not normal_indices:
        raise ValueError("normal cache contains no test/good images")
    normal_maps = maps[normal_indices]
    normal_scores = scores[normal_indices]

    gt_map = _mask_to_model_frame(
        mask, original_map.shape[0], input_size, crop_size
    )
    if not np.any(gt_map):
        raise ValueError("ground-truth mask is empty in the model crop frame")
    inside = original_map[gt_map]
    outside = original_map[~gt_map]
    energy_fraction = float(
        inside.sum() / max(float(original_map.sum()), 1e-12)
    )

    x_min = 0.0
    x_max = float(max(0.65, removed_map.max() * 1.05, normal_maps.max() * 1.05))
    bins = np.linspace(x_min, x_max, args.bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])

    original_curve = _density(original_map.reshape(-1), bins)
    removed_curve = _density(removed_map.reshape(-1), bins)
    normal_mean, normal_low, normal_high = _reference_band(normal_maps, bins)
    inside_curve = _density(inside, bins)
    outside_curve = _density(outside, bins)

    rng = np.random.default_rng(1307)
    normal_sample = rng.choice(
        normal_maps.reshape(-1), size=min(300_000, normal_maps.size), replace=False
    )
    ks_original_removed = ks_2samp(original_map.reshape(-1), removed_map.reshape(-1))
    ks_original_normal = ks_2samp(original_map.reshape(-1), normal_sample)
    ks_removed_normal = ks_2samp(removed_map.reshape(-1), normal_sample)

    normal_p95 = float(np.percentile(normal_maps, 95))
    normal_p99 = float(np.percentile(normal_maps, 99))
    tail_fractions = {
        "normal_p95": normal_p95,
        "normal_p99": normal_p99,
        "original_above_normal_p95": float((original_map > normal_p95).mean()),
        "removed_above_normal_p95": float((removed_map > normal_p95).mean()),
        "original_above_normal_p99": float((original_map > normal_p99).mean()),
        "removed_above_normal_p99": float((removed_map > normal_p99).mean()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    removed_path = args.output_dir / f"{args.sample}_anomaly_removed.png"
    Image.fromarray(removed_rgb).save(removed_path)
    np.savez_compressed(
        args.output_dir / "score_maps.npz",
        original=original_map,
        removed=removed_map,
        ground_truth_model_frame=gt_map.astype(np.uint8),
        bins=bins,
    )

    heat_max = float(max(
        np.percentile(original_map, 99.5),
        np.percentile(removed_map, 99.5),
    ))
    geometry = ScoreMapGeometry(
        MapFrame.MODEL_CENTER_CROP,
        input_size=input_size, crop_size=crop_size,
    )
    original_heat = project_score_map_to_original(
        original_map, rgb.shape[:2], geometry
    )
    removed_heat = project_score_map_to_original(
        removed_map, rgb.shape[:2], geometry
    )

    figure = plt.figure(figsize=(15.6, 8.6), constrained_layout=True)
    grid = figure.add_gridspec(2, 12, height_ratios=(1.0, 1.15))
    image_axes = [
        figure.add_subplot(grid[0, 0:3]),
        figure.add_subplot(grid[0, 3:6]),
        figure.add_subplot(grid[0, 6:9]),
        figure.add_subplot(grid[0, 9:12]),
    ]
    panels = (
        _outline(rgb, mask),
        removed_rgb,
        _overlay(rgb, original_heat, heat_max),
        _overlay(removed_rgb, removed_heat, heat_max),
    )
    titles = (
        "Original + GT outline",
        "Anomaly region removed (inpaint)",
        "Original INP score",
        "Removed INP score",
    )
    for axis, panel, title in zip(image_axes, panels, titles):
        axis.imshow(panel)
        axis.set_title(title, weight="bold")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[:].set_visible(False)

    density_axis = figure.add_subplot(grid[1, 0:4])
    density_axis.fill_between(
        centres, normal_low, normal_high, color=COLORS["normal"],
        alpha=0.18, linewidth=0, label="normal 5-95% band",
    )
    density_axis.plot(
        centres, normal_mean, color=COLORS["normal"], linewidth=1.4,
        linestyle="--", label="normal mean",
    )
    density_axis.plot(
        centres, original_curve, color=COLORS["original"], linewidth=2.2,
        label=f"original (mean={original_map.mean():.3f})",
    )
    density_axis.plot(
        centres, removed_curve, color=COLORS["removed"], linewidth=2.2,
        label=f"removed (mean={removed_map.mean():.3f})",
    )
    for value, colour in (
        (normal_p95, COLORS["normal"]),
        (np.percentile(original_map, 95), COLORS["original"]),
        (np.percentile(removed_map, 95), COLORS["removed"]),
    ):
        density_axis.axvline(value, color=colour, linewidth=1.0, linestyle=":")
    density_axis.text(
        0.985, 0.96,
        f"p95: normal {normal_p95:.3f} | original "
        f"{np.percentile(original_map, 95):.3f} | removed "
        f"{np.percentile(removed_map, 95):.3f}\n"
        f"> normal p99: original "
        f"{tail_fractions['original_above_normal_p99']:.1%} | removed "
        f"{tail_fractions['removed_above_normal_p99']:.1%}",
        transform=density_axis.transAxes, ha="right", va="top",
        fontsize=7.5, color="#27323A",
    )
    density_axis.set_xlim(x_min, x_max)
    density_axis.set_ylim(bottom=0.0)
    density_axis.set_xlabel("INP anomaly score")
    density_axis.set_ylabel("density")
    density_axis.set_title("Anomaly-score density", loc="left", weight="bold")
    density_axis.grid(color="#D7DEE3", linewidth=0.7)
    density_axis.spines[["top", "right"]].set_visible(False)
    density_axis.legend(frameon=False, fontsize=8, loc="center right")

    cdf_axis = figure.add_subplot(grid[1, 4:8])
    _cdf_panel(
        cdf_axis, bins,
        {"original": original_curve, "removed": removed_curve},
        (normal_mean, normal_low, normal_high),
    )
    cdf_axis.set_title("Anomaly-score ECDF", loc="left", weight="bold")

    region_axis = figure.add_subplot(grid[1, 8:12])
    region_axis.fill_between(
        centres, normal_low, normal_high, color=COLORS["normal"],
        alpha=0.14, linewidth=0, label="normal 5-95% band",
    )
    region_axis.plot(
        centres, normal_mean, color=COLORS["normal"], linewidth=1.2,
        linestyle="--", label="normal mean",
    )
    region_axis.plot(
        centres, inside_curve, color=COLORS["inside"], linewidth=2.2,
        label=f"GT region (mean={inside.mean():.3f})",
    )
    region_axis.plot(
        centres, outside_curve, color=COLORS["outside"], linewidth=2.2,
        label=f"outside GT (mean={outside.mean():.3f})",
    )
    region_axis.set_xlim(x_min, x_max)
    region_axis.set_ylim(bottom=0.0)
    region_axis.set_xlabel("INP anomaly score")
    region_axis.set_ylabel("density")
    region_axis.set_title(
        f"Original map: GT vs rest (GT energy={energy_fraction:.1%})",
        loc="left", weight="bold",
    )
    region_axis.grid(color="#D7DEE3", linewidth=0.7)
    region_axis.spines[["top", "right"]].set_visible(False)
    region_axis.legend(frameon=False, fontsize=8)

    figure.suptitle(
        f"{args.class_name} logical anomaly {args.sample}: "
        "does removing the GT region normalise the score distribution?",
        weight="bold",
    )
    figure_path = args.output_dir / "anomaly_score_distribution.png"
    figure.savefig(figure_path, dpi=190, facecolor="white")
    plt.close(figure)

    gaussian_path = args.output_dir / "anomaly_score_gaussian_fit.png"
    gaussian_stats = _save_gaussian_figure(
        original_map, removed_map, normal_maps, gaussian_path
    )
    pixel_stats = _save_pixel_intensity_figure(
        rgb, removed_rgb, args.output_dir / "pixel_intensity_distribution.png"
    )

    percentile = float(np.mean(normal_scores < original_score))
    report = {
        "schema": "loco-logical-distribution-analysis-v1",
        "dataset": "mvtec_loco",
        "class_name": args.class_name,
        "sample": args.sample,
        "defect_dir": args.defect_dir,
        "image": str(image_path),
        "ground_truth_mask": str(mask_path),
        "checkpoint": str(args.checkpoint),
        "normal_reference": {
            "cache": str(args.normal_cache),
            "images": len(normal_indices),
            "score_mean": float(normal_scores.mean()),
            "score_p50": float(np.median(normal_scores)),
            "score_p95": float(np.percentile(normal_scores, 95)),
            "map": _describe(normal_maps.reshape(-1)),
        },
        "removal": {
            "method": "OpenCV TELEA inpainting",
            "dilation_px": args.dilation,
            "radius_px": args.radius,
            "mask_area_fraction": float((mask > 0).mean()),
            "changed_pixel_fraction": float(np.any(removed_rgb != rgb, axis=2).mean()),
            "output_image": str(removed_path),
        },
        "image_scores": {
            "original": original_score,
            "removed": removed_score,
            "original_percentile_in_normal": percentile,
        },
        "map_stats": {
            "original": _describe(original_map),
            "removed": _describe(removed_map),
            "inside_gt": _describe(inside),
            "outside_gt": _describe(outside),
            "gt_energy_fraction": energy_fraction,
        },
        "distribution_distance": {
            "original_vs_removed": {
                "ks_statistic": float(ks_original_removed.statistic),
                "ks_pvalue": float(ks_original_removed.pvalue),
            },
            "original_vs_normal": {
                "ks_statistic": float(ks_original_normal.statistic),
                "ks_pvalue": float(ks_original_normal.pvalue),
            },
            "removed_vs_normal": {
                "ks_statistic": float(ks_removed_normal.statistic),
                "ks_pvalue": float(ks_removed_normal.pvalue),
            },
        },
        "tail_fractions": tail_fractions,
        "gaussian_fit": gaussian_stats,
        "score_maps": str(args.output_dir / "score_maps.npz"),
        "pixel_intensity": pixel_stats,
        "figures": {
            "anomaly_score_distribution": str(figure_path),
            "anomaly_score_gaussian_fit": str(gaussian_path),
            "pixel_intensity_distribution": str(
                args.output_dir / "pixel_intensity_distribution.png"
            ),
        },
    }
    report_path = args.output_dir / "analysis.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "figures": report["figures"],
        "report": str(report_path),
        "original_score": original_score,
        "removed_score": removed_score,
        "original_percentile_in_normal": percentile,
        "gt_energy_fraction": energy_fraction,
        "tail_fractions": tail_fractions,
        "gaussian_fit": gaussian_stats,
        "ks_original_vs_removed": report["distribution_distance"]["original_vs_removed"],
        "ks_original_vs_normal": report["distribution_distance"]["original_vs_normal"],
        "ks_removed_vs_normal": report["distribution_distance"]["removed_vs_normal"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
