#!/usr/bin/env python3
"""Render auditable baseline/Qwen/evolved comparisons for one LOCO class."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from symbolicAD.evaluation.spro import (
    MapFrame, ScoreMapGeometry, project_score_map_to_original,
)


GROUPS = ("all", "logical", "structural")
PALETTE = {
    "INP baseline": "#356A8A",
    "Raw Qwen": "#B67A2A",
    "Greedy": "#7A5A9C",
    "Evolved": "#3F7D5A",
}


def _json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_figure(
    baseline: Mapping[str, Any], rules: Mapping[str, Any], category: str, target: Path
) -> None:
    series = {
        "INP baseline": [
            baseline["per_class"][category][group]["image_auroc"] for group in GROUPS
        ],
        "Raw Qwen": [
            rules["rule_sets"]["raw_qwen_executable"]["metrics"][group]["image_auroc"]
            for group in GROUPS
        ],
        "Greedy": [
            rules["rule_sets"]["greedy"]["metrics"][group]["image_auroc"]
            for group in GROUPS
        ],
        "Evolved": [
            rules["rule_sets"]["evolved"]["metrics"][group]["image_auroc"]
            for group in GROUPS
        ],
    }
    fig, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    x = np.arange(len(GROUPS), dtype=float)
    width = 0.19
    offsets = (-1.5 * width, -.5 * width, .5 * width, 1.5 * width)
    for offset, (name, values) in zip(offsets, series.items()):
        bars = axis.bar(
            x + offset, values, width, label=name, color=PALETTE[name],
            edgecolor="#27323A", linewidth=0.5,
        )
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values],
                       padding=3, fontsize=9)
    axis.axhline(0.80, color="#A33E3E", linestyle="--", linewidth=1.2,
                 label="0.80 image-AUROC gate")
    axis.set_ylim(0.0, 1.08)
    axis.set_xticks(x, [name.title() for name in GROUPS])
    axis.set_ylabel("Image AUROC")
    axis.set_title(f"{category}: held-out image detection", loc="left", weight="bold")
    axis.grid(axis="y", color="#D7DEE3", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=2, loc="lower right")
    fig.savefig(target, dpi=190, facecolor="white")
    plt.close(fig)


def _pareto_figure(verification: Mapping[str, Any], target: Path) -> None:
    search = verification["search"]
    oracle = search["oracle"]["pareto_front"] if search.get("oracle") else []
    evolved = search["evolved"]["pareto_front"] if search.get("evolved") else []
    fig, axis = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    if oracle:
        axis.scatter(
            [row["L_normal"] for row in oracle], [row["L_cf"] for row in oracle],
            s=48, marker="o", facecolor="none", edgecolor="#356A8A",
            linewidth=1.3, label="Exhaustive Pareto front",
        )
    if evolved:
        axis.scatter(
            [row["L_normal"] for row in evolved], [row["L_cf"] for row in evolved],
            s=28, marker="x", color="#B34A43", linewidth=1.5,
            label="GA-discovered front",
        )
    selected = search.get("evolved", {}).get("selected")
    greedy = search.get("greedy_control", {}).get("selected")
    if greedy:
        axis.scatter(
            [greedy["L_normal"]], [greedy["L_cf"]], marker="D", s=70,
            color=PALETTE["Greedy"], edgecolor="#30263A", linewidth=.7,
            label="Greedy selected", zorder=4,
        )
    if selected:
        axis.scatter(
            [selected["L_normal"]], [selected["L_cf"]], marker="*", s=180,
            color="#3F7D5A", edgecolor="#243028", linewidth=0.7,
            label="Selected rule set", zorder=5,
        )
    axis.set_xlabel(r"$L_{normal}$ (minimize)")
    axis.set_ylabel(r"$L_{cf}$ (minimize)")
    axis.set_xlim(left=min(-0.02, axis.get_xlim()[0]))
    axis.set_ylim(bottom=min(-0.02, axis.get_ylim()[0]))
    axis.grid(color="#D7DEE3", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("Rule-set search objectives", loc="left", weight="bold")
    axis.legend(frameon=False)
    fig.savefig(target, dpi=190, facecolor="white")
    plt.close(fig)


def _spro_figure(
    baseline_curve_path: Path,
    rule_curve_path: Path,
    category: str,
    target: Path,
) -> None:
    with np.load(baseline_curve_path) as baseline, np.load(rule_curve_path) as rules:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
        for axis, family in zip(axes, ("logical", "structural")):
            baseline_fpr = baseline[f"{category}__{family}__fpr"]
            baseline_spro = baseline[f"{category}__{family}__spro"]
            rule_fpr = rules[f"evolved__logic_total__{family}__fpr"]
            rule_spro = rules[f"evolved__logic_total__{family}__spro"]
            greedy_fpr = rules[f"greedy__logic_total__{family}__fpr"]
            greedy_spro = rules[f"greedy__logic_total__{family}__spro"]
            axis.plot(baseline_fpr, baseline_spro, color=PALETTE["INP baseline"],
                      linewidth=2.0, label="INP baseline")
            axis.plot(rule_fpr, rule_spro, color=PALETTE["Evolved"],
                      linewidth=2.0, label="Evolved logic")
            axis.plot(greedy_fpr, greedy_spro, color=PALETTE["Greedy"],
                      linewidth=1.7, linestyle="--", label="Greedy logic")
            axis.set_xlim(0.0, 0.05)
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel("False-positive rate")
            axis.set_ylabel("Saturated PRO")
            axis.set_title(f"{family.title()} defects", loc="left", weight="bold")
            axis.grid(color="#D7DEE3", linewidth=0.7)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False)
        fig.suptitle(f"{category}: localization up to 5% FPR", weight="bold")
        fig.savefig(target, dpi=190, facecolor="white")
        plt.close(fig)


def _family(path: str) -> str:
    name = Path(path).parent.name.lower()
    if name == "good":
        return "good"
    if "logical" in name:
        return "logical"
    return "structural"


def _gt(path: Path) -> np.ndarray:
    if path.parent.name == "good":
        with Image.open(path) as image:
            return np.zeros((image.height, image.width), dtype=np.uint8)
    mask = path.parents[2] / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
    return np.asarray(Image.open(mask).convert("L"))


def _overlay(rgb: np.ndarray, values: np.ndarray, maximum: float) -> np.ndarray:
    normalized = np.clip(values / max(maximum, 1e-12), 0.0, 1.0)
    heat = plt.get_cmap("magma")(normalized)[..., :3]
    return np.clip(0.56 * rgb.astype(np.float32) / 255.0 + 0.44 * heat, 0.0, 1.0)


def _outline(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = rgb.copy()
    contours = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(output, contours, -1, (225, 55, 50), 4)
    return output


def _qualitative_figure(
    baseline_cache_path: Path,
    rule_maps_path: Path,
    target: Path,
    manifest_path: Path,
) -> None:
    with np.load(baseline_cache_path, allow_pickle=False) as base, np.load(
        rule_maps_path, allow_pickle=False
    ) as rule:
        base_paths = [str(value) for value in base["paths"]]
        rule_paths = [str(value) for value in rule["paths"]]
        if base_paths != rule_paths:
            raise ValueError("baseline and rule map paths are not aligned")
        baseline_maps = np.asarray(base["maps"], dtype=np.float32)
        baseline_scores = np.asarray(base["scores"], dtype=np.float32)
        raw_maps = np.asarray(rule["raw_qwen_executable__logic_total"], dtype=np.float32)
        greedy_maps = np.asarray(rule["greedy__logic_total"], dtype=np.float32)
        evolved_maps = np.asarray(rule["evolved__logic_total"], dtype=np.float32)
        selected = []
        for family in ("good", "logical", "structural"):
            indices = np.asarray([
                index for index, path in enumerate(base_paths) if _family(path) == family
            ])
            order = indices[np.argsort(baseline_scores[indices], kind="stable")]
            selected.append(int(order[(len(order) - 1) // 2]))
        baseline_max = float(np.quantile(baseline_maps, 0.995))
        logic_max = 1.0
        fig, axes = plt.subplots(3, 5, figsize=(16.2, 9.5), constrained_layout=True)
        manifest = {
            "selection": "median INP image score within good/logical/structural",
            "baseline_heat_scale": [0.0, baseline_max],
            "logic_heat_scale": [0.0, logic_max],
            "map_frames": {
                "baseline": "model_center_crop",
                "rules": "full_image_normalized",
            },
            "rows": [],
        }
        for row, index in enumerate(selected):
            path = Path(base_paths[index])
            rgb = np.asarray(Image.open(path).convert("RGB"))
            gt = _gt(path)
            shape = rgb.shape[:2]
            baseline = project_score_map_to_original(
                baseline_maps[index], shape,
                ScoreMapGeometry(MapFrame.MODEL_CENTER_CROP, 448, 392),
            )
            raw = project_score_map_to_original(
                raw_maps[index], shape,
                ScoreMapGeometry(MapFrame.FULL_IMAGE_NORMALIZED),
            )
            greedy = project_score_map_to_original(
                greedy_maps[index], shape,
                ScoreMapGeometry(MapFrame.FULL_IMAGE_NORMALIZED),
            )
            evolved = project_score_map_to_original(
                evolved_maps[index], shape,
                ScoreMapGeometry(MapFrame.FULL_IMAGE_NORMALIZED),
            )
            panels = (
                _outline(rgb, gt),
                _overlay(rgb, baseline, baseline_max),
                _overlay(rgb, raw, logic_max),
                _overlay(rgb, greedy, logic_max),
                _overlay(rgb, evolved, logic_max),
            )
            for column, panel in enumerate(panels):
                axes[row, column].imshow(panel)
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                axes[row, column].spines[:].set_visible(False)
            axes[row, 0].set_ylabel(
                f"{_family(str(path)).title()}\n{path.stem}", rotation=0,
                ha="right", va="center", fontsize=10,
            )
            manifest["rows"].append({
                "family": _family(str(path)), "path": str(path),
                "baseline_score": float(baseline_scores[index]),
                "gt_pixels": int(np.count_nonzero(gt)),
            })
        for axis, title in zip(axes[0], (
            "Image + GT outline", "INP baseline", "Raw Qwen logic",
            "Greedy logic", "Evolved logic"
        )):
            axis.set_title(title, weight="bold")
        fig.suptitle("Deterministically selected localization examples", weight="bold")
        fig.savefig(target, dpi=170, facecolor="white")
        plt.close(fig)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--rule-run", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _json(args.baseline_run / "metrics.json")
    rule_metrics = _json(args.rule_run / "rule_metrics.json")
    verification = _json(args.verification)
    _metric_figure(
        baseline, rule_metrics, args.category,
        args.output_dir / "image_auroc_comparison.png",
    )
    _pareto_figure(verification, args.output_dir / "search_pareto.png")
    _spro_figure(
        args.baseline_run / "spro_curves.npz",
        args.rule_run / "spro_curves.npz", args.category,
        args.output_dir / "spro_curves.png",
    )
    _qualitative_figure(
        args.baseline_run / "cache" / f"{args.category}.npz",
        args.rule_run / "rule_maps.npz",
        args.output_dir / "localization_examples.png",
        args.output_dir / "localization_examples_manifest.json",
    )
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
