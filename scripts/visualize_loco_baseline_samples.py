#!/usr/bin/env python3
"""Visualize deterministic logical/structural examples for five LOCO runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from PIL import Image


LOCO_CLASSES = (
    "breakfast_box", "juice_bottle", "pushpins", "screw_bag",
    "splicing_connectors",
)


def _cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "paths": payload["paths"].astype(str),
            "maps": payload["maps"].astype(np.float32),
            "scores": payload["scores"].astype(np.float32),
        }


def _family(path: str) -> str:
    defect = Path(path).parent.name.lower()
    if "logical" in defect:
        return "logical"
    if "structural" in defect:
        return "structural"
    return "good"


def _median_index(cache: dict[str, np.ndarray], family: str) -> int:
    indices = np.asarray([
        index for index, path in enumerate(cache["paths"])
        if _family(str(path)) == family
    ], dtype=np.int64)
    ordered = indices[np.argsort(cache["scores"][indices], kind="stable")]
    return int(ordered[(len(ordered) - 1) // 2])


def _model_crop(image: Image.Image, size: int, *, nearest: bool = False) -> np.ndarray:
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    image = image.resize((448, 448), resample=method)
    image = image.crop((28, 28, 420, 420))
    return np.asarray(image.resize((size, size), resample=method))


def _image_and_mask(path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as handle:
        image = _model_crop(handle.convert("RGB"), size)
    mask_path = (
        path.parents[2] / "ground_truth" / path.parent.name
        / f"{path.stem}_mask.png"
    )
    with Image.open(mask_path) as handle:
        mask = _model_crop(handle.convert("L"), size, nearest=True)
    return image, mask


def _outline(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    contours = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )[0]
    cv2.drawContours(result, contours, -1, (230, 55, 55), 2)
    return result


def _overlay(image: np.ndarray, score_map: np.ndarray, maximum: float) -> np.ndarray:
    normalized = np.clip(score_map / max(maximum, 1e-12), 0.0, 1.0)
    heat = plt.get_cmap("magma")(normalized)[..., :3]
    return np.clip(.55 * image / 255.0 + .45 * heat, 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True,
        help="CLASS=directory containing cache/CLASS.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for entry in args.run:
        category, separator, raw_path = entry.partition("=")
        if not separator or category not in LOCO_CLASSES:
            raise ValueError("--run must be LOCO_CLASS=RUN_DIR")
        runs[category] = Path(raw_path).expanduser().resolve()
    if set(runs) != set(LOCO_CLASSES):
        raise ValueError("exactly one run for every LOCO class is required")

    caches = {
        name: _cache(runs[name] / "cache" / f"{name}.npz")
        for name in LOCO_CLASSES
    }
    maximum = float(np.quantile(np.concatenate([
        row["maps"].reshape(-1) for row in caches.values()
    ]), .995))
    size = 256
    fig, axes = plt.subplots(
        len(LOCO_CLASSES), 6, figsize=(18, 14.5), constrained_layout=True,
    )
    manifest = {
        "selection": "median frozen-baseline image score within class/family",
        "families": ["logical", "structural"],
        "map_frame": "model_center_crop",
        "heatmap_common_q995": maximum,
        "rows": [],
    }
    for row_index, category in enumerate(LOCO_CLASSES):
        cache = caches[category]
        for family_index, family in enumerate(("logical", "structural")):
            index = _median_index(cache, family)
            path = Path(str(cache["paths"][index]))
            image, mask = _image_and_mask(path, size)
            score_map = cv2.resize(
                cache["maps"][index], (size, size), interpolation=cv2.INTER_LINEAR
            )
            panels = (
                _outline(image, mask),
                _overlay(image, score_map, maximum),
                score_map,
            )
            for panel_index, panel in enumerate(panels):
                column = family_index * 3 + panel_index
                if panel_index == 2:
                    axes[row_index, column].imshow(
                        panel, cmap="magma", vmin=0.0, vmax=maximum,
                    )
                else:
                    axes[row_index, column].imshow(panel)
                axes[row_index, column].set_xticks([])
                axes[row_index, column].set_yticks([])
                axes[row_index, column].spines[:].set_visible(False)
            axes[row_index, family_index * 3 + 1].set_xlabel(
                f"score={cache['scores'][index]:.3f} · {path.stem}", fontsize=9,
            )
            manifest["rows"].append({
                "class_name": category, "family": family,
                "path": str(path), "score": float(cache["scores"][index]),
                "cache_index": index,
            })
        axes[row_index, 0].set_ylabel(
            category.replace("_", "\n"), rotation=0, ha="right",
            va="center", fontsize=11, weight="bold",
        )
    titles = (
        "Logical · GT outline", "Logical · INP overlay", "Logical · raw map",
        "Structural · GT outline", "Structural · INP overlay", "Structural · raw map",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, weight="bold", fontsize=11)
    scalar = plt.cm.ScalarMappable(
        norm=colors.Normalize(vmin=0.0, vmax=maximum), cmap="magma"
    )
    fig.colorbar(
        scalar, ax=axes[:, [2, 5]], location="bottom", shrink=.45,
        label="Raw INP map value · common five-class 0–99.5% scale",
    )
    fig.suptitle(
        "MVTec LOCO K=4 baseline · deterministic median-score examples",
        weight="bold", fontsize=16,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor="white")
    plt.close(fig)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
