"""Save per-image combination figures, raw maps and reasoning traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from symbolicAD.config import Config


def _to_rgb(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[-1] == 4:
        array = array[..., :3]
    return array.astype(np.uint8)


def _resize(map_values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    return cv2.resize(
        np.asarray(map_values, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def save_reference_triplet(
    imgs_dir: str | Path,
    category: str,
    defect_type: str,
    sample_id: str,
    image,
    gt_mask: np.ndarray,
    final_map: np.ndarray,
    method_name: str = "SymbolicAD",
    image_size: int = 400,
) -> None:
    """Save the reference-format ``*_ori.jpg``, ``*_gt.jpg``, ``*_{method}.jpg``."""
    imgs_dir = Path(imgs_dir)
    imgs_dir.mkdir(parents=True, exist_ok=True)
    size = int(image_size)
    defect = str(defect_type).replace("/", "_").replace("\\", "_")
    sample_id = str(sample_id).replace("/", "_")
    stem = f"{category}-{defect}-{sample_id}"

    rgb = cv2.resize(_to_rgb(image), (size, size), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(
        str(imgs_dir / f"{stem}_ori.jpg"),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    gt = cv2.resize(
        np.asarray(gt_mask, dtype=np.uint8),
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    )
    gt_overlay = rgb.copy()
    if gt.any():
        mask = gt > 0
        gt_overlay[mask] = (
            0.55 * np.array([255, 0, 0], dtype=np.float32)
            + 0.45 * gt_overlay[mask].astype(np.float32)
        ).astype(np.uint8)
    cv2.imwrite(
        str(imgs_dir / f"{stem}_gt.jpg"),
        cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    heat = cv2.resize(
        np.asarray(final_map, dtype=np.float32),
        (size, size),
        interpolation=cv2.INTER_LINEAR,
    )
    heat = np.clip(heat, 0.0, 1.0)
    heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    prediction = (
        0.5 * rgb.astype(np.float32) + 0.5 * heat.astype(np.float32)
    ).astype(np.uint8)
    cv2.imwrite(
        str(imgs_dir / f"{stem}_{method_name}.jpg"),
        cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )


class ResultRenderer:
    def __init__(self, config: Config):
        self.config = config
        root = config.results_dir
        self.root = root
        self.csv_dir = root / "csv"
        self.imgs_dir = root / "imgs"
        self.checkpoint_dir = root / "checkpoint"
        self.visualization_dir = root / "visualizations"
        self.map_dir = root / "maps"
        self.trace_dir = root / "traces"
        for directory in (
            self.csv_dir,
            self.imgs_dir,
            self.checkpoint_dir,
            self.visualization_dir,
            self.map_dir,
            self.trace_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _map_dtype(self):
        return np.float16 if self.config.output.map_dtype == "float16" else np.float32

    def save(
        self,
        category: str,
        image_id: str,
        image,
        gt_mask: np.ndarray,
        maps: Dict[str, np.ndarray],
        scores: Dict[str, float],
        atom_results: Optional[Dict[str, object]] = None,
        logic_result: Optional[object] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        rgb = _to_rgb(image)
        if self.config.output.save_reference_images:
            self._save_triplet(
                category=category,
                image_id=image_id,
                image=rgb,
                gt_mask=gt_mask,
                final_map=maps["final"],
                extra=extra or {},
            )
        if self.config.output.save_maps:
            map_dir = self.map_dir / category
            map_dir.mkdir(parents=True, exist_ok=True)
            for name, values in maps.items():
                np.save(
                    map_dir / f"{image_id}_{name}.npy",
                    np.asarray(values, dtype=self._map_dtype()),
                )
        if self.config.output.save_traces:
            trace_dir = self.trace_dir / category
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace = {
                "category": category,
                "image_id": image_id,
                "scores": {key: float(value) for key, value in scores.items()},
                "atoms": [
                    result.to_dict() for result in (atom_results or {}).values()
                ],
                "logic": logic_result.to_dict() if logic_result is not None else None,
                "extra": extra or {},
            }
            (trace_dir / f"{image_id}.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if self.config.output.save_visualizations:
            self._render(
                category, image_id, rgb, gt_mask, maps, atom_results, logic_result
            )

    def _save_triplet(
        self,
        category: str,
        image_id: str,
        image: np.ndarray,
        gt_mask: np.ndarray,
        final_map: np.ndarray,
        extra: Dict[str, object],
    ) -> None:
        save_reference_triplet(
            imgs_dir=self.imgs_dir,
            category=category,
            defect_type=str(extra.get("defect_type", extra.get("group", "unknown"))),
            sample_id=str(extra.get("sample_id", image_id)),
            image=image,
            gt_mask=gt_mask,
            final_map=final_map,
            method_name=str(self.config.output.method_name),
            image_size=int(self.config.output.image_size),
        )

    def _render(
        self,
        category: str,
        image_id: str,
        rgb: np.ndarray,
        gt_mask: np.ndarray,
        maps: Dict[str, np.ndarray],
        atom_results,
        logic_result,
    ) -> None:
        size = (rgb.shape[0], rgb.shape[1])
        base = _resize(maps["baseline"], size)
        logic = _resize(maps["logic"], size)
        final = _resize(maps["final"], size)
        mask = _resize(gt_mask.astype(np.float32), size)

        figure, axes = plt.subplots(1, 6, figsize=(21, 4))
        panels = [
            ("image", rgb, None),
            ("gt", mask, "gray"),
            ("baseline", base, "jet"),
            ("logic", logic, "jet"),
            ("final", final, "jet"),
        ]
        for axis, (title, values, cmap) in zip(axes[:5], panels):
            if title == "image":
                axis.imshow(values)
            else:
                axis.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0)
            axis.set_title(title)
            axis.axis("off")
        axes[5].imshow(rgb)
        axes[5].imshow(final, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
        axes[5].set_title("final overlay")
        axes[5].axis("off")

        lines = []
        if logic_result is not None:
            for failed in getattr(logic_result, "failed_atoms", [])[:6]:
                lines.append(f"{failed['expression']}: {failed['truth']:.2f}")
        if lines:
            figure.text(
                0.01,
                0.01,
                "  |  ".join(lines),
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.6},
            )
        figure.tight_layout()
        output = self.visualization_dir / category / f"{image_id}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=110, bbox_inches="tight")
        plt.close(figure)
