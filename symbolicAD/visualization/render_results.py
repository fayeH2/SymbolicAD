"""Save per-image combination figures, raw maps and reasoning traces."""

from __future__ import annotations

import json
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


class ResultRenderer:
    def __init__(self, config: Config):
        self.config = config
        self.visualization_dir = config.results_dir / "visualizations"
        self.map_dir = config.results_dir / "maps"
        self.trace_dir = config.results_dir / "traces"

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
