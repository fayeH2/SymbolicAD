"""Timestamped experiment layout matching the H2VLR-style result folders.

Reference layout::

    <run_root>/
    ├── csv/
    ├── imgs/
    ├── checkpoint/
    ├── maps/
    ├── traces/
    └── visualizations/

Every sample produces three images in ``imgs``:
``{category}-{defect}-{sample}_ori.jpg``, ``_gt.jpg`` and
``_{method}.jpg``.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class RunLayout:
    root: Path
    csv_dir: Path
    imgs_dir: Path
    checkpoint_dir: Path
    maps_dir: Path
    traces_dir: Path
    visualizations_dir: Path
    predictions_dir: Path

    @classmethod
    def create(
        cls,
        base_dir: str | Path,
        config=None,
        timestamp: Optional[str] = None,
    ) -> "RunLayout":
        base_dir = Path(base_dir).expanduser()
        # Experiment directories are always unique and self-identifying.  The
        # former opt-out made unrelated runs silently share one top-level
        # folder and overwrite each other's artifacts.
        stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        shot = getattr(config, "shot", None) if config is not None else None
        name = f"{stamp}_k{shot}" if shot else stamp
        root = base_dir / name
        layout = cls(
            root=root,
            csv_dir=root / "csv",
            imgs_dir=root / "imgs",
            checkpoint_dir=root / "checkpoint",
            maps_dir=root / "maps",
            traces_dir=root / "traces",
            visualizations_dir=root / "visualizations",
            predictions_dir=root / "predictions",
        )
        for directory in (
            layout.root,
            layout.csv_dir,
            layout.imgs_dir,
            layout.checkpoint_dir,
            layout.maps_dir,
            layout.traces_dir,
            layout.visualizations_dir,
            layout.predictions_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return layout


def latest_run_dir(base_dir: str | Path) -> Path:
    """Return the newest timestamped run directory under ``base_dir``."""
    base_dir = Path(base_dir).expanduser()
    if not base_dir.is_dir():
        raise FileNotFoundError(base_dir)
    candidates = [path for path in base_dir.iterdir() if path.is_dir()]
    if not candidates:
        return base_dir
    timestamped = [
        path for path in candidates if path.name[:8].isdigit() and path.name[8] == "_"
    ]
    return max(timestamped or candidates, key=lambda path: path.name)


def _metric(value) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number != number:  # NaN
        return ""
    return f"{number * 100.0:.2f}"


def write_reference_csv(all_metrics: Dict[str, dict], path: str | Path) -> Path:
    """Write an H2VLR-style per-class CSV (final/all variant, percent values)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "", "i_roc", "p_roc", "i_ap", "p_ap", "i_f1", "p_f1", "aupro",
    ]
    rows: List[dict] = []
    for category, payload in all_metrics.items():
        group = payload.get("groups", {}).get("all", {})
        final = group.get("final", {})
        image = final.get("image", {})
        pixel = final.get("pixel", {})
        pro = final.get("pro", {})
        rows.append(
            {
                "": f"mvtec-{category}",
                "i_roc": _metric(image.get("auroc")),
                "p_roc": _metric(pixel.get("auroc")),
                "i_ap": _metric(image.get("ap")),
                "p_ap": _metric(pixel.get("ap")),
                "i_f1": _metric(image.get("f1max")),
                "p_f1": _metric(pixel.get("f1max")),
                "aupro": _metric(pro.get("aupro")),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_metrics_csv(all_metrics: Dict[str, dict], path: str | Path) -> Path:
    """Write all group/variant metrics as a flat CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category", "group", "variant", "n_samples",
        "image_auroc", "image_ap", "image_f1max",
        "pixel_auroc", "pixel_ap", "pixel_f1max", "aupro",
    ]
    rows = []
    for category, payload in all_metrics.items():
        for group, group_payload in payload.get("groups", {}).items():
            for variant in ("baseline", "logic", "final"):
                item = group_payload.get(variant, {})
                image = item.get("image", {})
                pixel = item.get("pixel", {})
                pro = item.get("pro", {})
                rows.append(
                    {
                        "category": category,
                        "group": group,
                        "variant": variant,
                        "n_samples": group_payload.get("n_samples"),
                        "image_auroc": image.get("auroc"),
                        "image_ap": image.get("ap"),
                        "image_f1max": image.get("f1max"),
                        "pixel_auroc": pixel.get("auroc"),
                        "pixel_ap": pixel.get("ap"),
                        "pixel_f1max": pixel.get("f1max"),
                        "aupro": pro.get("aupro"),
                    }
                )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_run_checkpoint(
    layout: RunLayout,
    config,
    categories: Iterable[str],
) -> None:
    """Persist rules, support memory and run config in ``checkpoint/``."""
    layout.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (layout.checkpoint_dir / "run_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for category in categories:
        program = config.cache_dir / "static_programs" / f"{category}_k{config.shot}"
        for source_name, target_name in (
            ("rules.json", f"{category}_static_rules.json"),
            ("support_memory", f"{category}_static_support_memory"),
            ("symbolicdet", f"{category}_static_symbolicdet"),
            ("symbolicdet_expressions.json", f"{category}_symbolicdet.json"),
            ("manifest.json", f"{category}_static_manifest.json"),
        ):
            source = program / source_name
            if source.exists():
                target = layout.checkpoint_dir / target_name
                if source.resolve() != target.resolve():
                    if source.is_dir():
                        shutil.copytree(source, target, dirs_exist_ok=True)
                    else:
                        shutil.copyfile(source, target)
