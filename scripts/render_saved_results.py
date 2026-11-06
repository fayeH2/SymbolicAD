#!/usr/bin/env python
"""Render reference images and 6-panel figures from a completed run.

The full experiments are often executed with ``--no-visualizations --no-maps``
for speed.  This script reconstructs the visual outputs from the saved
``predictions/{category}/`` and ``traces/{category}/*.json`` without running
CLIP again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as _Path
from pathlib import Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import numpy as np

from symbolicAD.config import load_config
from symbolicAD.array_store import load_array_bundle
from symbolicAD.datasets.mvtec_loco import load_gt_mask, load_image
from symbolicAD.rules.executor import LogicResult
from symbolicAD.visualization.render_results import ResultRenderer
from symbolicAD.visualization.result_layout import latest_run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render saved SymbolicAD results")
    parser.add_argument(
        "--run-dir",
        action="append",
        default=None,
        help="completed run directory; may be repeated",
    )
    parser.add_argument(
        "--results-root",
        default="results_2shot",
        help="base directory used by --latest",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="render the newest timestamped run under --results-root",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="render at most N samples per category (0 = all)",
    )
    return parser


def _logic_result_from_trace(payload: dict) -> LogicResult | None:
    logic = payload.get("logic")
    if not isinstance(logic, dict):
        return None
    return LogicResult(
        root_truth=float(logic.get("root_truth", 1.0)),
        logic_score=float(logic.get("logic_score", 0.0)),
        failed_atoms=list(logic.get("failed_atoms", [])),
        atom_truths=dict(logic.get("atom_truths", {})),
        active_failed_atom_ids=list(logic.get("active_failed_atom_ids", [])),
        node_trace=dict(logic.get("node_trace", {})),
    )


def render_run(run_dir: Path, limit: int = 0) -> None:
    run_dir = Path(run_dir)
    config_path = run_dir / "checkpoint" / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = load_config(config_path)
    config.paths.results_dir = str(run_dir)
    config.output.save_reference_images = True
    config.output.save_visualizations = True
    config.output.save_maps = True
    config.output.save_traces = False

    metrics_path = run_dir / "metrics_summary.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    renderer = ResultRenderer(config)

    for category in metrics:
        trace_dir = run_dir / "traces" / category
        prediction_path = run_dir / "predictions" / category
        legacy_prediction = run_dir / "predictions" / f"{category}.npz"
        if not prediction_path.is_dir() and legacy_prediction.is_file():
            prediction_path = legacy_prediction
        if not trace_dir.is_dir() or not prediction_path.exists():
            print(f"[render] skip {category}: missing traces or predictions")
            continue
        traces = sorted(trace_dir.glob("*.json"))
        if limit > 0:
            traces = traces[:limit]
        if prediction_path.is_dir():
            payload, _ = load_array_bundle(prediction_path)
        else:
            payload = np.load(prediction_path, allow_pickle=False)
        maps = {
            name: payload[f"{name}_maps"].astype(np.float32)
            for name in ("baseline", "logic", "final")
        }
        print(f"[render] {category}: {len(traces)} samples")
        for index, trace_path in enumerate(traces):
            if index >= len(maps["final"]):
                break
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            extra = dict(trace.get("extra", {}))
            image_path = extra.get("image_path")
            if not image_path:
                continue
            image = load_image(image_path)
            sample_id = str(extra.get("sample_id", trace_path.stem.split("_", 1)[-1]))
            defect_type = str(extra.get("defect_type", extra.get("group", "unknown")))
            mask_path = (
                config.data_root
                / category
                / "ground_truth"
                / defect_type
                / f"{sample_id}_mask.png"
            )
            gt_mask = load_gt_mask(mask_path, (image.height, image.width))
            scores = {
                name: float(trace.get("scores", {}).get(name, 0.0))
                for name in ("baseline", "logic", "final")
            }
            renderer.save(
                category=category,
                image_id=f"{extra.get('group', 'group')}_{sample_id}",
                image=image,
                gt_mask=gt_mask,
                maps={name: maps[name][index] for name in maps},
                scores=scores,
                atom_results=None,
                logic_result=_logic_result_from_trace(trace),
                extra=extra,
            )
    print(f"[render] outputs written to {run_dir}")


def main() -> None:
    args = build_parser().parse_args()
    run_dirs = [Path(path) for path in (args.run_dir or [])]
    if not run_dirs:
        root = Path(args.results_root)
        if not args.latest and not root.is_dir():
            raise FileNotFoundError(root)
        run_dirs = [latest_run_dir(root)]
    for run_dir in run_dirs:
        render_run(run_dir, limit=int(args.limit))


if __name__ == "__main__":
    main()
