#!/usr/bin/env python
"""Thin inference wrapper around the staged pipeline."""


from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse

from symbolicAD.config import load_config
from symbolicAD.datasets.mvtec_loco import LOCO_CLASSES
from symbolicAD.pipeline import run_categories


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SymbolicAD inference")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--mode", choices=["baseline", "baseline_logic", "logic"], default="baseline_logic"
    )
    parser.add_argument("--category", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--no-visualizations", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config.mode = args.mode
    if args.max_images is not None:
        config.max_images_per_group = args.max_images
    if args.no_visualizations:
        config.output.save_visualizations = False
    categories = list(LOCO_CLASSES) if args.all or not args.category else [args.category]
    run_categories(config, categories)


if __name__ == "__main__":
    main()
