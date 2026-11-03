#!/usr/bin/env python
"""Entry point for the staged SymbolicAD V3.2 pipeline.

Examples
--------
Baseline-only smoke run::

    python run_pipeline.py --mode baseline --category juice_bottle --max-images 20

Baseline + logic (Stages 2-3)::

    python run_pipeline.py --mode baseline_logic --category juice_bottle
"""

from __future__ import annotations

import argparse
from pathlib import Path

from symbolicAD.config import load_config
from symbolicAD.datasets.mvtec_loco import LOCO_CLASSES
from symbolicAD.pipeline import run_categories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SymbolicAD V3.2 staged pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["baseline", "baseline_logic", "logic"], default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--category", default=None, help="single category name")
    parser.add_argument("--all", action="store_true", help="run every LOCO class")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None, help="limit per test group")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--force-support", action="store_true", help="rebuild support memory")
    parser.add_argument("--force-rules", action="store_true", help="reload rule templates")
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--no-maps", action="store_true")
    parser.add_argument("--no-pro", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    if args.mode:
        config.mode = args.mode
    if args.dataset:
        config.dataset = args.dataset
    if args.data_root:
        config.paths.data_root = args.data_root
    if args.shot is not None:
        config.shot = args.shot
    if args.seed is not None:
        config.seed = args.seed
        config.support_seed = args.seed
    if args.device:
        config.model.device = args.device
    if args.batch_size is not None:
        config.model.batch_size = args.batch_size
    if args.max_images is not None:
        config.max_images_per_group = args.max_images
    if args.cache_dir:
        config.paths.cache_dir = args.cache_dir
    if args.results_dir:
        config.paths.results_dir = args.results_dir
    if args.no_visualizations:
        config.output.save_visualizations = False
    if args.no_maps:
        config.output.save_maps = False
    if args.no_pro:
        config.evaluation.enable_pro = False

    if args.all:
        categories = list(LOCO_CLASSES)
    elif args.category:
        categories = [args.category]
    else:
        categories = [config.category] if config.category else list(LOCO_CLASSES)

    print(
        f"[SymbolicAD] dataset={config.dataset} categories={categories} "
        f"mode={config.mode} shot={config.shot} device={config.model.device}",
        flush=True,
    )
    run_categories(
        config,
        categories,
        force_support=args.force_support,
        force_rules=args.force_rules,
    )
    print(f"[SymbolicAD] results written to {Path(config.results_dir).resolve()}")


if __name__ == "__main__":
    main()
