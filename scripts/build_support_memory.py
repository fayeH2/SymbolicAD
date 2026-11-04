#!/usr/bin/env python
"""Build normal support memory for one or more categories."""


from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse

from symbolicAD.config import load_config
from symbolicAD.datasets.mvtec_loco import LOCO_CLASSES, load_image, select_support
from symbolicAD.grounding.support_memory import build_support_memory, save_support_memory
from symbolicAD.models.baseline_adapter import build_baseline
from symbolicAD.models.clip_backbone import CLIPBackbone
from symbolicAD.rules.qwen_rule_generator import load_or_generate_rules
from symbolicAD.rules.validator import validate_rule_kb


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SymbolicAD support memory")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config.paths.data_root = args.data_root
    if args.shot is not None:
        config.shot = args.shot
    if args.device:
        config.model.device = args.device

    categories = list(LOCO_CLASSES) if args.all or not args.category else [args.category]
    backbone = CLIPBackbone(
        model_name=config.model.clip_model,
        device=config.model.device,
        image_size=config.model.image_size,
        feature_layers=config.model.feature_layers,
    )
    try:
        for category in categories:
            class_root = config.data_root / category
            support_paths = select_support(class_root, config.shot, config.support_seed)
            support_images = [load_image(path) for path in support_paths]
            rule_kb = load_or_generate_rules(
                category,
                config.rule_cache_dir,
                support_images=support_images,
                config=config,
            )
            rule_kb, _ = validate_rule_kb(rule_kb)
            baseline = build_baseline(backbone, support_images, category, config)
            memory = build_support_memory(
                backbone, support_images, rule_kb, config, baseline=baseline
            )
            path = config.support_memory_dir / f"{category}_k{config.shot}.npz"
            save_support_memory(memory, path)
            print(f"{category}: wrote {path}")
    finally:
        backbone.close()


if __name__ == "__main__":
    main()
