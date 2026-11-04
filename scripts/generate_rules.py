#!/usr/bin/env python
"""Offline-only Qwen rule proposal generation into ``cache/rules``.

This is the only deployment workflow command allowed to load Qwen. The output
must subsequently be frozen by ``compile_static_program.py``.
"""


from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from symbolicAD.config import load_config
from symbolicAD.datasets.mvtec_loco import LOCO_CLASSES, load_image, select_support
from symbolicAD.rules.qwen_rule_generator import (
    DEFAULT_TEMPLATE_DIR,
    load_or_generate_rules,
)
from symbolicAD.rules.validator import validate_rule_kb


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/cache SymbolicAD rule banks")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--template-dir", default=str(DEFAULT_TEMPLATE_DIR))
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.cache_dir:
        config.paths.cache_dir = args.cache_dir
    if args.shot is not None:
        config.shot = args.shot
    rule_cache_dir = config.rule_cache_dir
    categories = list(LOCO_CLASSES) if args.all or not args.category else [args.category]
    for category in categories:
        class_root = config.data_root / category
        support_paths = select_support(class_root, config.shot, config.support_seed)
        support_images = [load_image(path) for path in support_paths]
        rule_kb = load_or_generate_rules(
            category,
            rule_cache_dir,
            template_dir=Path(args.template_dir),
            force=args.force,
            support_images=support_images,
            config=config,
        )
        rule_kb, report = validate_rule_kb(rule_kb)
        rule_kb.save(rule_cache_dir / f"{category}.json")
        print(
            f"{category}: {len(rule_kb.components)} components, "
            f"{sum(atom.weight > 0 for atom in rule_kb.atoms)} active atoms"
        )
        for issue in report.issues:
            print(f"  - {issue}")


if __name__ == "__main__":
    main()
