#!/usr/bin/env python
"""Compile Qwen proposals and normal CLIP support into an LLM-free program."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from symbolicAD.config import load_config
from symbolicAD.datasets.mvtec_loco import LOCO_CLASSES, load_image, select_support
from symbolicAD.grounding.support_memory import (
    LOGIC_CALIBRATION_VERSION,
    SUPPORT_MEMORY_VERSION,
    build_support_memory,
    rule_grounding_signature,
    save_support_memory,
    support_image_signature,
)
from symbolicAD.grounding.offline_calibration import calibrate_logic_on_support
from symbolicAD.models.baseline_adapter import build_baseline
from symbolicAD.models.clip_backbone import CLIPBackbone
from symbolicAD.rules.compiler import compile_rule_kb
from symbolicAD.rules.schema import RuleKB
from symbolicAD.rules.validator import validate_against_support, validate_rule_kb
from symbolicAD.static_program import STATIC_PROGRAM_VERSION, StaticProgramPaths
from symbolicAD.symbolicdet import SegmentFirstSymbolicDet


def _compile_category(config, category: str, backbone: CLIPBackbone) -> None:
    source = config.rule_cache_dir / f"{category}.json"
    if not source.is_file():
        raise FileNotFoundError(
            f"Qwen rule proposal is missing: {source}. "
            "Run scripts/generate_rules.py first"
        )
    class_root = config.data_root / category
    support_paths = select_support(class_root, config.shot, config.support_seed)
    support_images = [load_image(path) for path in support_paths]

    rule_kb, structural = validate_rule_kb(RuleKB.load(source))
    baseline = build_baseline(backbone, support_images, category, config)
    memory = build_support_memory(
        backbone, support_images, rule_kb, config, baseline=baseline
    )
    memory = calibrate_logic_on_support(
        backbone, support_images, memory, rule_kb, config
    )
    rule_kb, compile_report = compile_rule_kb(rule_kb, memory, config)
    support_report = validate_against_support(
        rule_kb,
        memory,
        disable_on_mismatch=bool(config.compiler.disable_inconsistent_atoms),
        min_attribute_support_score=float(config.compiler.min_attribute_support_score),
        min_atom_support_truth=float(config.logic.min_atom_support_truth),
    )
    # The saved memory and saved expression must have the same immutable
    # signature. Inference rejects any mismatch rather than silently rebuilding.
    memory.metadata["rule_grounding_signature"] = rule_grounding_signature(rule_kb)

    encoded = backbone.encode_images(support_images)
    symbolicdet = SegmentFirstSymbolicDet(config.symbolicdet).fit(
        support_images, encoded["patches"]
    )
    paths = StaticProgramPaths.for_category(config, category)
    paths.root.mkdir(parents=True, exist_ok=True)
    rule_kb.save(paths.rules)
    save_support_memory(memory, paths.support_memory)
    symbolicdet.save(paths.symbolicdet)
    (paths.root / "symbolicdet_expressions.json").write_text(
        json.dumps(symbolicdet.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "version": STATIC_PROGRAM_VERSION,
        "category": category,
        "shot": int(config.shot),
        "support_seed": int(config.support_seed),
        "support_paths": [str(path) for path in support_paths],
        "support_signature": support_image_signature(support_images),
        "clip_model": str(config.model.clip_model),
        "image_size": int(config.model.image_size),
        "rule_source": str(source),
        "rule_source_metadata": dict(rule_kb.metadata),
        "rule_grounding_signature": rule_grounding_signature(rule_kb),
        "support_memory_version": SUPPORT_MEMORY_VERSION,
        "logic_calibration_version": LOGIC_CALIBRATION_VERSION,
        "symbolicdet_version": symbolicdet.to_dict()["version"],
        "n_expressions": len(symbolicdet.expressions),
        "reports": {
            "structural": structural.to_dict(),
            "compile": compile_report.to_dict(),
            "support": support_report.to_dict(),
        },
        "runtime_contract": {
            "llm_allowed": False,
            "query_set_adaptation": False,
            "operation": "CLIP feature extraction plus static expression evaluation",
        },
    }
    paths.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{category}: compiled static program -> {paths.root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root:
        config.paths.data_root = args.data_root
    if args.shot is not None:
        config.shot = args.shot
    if args.seed is not None:
        config.seed = args.seed
        config.support_seed = args.seed
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
            _compile_category(config, category, backbone)
    finally:
        backbone.close()


if __name__ == "__main__":
    main()
