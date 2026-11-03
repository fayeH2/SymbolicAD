"""End-to-end staged pipeline: support -> baseline -> grounding -> logic -> metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .config import Config
from .datasets.mvtec_loco import (
    collect_test_samples,
    limit_per_group,
    load_gt_mask,
    load_image,
    select_support,
)
from .evaluation.evaluator import CategoryEvaluator
from .grounding.component_maps import build_component_maps, normalize_with_support
from .grounding.instance_extractor import extract_instances
from .grounding.relation_grounder import evaluate_atoms
from .grounding.scene_graph import SceneGraph
from .grounding.support_memory import (
    SupportMemory,
    build_support_memory,
    load_support_memory,
    save_support_memory,
)
from .localization.fusion import (
    final_image_score,
    fuse_maps,
    logic_image_score,
    normalize_map,
    topk_mean,
)
from .localization.responsibility import build_logic_heatmap
from .models.baseline_adapter import CLIPZeroShotBaseline
from .models.clip_backbone import CLIPBackbone
from .rules.executor import execute_formula
from .rules.qwen_rule_generator import load_or_generate_rules
from .rules.schema import RuleKB
from .rules.validator import validate_against_support, validate_rule_kb
from .visualization.render_results import ResultRenderer


def _support_memory_path(config: Config, category: str) -> Path:
    return config.support_memory_dir / f"{category}_k{config.shot}.npz"


def _build_scene_graph(
    image_index: int,
    rule_kb: RuleKB,
    component_maps: Dict[str, np.ndarray],
    support_memory: SupportMemory,
    patches: np.ndarray,
    config: Config,
) -> SceneGraph:
    scene_graph = SceneGraph(component_maps={}, instances={}, patch_features=patches)
    for symbol in rule_kb.components:
        name = symbol.name
        raw = component_maps[name][image_index]
        if support_memory.has_component(name):
            memory = support_memory.components[name]
            normalized = normalize_with_support(
                raw, memory.score_q_low, memory.score_q_high
            )
        else:
            normalized = np.clip(raw, 0.0, 1.0)
        scene_graph.component_maps[name] = normalized
        scene_graph.instances[name] = extract_instances(
            component_map=normalized,
            patch_features=patches,
            concept=name,
            threshold=config.grounding.component_threshold,
            min_area=config.grounding.min_instance_area,
            morphology_kernel=config.grounding.morphology_kernel,
        )
    return scene_graph


def _baseline_scores_and_maps(
    baseline: CLIPZeroShotBaseline,
    patches: np.ndarray,
    baseline_stats: Dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    scored = baseline.score_from_features(patches)
    raw_maps = scored["anomaly_map"]
    if baseline_stats and "q_low" in baseline_stats and "q_high" in baseline_stats:
        maps = np.stack(
            [
                normalize_map(
                    raw_maps[index],
                    baseline_stats["q_low"],
                    baseline_stats["q_high"],
                )
                for index in range(raw_maps.shape[0])
            ],
            axis=0,
        )
    else:
        maps = np.clip(raw_maps, 0.0, 1.0)
    return maps, scored["image_score"]


def _prepare_rule_bank(
    category: str,
    config: Config,
    support_memory: Optional[SupportMemory],
    force_rules: bool,
) -> tuple[RuleKB, Dict[str, object]]:
    rule_kb = load_or_generate_rules(category, config.rule_cache_dir, force=force_rules)
    rule_kb, structural_report = validate_rule_kb(rule_kb)
    support_report = None
    if support_memory is not None:
        support_report = validate_against_support(rule_kb, support_memory)
    return rule_kb, {
        "structural": structural_report.to_dict(),
        "support": support_report.to_dict() if support_report is not None else None,
    }


def run_category(
    config: Config,
    category: str,
    backbone: Optional[CLIPBackbone] = None,
    force_support: bool = False,
    force_rules: bool = False,
) -> Dict[str, object]:
    config.category = category
    if config.mode not in {"baseline", "baseline_logic", "logic"}:
        raise ValueError(f"unsupported mode: {config.mode}")

    class_root = config.data_root / category
    if not class_root.is_dir():
        raise FileNotFoundError(class_root)

    support_paths = select_support(class_root, config.shot, config.support_seed)
    support_images = [load_image(path) for path in support_paths]

    owns_backbone = backbone is None
    if backbone is None:
        backbone = CLIPBackbone(
            model_name=config.model.clip_model,
            device=config.model.device,
            image_size=config.model.image_size,
        )

    baseline = CLIPZeroShotBaseline(
        backbone,
        category=category,
        cache_dir=config.text_embedding_dir,
        topk_percent=config.fusion.topk_percent,
    )

    use_logic = config.mode in {"baseline_logic", "logic"}
    support_memory: Optional[SupportMemory] = None
    rule_kb: Optional[RuleKB] = None
    validation: Dict[str, object] = {}

    memory_path = _support_memory_path(config, category)
    if use_logic:
        if memory_path.is_file() and not force_support:
            support_memory = load_support_memory(memory_path)
            if not support_memory.baseline_stats:
                support_memory.baseline_stats = baseline.support_statistics(
                    support_images,
                    low_quantile=config.grounding.support_quantile_low,
                    high_quantile=config.grounding.support_quantile_high,
                )
                save_support_memory(support_memory, memory_path)
        else:
            rule_kb = load_or_generate_rules(category, config.rule_cache_dir, force=force_rules)
            rule_kb, _ = validate_rule_kb(rule_kb)
            support_memory = build_support_memory(
                backbone, support_images, rule_kb, config, baseline=baseline
            )
            save_support_memory(support_memory, memory_path)
        rule_kb, validation = _prepare_rule_bank(
            category, config, support_memory, force_rules
        )
    else:
        baseline_stats = baseline.support_statistics(
            support_images,
            low_quantile=config.grounding.support_quantile_low,
            high_quantile=config.grounding.support_quantile_high,
        )
        support_memory = SupportMemory(
            category=category,
            grid_size=(backbone.grid_size, backbone.grid_size),
            baseline_stats=baseline_stats,
            feature_dim=backbone.embed_dim,
        )

    samples = collect_test_samples(class_root)
    samples = limit_per_group(samples, config.max_images_per_group, seed=config.seed)

    evaluator = CategoryEvaluator(config)
    renderer = ResultRenderer(config)
    batch_size = max(1, int(config.model.batch_size))

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        images = [load_image(sample.path) for sample in batch]
        encoded = backbone.encode_images(images)
        patches = encoded["patches"]
        base_maps, raw_scores = _baseline_scores_and_maps(
            baseline, patches, support_memory.baseline_stats
        )

        if use_logic and rule_kb is not None and support_memory is not None:
            component_maps = build_component_maps(
                backbone,
                patches,
                rule_kb,
                support_memory=support_memory,
                alpha=config.grounding.alpha,
                cache_dir=config.text_embedding_dir,
                text_cache_enabled=config.grounding.text_embedding_cache,
            )
        else:
            component_maps = {}

        for offset, sample in enumerate(batch):
            image = images[offset]
            gt_mask = load_gt_mask(sample.mask_path, (image.height, image.width))
            base_map = base_maps[offset]
            logic_map = np.zeros_like(base_map, dtype=np.float32)
            logic_result = None
            atom_results = {}
            scene_summary: Dict[str, object] = {}

            if use_logic and rule_kb is not None and support_memory is not None:
                scene_graph = _build_scene_graph(
                    offset, rule_kb, component_maps, support_memory, patches[offset], config
                )
                atom_results = evaluate_atoms(
                    scene_graph, rule_kb, support_memory, config
                )
                logic_result = execute_formula(
                    rule_kb.formula,
                    atom_results,
                    fail_threshold=config.logic.fail_threshold,
                    epsilon=config.logic.and_epsilon,
                )
                logic_map, _ = build_logic_heatmap(
                    atom_results,
                    rule_kb,
                    patches[offset],
                    backbone,
                    config,
                    logic_result=logic_result,
                )
                final_map = fuse_maps(base_map, logic_map, config.fusion.lambda_logic)
                base_score = float(
                    topk_mean(base_map[None, ...], config.fusion.topk_percent)[0]
                )
                logic_score = logic_image_score(
                    logic_map, logic_result.root_truth, config.fusion.topk_percent
                )
                final_score = final_image_score(
                    base_score, logic_score, config.fusion.beta
                )
                scene_summary = {
                    "counts": {
                        name: len(instances)
                        for name, instances in scene_graph.instances.items()
                    },
                    "estimated_counts": {
                        atom_id: result.evidence.get("estimated_count")
                        for atom_id, result in atom_results.items()
                        if result.kind == "CountEq"
                    },
                }
            else:
                final_map = base_map
                base_score = float(
                    topk_mean(base_map[None, ...], config.fusion.topk_percent)[0]
                )
                logic_score = 0.0
                final_score = base_score

            scores = {
                "baseline": base_score,
                "logic": logic_score,
                "final": final_score,
            }
            maps = {
                "baseline": base_map,
                "logic": logic_map,
                "final": final_map,
            }
            evaluator.update(
                scores, maps, label=sample.label, group=sample.group, gt_mask=gt_mask
            )
            image_id = f"{sample.group}_{sample.image_id}"
            renderer.save(
                category=category,
                image_id=image_id,
                image=image,
                gt_mask=gt_mask,
                maps=maps,
                scores=scores,
                atom_results=atom_results,
                logic_result=logic_result,
                extra={
                    "image_path": str(sample.path),
                    "label": int(sample.label),
                    "group": sample.group,
                    "defect_type": sample.defect_type,
                    "baseline_raw_score": float(raw_scores[offset]),
                    **scene_summary,
                },
            )

    metrics = evaluator.compute()
    metrics["category"] = category
    metrics["mode"] = config.mode
    metrics["support"] = [str(path) for path in support_paths]
    metrics["validation"] = validation
    metrics["support_memory"] = str(memory_path) if use_logic else None

    if config.output.save_predictions:
        prediction_dir = config.results_dir / "predictions"
        evaluator.dump_predictions(prediction_dir / f"{category}.npz")

    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"metrics_{category}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if owns_backbone:
        backbone.close()
    return metrics


def run_categories(
    config: Config,
    categories: Sequence[str],
    force_support: bool = False,
    force_rules: bool = False,
) -> Dict[str, object]:
    backbone = CLIPBackbone(
        model_name=config.model.clip_model,
        device=config.model.device,
        image_size=config.model.image_size,
    )
    all_metrics: Dict[str, object] = {}
    try:
        for category in categories:
            print(f"[SymbolicAD] running {category} ({config.mode})", flush=True)
            all_metrics[category] = run_category(
                config,
                category,
                backbone=backbone,
                force_support=force_support,
                force_rules=force_rules,
            )
    finally:
        backbone.close()

    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics_summary.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(all_metrics, results_dir / "metrics_per_class.csv")
    return all_metrics


def _write_csv(all_metrics: Dict[str, object], path: Path) -> None:
    import csv

    rows = []
    for category, payload in all_metrics.items():
        for group, group_payload in payload.get("groups", {}).items():
            for variant in ("baseline", "logic", "final"):
                variant_payload = group_payload.get(variant, {})
                image = variant_payload.get("image", {})
                pixel = variant_payload.get("pixel", {})
                pro = variant_payload.get("pro", {})
                rows.append(
                    {
                        "category": category,
                        "group": group,
                        "variant": variant,
                        "image_auroc": image.get("auroc"),
                        "image_ap": image.get("ap"),
                        "image_f1max": image.get("f1max"),
                        "pixel_auroc": pixel.get("auroc"),
                        "pixel_ap": pixel.get("ap"),
                        "pixel_f1max": pixel.get("f1max"),
                        "aupro": pro.get("aupro"),
                        "n_samples": group_payload.get("n_samples"),
                    }
                )
    fieldnames = [
        "category", "group", "variant", "image_auroc", "image_ap",
        "image_f1max", "pixel_auroc", "pixel_ap", "pixel_f1max", "aupro",
        "n_samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
