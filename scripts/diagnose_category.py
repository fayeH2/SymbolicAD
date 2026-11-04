#!/usr/bin/env python
"""Intermediate-process diagnosis for one MVTec LOCO category.

Produces:
  * the raw rule source and the final validated static rule bank;
  * support-memory statistics;
  * CLIP text / prototype / patch alignment diagnostics;
  * per-image groundings, atom truths and logic heat-map overlap;
  * baseline / logic / final metrics and qualitative visualisations.

Example:
    python scripts/diagnose_category.py --category juice_bottle \
        --output-dir 1103results --shot 4
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from symbolicAD.config import load_config
from symbolicAD.datasets.mvtec_loco import (
    collect_test_samples,
    limit_per_group,
    load_gt_mask,
    load_image,
    select_support,
)
from symbolicAD.evaluation.evaluator import CategoryEvaluator
from symbolicAD.evaluation.image_metrics import image_metrics
from symbolicAD.grounding.component_maps import (
    build_component_maps,
    normalize_with_support,
    prototype_similarity_map,
    text_similarity_map,
)
from symbolicAD.grounding.instance_extractor import extract_instances
from symbolicAD.grounding.relation_grounder import evaluate_atoms
from symbolicAD.grounding.scene_graph import SceneGraph
from symbolicAD.grounding.support_memory import build_support_memory
from symbolicAD.localization.fusion import (
    final_image_score,
    fuse_maps,
    logic_image_score,
    normalize_map,
    topk_mean,
)
from symbolicAD.localization.responsibility import build_logic_heatmap
from symbolicAD.models.baseline_adapter import build_baseline
from symbolicAD.models.clip_backbone import CLIPBackbone
from symbolicAD.models.text_embeddings import encode_prompts_cached
from symbolicAD.rules.compiler import compile_rule_kb
from symbolicAD.rules.executor import execute_formula
from symbolicAD.rules.qwen_rule_generator import (
    DEFAULT_TEMPLATE_DIR,
    load_or_generate_rules,
)
from symbolicAD.rules.schema import RuleKB
from symbolicAD.rules.validator import validate_against_support, validate_rule_kb
from symbolicAD.visualization.render_results import save_reference_triplet
from symbolicAD.visualization.result_layout import (
    RunLayout,
    save_run_checkpoint,
    write_metrics_csv,
    write_reference_csv,
)

GROUPS = ["good", "logical", "structural"]


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return (
        cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        > 0
    )


def topk_fraction_in_mask(
    values: np.ndarray, mask: np.ndarray, fraction: float = 0.05
) -> float:
    values = np.asarray(values, dtype=np.float32)
    flat = values.reshape(-1)
    count = max(1, int(round(flat.size * fraction)))
    indices = np.argpartition(flat, flat.size - count)[-count:]
    selected = np.zeros(flat.size, dtype=bool)
    selected[indices] = True
    selected = selected.reshape(values.shape)
    mask = mask.astype(bool)
    return float((selected & mask).sum() / max(mask.sum(), 1)) if mask.any() else 0.0


def contrast(values: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not mask.any() or mask.all():
        return 0.0
    return float(values[mask].mean() - values[~mask].mean())


def gt_overlap(
    heatmap: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    gt = resize_mask(gt_mask, heatmap.shape)
    if not gt.any():
        return {
            "gt_pixels": 0.0,
            "mass_on_gt": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "iou": float("nan"),
        }
    prediction = heatmap >= threshold
    tp = float((prediction & gt).sum())
    fp = float((prediction & ~gt).sum())
    fn = float((~prediction & gt).sum())
    mass = float(heatmap.sum())
    return {
        "gt_pixels": float(gt.sum()),
        "mass_on_gt": float(heatmap[gt].sum() / mass) if mass > 1e-12 else 0.0,
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "iou": tp / max(tp + fp + fn, 1.0),
    }


def atom_auroc(
    records: List[Dict[str, object]],
    atom_id: str,
    target_group: str,
) -> float:
    labels, scores = [], []
    for record in records:
        group = record["group"]
        if target_group == "all":
            if group == "good":
                labels.append(0)
                scores.append(1.0 - float(record["atom_truths"][atom_id]))
            else:
                labels.append(1)
                scores.append(1.0 - float(record["atom_truths"][atom_id]))
        elif group == "good":
            labels.append(0)
            scores.append(1.0 - float(record["atom_truths"][atom_id]))
        elif group == target_group:
            labels.append(1)
            scores.append(1.0 - float(record["atom_truths"][atom_id]))
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def mean_std(values: List[float]) -> tuple[float, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std())


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose SymbolicAD intermediate outputs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", default="juice_bottle")
    parser.add_argument("--output-dir", default="1103results")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--vis-per-group", type=int, default=3)
    parser.add_argument("--enable-pro", action="store_true")
    parser.add_argument("--force-rules", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    category = args.category
    config.category = category
    config.mode = "baseline_logic"
    if args.shot is not None:
        config.shot = args.shot
    if args.device:
        config.model.device = args.device
    if args.batch_size is not None:
        config.model.batch_size = args.batch_size

    layout = RunLayout.create(Path(args.output_dir), config)
    output_dir = layout.root
    config.paths.cache_dir = args.cache_dir or str(output_dir / "cache")
    config.paths.results_dir = str(output_dir)
    for name in ["rules", "support", "clip_alignment", "grounding", "logic", "metrics", "plots", "visualizations"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    config.output.save_visualizations = False
    config.output.save_maps = False
    config.output.save_traces = False
    config.output.save_predictions = True
    config.evaluation.enable_pro = bool(args.enable_pro)

    class_root = config.data_root / category
    support_paths = select_support(class_root, config.shot, config.support_seed)
    support_images = [load_image(path) for path in support_paths]
    (output_dir / "support" / "support_images.txt").write_text(
        "\n".join(str(path) for path in support_paths) + "\n", encoding="utf-8"
    )

    # --------------------------------------------------------------- rule banks
    # Qwen3-VL inspects the support images and emits a structured description.
    # The deterministic builder converts it into atoms; nothing category-specific
    # is hand-written here.
    rule_kb = load_or_generate_rules(
        category,
        config.rule_cache_dir,
        force=args.force_rules,
        support_images=support_images,
        config=config,
    )
    qwen_raw_path = (
        config.cache_dir
        / config.qwen.cache_subdir
        / f"{category}_k{config.shot}.raw.txt"
    )
    if qwen_raw_path.is_file():
        (output_dir / "rules" / "01_qwen_raw_response.txt").write_text(
            qwen_raw_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    description = rule_kb.metadata.get("description")
    if description is not None:
        dump_json(
            output_dir / "rules" / "02_qwen_normal_structure.json", description
        )
    dump_json(output_dir / "rules" / "03_rule_bank_raw.json", rule_kb.to_dict())
    rule_kb, structural_report = validate_rule_kb(rule_kb)
    dump_json(
        output_dir / "rules" / "03b_structural_validation.json",
        structural_report.to_dict(),
    )

    backbone = CLIPBackbone(
        model_name=config.model.clip_model,
        device=config.model.device,
        image_size=config.model.image_size,
    )
    baseline = build_baseline(backbone, support_images, category, config)

    # ----------------------------------------------------------- support memory
    support_memory = build_support_memory(
        backbone, support_images, rule_kb, config, baseline=baseline
    )
    compiled_kb, compile_report = compile_rule_kb(rule_kb, support_memory, config)
    rule_kb = compiled_kb
    support_report = validate_against_support(rule_kb, support_memory)
    dump_json(
        output_dir / "rules" / "04_final_static_rule_kb.json", rule_kb.to_dict()
    )
    dump_json(
        output_dir / "rules" / "05_support_validation.json",
        support_report.to_dict(),
    )
    dump_json(
        output_dir / "rules" / "06_compile_report.json",
        compile_report.to_dict(),
    )
    dump_json(
        output_dir / "rules" / "07_rule_tree.json",
        rule_kb.formula.to_dict(),
    )

    component_rows = []
    for name, memory in support_memory.components.items():
        component_rows.append(
            {
                "component": name,
                "prompts": " | ".join(rule_kb.prompts_for(name)),
                "prototypes": int(memory.prototypes.shape[0]),
                "expected_count": memory.expected_count,
                "count_std": memory.count_std,
                "q_low": memory.score_q_low,
                "q_high": memory.score_q_high,
                "expected_mask_area": float(memory.expected_mask.mean()),
                "slots": int(memory.slot_masks.shape[0]),
            }
        )
    write_csv(output_dir / "support" / "component_memory.csv", component_rows)
    relation_rows = []
    for key, memory in support_memory.relations.items():
        relation_rows.append(
            {
                "key": key,
                "relation": memory.relation,
                "component_a": memory.component_a,
                "component_b": memory.component_b,
                "sample_count": memory.sample_count,
                "mean_geometry": np.asarray(memory.mean_geometry).tolist(),
                "has_pair_prototype": memory.pair_prototype is not None,
            }
        )
    write_csv(output_dir / "support" / "relation_memory.csv", relation_rows)
    dump_json(
        output_dir / "support" / "baseline_stats.json", support_memory.baseline_stats
    )
    np.savez_compressed(
        output_dir / "support" / "support_memory.npz",
        **{
            f"{name}_prototypes": memory.prototypes
            for name, memory in support_memory.components.items()
        },
    )

    # --------------------------------------------------------------- rule summary
    atom_rows = []
    for atom in rule_kb.atoms:
        atom_rows.append(
            {
                "atom_id": atom.atom_id,
                "expression": atom.expression,
                "kind": atom.kind,
                "weight": atom.weight,
                "enabled": atom.weight > 0,
                "normal_prompt": atom.normal_prompt,
                "violation_prompt": atom.violation_prompt,
            }
        )
    write_csv(output_dir / "rules" / "atom_table.csv", atom_rows)
    rule_summary = [
        f"# Rule diagnosis: {category}",
        "",
        "## Rule source",
        "",
        f"- Generator: **{rule_kb.metadata.get('source', 'unknown')}**.",
        "- Qwen3-VL inspected the normal support images and emitted a structured description.",
        "- A deterministic builder converted that description into atoms; no category-specific rules were hand-written.",
        "- The compiler then deduplicates, calibrates counts and builds a hierarchy.",
        "",
        "## Static rule bank after validation",
        "",
        "| atom | expression | weight | enabled | normal prompt | violation prompt |",
        "|---|---|---:|---|---|---|",
    ]
    for row in atom_rows:
        rule_summary.append(
            f"| {row['atom_id']} | `{row['expression']}` | {row['weight']:.2f} | "
            f"{row['enabled']} | {row['normal_prompt']} | {row['violation_prompt']} |"
        )
    rule_summary.extend(
        [
            "",
            f"- Structural validation issues: {structural_report.to_dict()['issues']}",
            f"- Support validation issues: {support_report.to_dict()['issues']}",
            f"- Disabled atoms: {support_report.to_dict()['dropped_atoms']}",
            "",
            "## Compiler report",
            "",
            f"- Tree depth: {compile_report.tree_depth}",
            f"- Removed duplicates: {compile_report.removed_duplicates}",
            f"- Removed redundant: {compile_report.removed_redundant}",
            f"- Repaired counts: {compile_report.repaired_counts}",
            f"- Added atoms: {compile_report.added_atoms}",
            f"- Coverage issues: {compile_report.coverage_issues}",
            f"- Tree: `{rule_kb.formula.to_dict()}`",
            "",
        ]
    )
    (output_dir / "rules" / "rule_summary.md").write_text(
        "\n".join(rule_summary), encoding="utf-8"
    )

    # --------------------------------------------------- support CLIP alignment
    support_encoded = backbone.encode_images(support_images)
    support_patches = support_encoded["patches"]
    alignment_summary = {}
    per_prompt_rows = []
    for symbol in rule_kb.components:
        name = symbol.name
        memory = support_memory.components[name]
        prompts = rule_kb.prompts_for(name)
        text_embeddings = encode_prompts_cached(
            backbone,
            prompts,
            cache_dir=config.text_embedding_dir,
            enabled=config.grounding.text_embedding_cache,
        )
        similarity = text_embeddings @ memory.prototypes.T
        text_centroid = text_embeddings.mean(axis=0)
        text_centroid /= max(float(np.linalg.norm(text_centroid)), 1e-8)
        proto_centroid = memory.global_prototype
        proto_centroid = proto_centroid / max(float(np.linalg.norm(proto_centroid)), 1e-8)
        centroid_cos = float(np.dot(text_centroid, proto_centroid))
        expected = memory.expected_mask

        corr_values, text_contrast, proto_contrast, topk_values = [], [], [], []
        text_map = text_similarity_map(backbone, support_patches, prompts)
        proto_map = prototype_similarity_map(support_patches, memory.prototypes)
        for index in range(text_map.shape[0]):
            corr_values.append(pearson(text_map[index], proto_map[index]))
            text_contrast.append(contrast(text_map[index], expected))
            proto_contrast.append(contrast(proto_map[index], expected))
            topk_values.append(topk_fraction_in_mask(text_map[index], expected))

        alignment_summary[name] = {
            "prompts": prompts,
            "text_proto_max_cos": float(similarity.max()),
            "text_proto_mean_cos": float(similarity.mean()),
            "text_proto_centroid_cos": centroid_cos,
            "text_proto_map_corr_support": mean_std(corr_values)[0],
            "text_contrast_support": mean_std(text_contrast)[0],
            "proto_contrast_support": mean_std(proto_contrast)[0],
            "text_topk_in_expected_support": mean_std(topk_values)[0],
            "num_prototypes": int(memory.prototypes.shape[0]),
        }
        for prompt_index, prompt in enumerate(prompts):
            per_prompt_rows.append(
                {
                    "component": name,
                    "prompt": prompt,
                    "max_proto_cos": float(similarity[prompt_index].max()),
                    "mean_proto_cos": float(similarity[prompt_index].mean()),
                    "best_prototype": int(similarity[prompt_index].argmax()),
                }
            )
    write_csv(
        output_dir / "clip_alignment" / "component_alignment_support.csv",
        [{"component": key, **value} for key, value in alignment_summary.items()],
    )
    write_csv(
        output_dir / "clip_alignment" / "per_prompt_alignment_support.csv",
        per_prompt_rows,
    )
    dump_json(
        output_dir / "clip_alignment" / "component_alignment_support.json",
        alignment_summary,
    )

    # ------------------------------------------------------------- query loop
    samples = collect_test_samples(class_root)
    if args.max_images > 0:
        samples = limit_per_group(samples, args.max_images, seed=config.seed)

    evaluator = CategoryEvaluator(config)
    records: List[Dict[str, object]] = []
    stored_maps: Dict[str, Dict[str, object]] = {}
    instance_rows = []
    query_alignment = defaultdict(
        lambda: {
            "corr": [],
            "text_contrast": [],
            "proto_contrast": [],
            "fused_contrast": [],
            "topk": [],
        }
    )

    batch_size = max(1, int(config.model.batch_size))
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        images = [load_image(sample.path) for sample in batch]
        encoded = backbone.encode_images(images)
        patches = encoded["patches"]
        base_maps, raw_scores = _baseline_maps(baseline, patches, support_memory.baseline_stats)
        component_maps = build_component_maps(
            backbone,
            patches,
            rule_kb,
            support_memory=support_memory,
            alpha=config.grounding.alpha,
            cache_dir=config.text_embedding_dir,
            text_cache_enabled=config.grounding.text_embedding_cache,
        )

        for offset, sample in enumerate(batch):
            image = images[offset]
            image_patches = patches[offset]
            gt_mask = load_gt_mask(sample.mask_path, (image.height, image.width))

            scene_graph = _scene_graph(
                offset, rule_kb, component_maps, support_memory, image_patches, config
            )
            atom_results = evaluate_atoms(
                scene_graph, rule_kb, support_memory, config, backbone=backbone
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
                image_patches,
                backbone,
                config,
                logic_result=logic_result,
            )
            base_map = base_maps[offset]
            final_map = fuse_maps(base_map, logic_map, config.fusion.lambda_logic)
            score_size = int(config.evaluation.pixel_eval_size)
            base_score = float(
                topk_mean(
                    cv2.resize(
                        base_map,
                        (score_size, score_size),
                        interpolation=cv2.INTER_LINEAR,
                    )[None, ...],
                    config.fusion.topk_percent,
                )[0]
            )
            logic_score = logic_image_score(
                cv2.resize(
                    logic_map,
                    (score_size, score_size),
                    interpolation=cv2.INTER_LINEAR,
                ),
                logic_result.root_truth,
                config.fusion.topk_percent,
            )
            final_score = final_image_score(base_score, logic_score, config.fusion.beta)

            evaluator.update(
                {"baseline": base_score, "logic": logic_score, "final": final_score},
                {"baseline": base_map, "logic": logic_map, "final": final_map},
                label=sample.label,
                group=sample.group,
                gt_mask=gt_mask,
            )

            logic_overlap = gt_overlap(logic_map, gt_mask, threshold=0.5)
            base_overlap = gt_overlap(base_map, gt_mask, threshold=0.5)
            record = {
                "image_id": f"{sample.group}_{sample.image_id}",
                "path": str(sample.path),
                "mask_path": str(sample.mask_path) if sample.mask_path else None,
                "sample_id": sample.image_id,
                "group": sample.group,
                "label": int(sample.label),
                "defect_type": sample.defect_type,
                "base_score": base_score,
                "logic_score": logic_score,
                "final_score": final_score,
                "baseline_raw_score": float(raw_scores[offset]),
                "root_truth": float(logic_result.root_truth),
                "counts": {name: len(items) for name, items in scene_graph.instances.items()},
                "atom_truths": {
                    atom_id: float(result.truth)
                    for atom_id, result in atom_results.items()
                },
                "atom_valid": {
                    atom_id: bool(result.valid) and result.weight > 0
                    for atom_id, result in atom_results.items()
                },
                "atom_evidence": {
                    atom_id: result.evidence for atom_id, result in atom_results.items()
                },
                "failed_atoms": logic_result.failed_atoms,
                "logic_map_mass": float(logic_map.sum()),
                "logic_map_max": float(logic_map.max()),
                "logic_mass_on_gt": logic_overlap["mass_on_gt"],
                "logic_precision": logic_overlap["precision"],
                "logic_recall": logic_overlap["recall"],
                "logic_iou": logic_overlap["iou"],
                "base_mass_on_gt": base_overlap["mass_on_gt"],
                "base_precision": base_overlap["precision"],
                "base_recall": base_overlap["recall"],
                "base_iou": base_overlap["iou"],
            }
            records.append(record)
            stored_maps[record["image_id"]] = {
                "component_maps": {
                    name: value[offset].copy()
                    for name, value in component_maps.items()
                },
                "logic": logic_map.copy(),
                "base": base_map.copy(),
                "final": final_map.copy(),
            }

            for instance_name, instances in scene_graph.instances.items():
                for instance in instances:
                    instance_rows.append(
                        {
                            "image_id": record["image_id"],
                            "group": sample.group,
                            "component": instance_name,
                            "instance_id": instance.instance_id,
                            "confidence": instance.confidence,
                            "bbox": " ".join(f"{value:.4f}" for value in instance.bbox),
                            "center": " ".join(f"{value:.4f}" for value in instance.center),
                            "area": instance.area,
                        }
                    )

            # query-time CLIP alignment for this image
            for symbol in rule_kb.components:
                name = symbol.name
                memory = support_memory.components[name]
                prompts = rule_kb.prompts_for(name)
                text_map = text_similarity_map(
                    backbone, image_patches[None, ...], prompts
                )[0]
                proto_map = prototype_similarity_map(
                    image_patches[None, ...], memory.prototypes
                )[0]
                fused = (
                    config.grounding.alpha * text_map
                    + (1.0 - config.grounding.alpha) * proto_map
                )
                expected = memory.expected_mask
                query_alignment[name]["corr"].append(pearson(text_map, proto_map))
                query_alignment[name]["text_contrast"].append(contrast(text_map, expected))
                query_alignment[name]["proto_contrast"].append(contrast(proto_map, expected))
                query_alignment[name]["fused_contrast"].append(contrast(fused, expected))
                query_alignment[name]["topk"].append(
                    topk_fraction_in_mask(text_map, expected)
                )

    # --------------------------------------------------------------- aggregates
    write_csv(output_dir / "grounding" / "instances.csv", instance_rows)

    summary_rows = []
    for record in records:
        row = {
            key: record[key]
            for key in [
                "image_id", "group", "label", "defect_type", "base_score",
                "logic_score", "final_score", "root_truth", "logic_map_mass",
                "logic_map_max",
            ]
        }
        row.update(
            {
                "logic_mass_on_gt": record["logic_mass_on_gt"],
                "logic_precision": record["logic_precision"],
                "logic_recall": record["logic_recall"],
                "logic_iou": record["logic_iou"],
                "base_mass_on_gt": record["base_mass_on_gt"],
                "base_precision": record["base_precision"],
                "base_recall": record["base_recall"],
                "base_iou": record["base_iou"],
            }
        )
        for name in rule_kb.components:
            row[f"count_{name.name}"] = record["counts"].get(name.name, 0)
        summary_rows.append(row)
    write_csv(output_dir / "grounding" / "per_image_summary.csv", summary_rows)

    atom_rows_per_image = []
    for record in records:
        for atom in rule_kb.atoms:
            atom_rows_per_image.append(
                {
                    "image_id": record["image_id"],
                    "group": record["group"],
                    "label": record["label"],
                    "atom_id": atom.atom_id,
                    "expression": atom.expression,
                    "enabled": record["atom_valid"].get(atom.atom_id, False),
                    "truth": record["atom_truths"].get(atom.atom_id, float("nan")),
                    "violation": 1.0 - record["atom_truths"].get(atom.atom_id, float("nan")),
                }
            )
    write_csv(output_dir / "logic" / "per_image_atoms.csv", atom_rows_per_image)
    dump_json(output_dir / "grounding" / "per_image_records.json", records)

    atom_group_rows = []
    atom_auroc_rows = []
    for atom in rule_kb.atoms:
        row = {"atom_id": atom.atom_id, "expression": atom.expression}
        for group in GROUPS:
            values = [
                float(record["atom_truths"][atom.atom_id])
                for record in records
                if record["group"] == group
            ]
            mean, std = mean_std(values)
            failure = float(np.mean([value < 0.5 for value in values])) if values else float("nan")
            row[f"{group}_mean"] = mean
            row[f"{group}_std"] = std
            row[f"{group}_failure_rate"] = failure
        atom_group_rows.append(row)
        atom_auroc_rows.append(
            {
                "atom_id": atom.atom_id,
                "expression": atom.expression,
                "auroc_logical": atom_auroc(records, atom.atom_id, "logical"),
                "auroc_structural": atom_auroc(records, atom.atom_id, "structural"),
                "auroc_all": atom_auroc(records, atom.atom_id, "all"),
            }
        )
    write_csv(output_dir / "logic" / "atom_truth_by_group.csv", atom_group_rows)
    write_csv(output_dir / "logic" / "atom_auroc.csv", atom_auroc_rows)

    score_summary = []
    for group in GROUPS:
        subset = [record for record in records if record["group"] == group]
        if not subset:
            continue
        score_summary.append(
            {
                "group": group,
                "n": len(subset),
                "base_score_mean": mean_std([record["base_score"] for record in subset])[0],
                "logic_score_mean": mean_std([record["logic_score"] for record in subset])[0],
                "final_score_mean": mean_std([record["final_score"] for record in subset])[0],
                "root_truth_mean": mean_std([record["root_truth"] for record in subset])[0],
                "logic_mass_on_gt_mean": mean_std(
                    [record["logic_mass_on_gt"] for record in subset]
                )[0],
                "logic_iou_mean": mean_std([record["logic_iou"] for record in subset])[0],
                "logic_map_mass_mean": mean_std(
                    [record["logic_map_mass"] for record in subset]
                )[0],
            }
        )
    write_csv(output_dir / "logic" / "score_summary_by_group.csv", score_summary)

    count_rows = []
    for symbol in rule_kb.components:
        name = symbol.name
        row = {"component": name}
        for group in GROUPS:
            values = [
                float(record["counts"].get(name, 0))
                for record in records
                if record["group"] == group
            ]
            mean, std = mean_std(values)
            row[f"{group}_mean"] = mean
            row[f"{group}_std"] = std
            row[f"{group}_distribution"] = dict(
                sorted(Counter(int(value) for value in values).items())
            )
        count_rows.append(row)
    write_csv(output_dir / "grounding" / "instance_count_by_group.csv", count_rows)

    query_alignment_rows = []
    for name, values in query_alignment.items():
        query_alignment_rows.append(
            {
                "component": name,
                "query_text_proto_map_corr": mean_std(values["corr"])[0],
                "query_text_contrast": mean_std(values["text_contrast"])[0],
                "query_proto_contrast": mean_std(values["proto_contrast"])[0],
                "query_fused_contrast": mean_std(values["fused_contrast"])[0],
                "query_text_topk_in_expected": mean_std(values["topk"])[0],
            }
        )
    write_csv(
        output_dir / "clip_alignment" / "component_alignment_query.csv",
        query_alignment_rows,
    )

    metrics = evaluator.compute()
    metrics["category"] = category
    metrics["shot"] = config.shot
    metrics["rule_source"] = str(rule_kb.metadata.get("source", "unknown"))
    metrics["qwen_raw_response"] = str(qwen_raw_path)
    metrics["support_validation"] = support_report.to_dict()
    dump_json(output_dir / "metrics" / "metrics_baseline_logic.json", metrics)
    metric_rows = []
    for group, payload in metrics["groups"].items():
        for variant in ["baseline", "logic", "final"]:
            item = payload[variant]
            metric_rows.append(
                {
                    "group": group,
                    "variant": variant,
                    "n": payload["n_samples"],
                    "image_auroc": item["image"].get("auroc"),
                    "image_ap": item["image"].get("ap"),
                    "image_f1max": item["image"].get("f1max"),
                    "pixel_auroc": item["pixel"].get("auroc"),
                    "pixel_ap": item["pixel"].get("ap"),
                    "pixel_f1max": item["pixel"].get("f1max"),
                    "aupro": item.get("pro", {}).get("aupro"),
                }
            )
    write_csv(output_dir / "metrics" / "metrics_by_group_variant.csv", metric_rows)
    evaluator.dump_predictions(output_dir / "metrics" / "predictions.npz")

    # ------------------------------------------------ reference-style outputs
    # imgs/{category}-{defect}-{sample}_{ori,gt,method}.jpg + csv/ + checkpoint/
    for record in records:
        image = load_image(record["path"])
        gt_mask = load_gt_mask(
            record.get("mask_path"), (image.height, image.width)
        )
        stored = stored_maps[record["image_id"]]
        save_reference_triplet(
            imgs_dir=layout.imgs_dir,
            category=category,
            defect_type=record["defect_type"],
            sample_id=record["sample_id"],
            image=image,
            gt_mask=gt_mask,
            final_map=stored["final"],
            method_name=config.output.method_name,
            image_size=config.output.image_size,
        )
    write_reference_csv({category: metrics}, layout.csv_dir / "results.csv")
    write_metrics_csv({category: metrics}, layout.csv_dir / "metrics_by_variant.csv")
    (layout.checkpoint_dir / "metrics_summary.json").write_text(
        json.dumps({category: metrics}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_run_checkpoint(layout, config, [category])

    # ------------------------------------------------------------------- plots
    _plot_atom_truths(output_dir / "plots" / "atom_truth_by_group.png", atom_group_rows)
    _plot_atom_auroc(output_dir / "plots" / "atom_auroc.png", atom_auroc_rows)
    _plot_counts(output_dir / "plots" / "instance_counts_by_group.png", count_rows)
    _plot_scores(output_dir / "plots" / "score_distributions.png", records)
    _plot_alignment(
        output_dir / "plots" / "clip_alignment.png",
        alignment_summary,
        {row["component"]: row for row in query_alignment_rows},
    )

    # ---------------------------------------------------------- visualisations
    _render_examples(
        output_dir / "visualizations",
        records,
        stored_maps,
        rule_kb,
        support_memory,
        config,
        per_group=args.vis_per_group,
    )

    _write_summary(
        output_dir / "SUMMARY.md",
        category,
        str(rule_kb.metadata.get("source", "unknown")),
        rule_kb,
        structural_report,
        support_report,
        alignment_summary,
        {row["component"]: row for row in query_alignment_rows},
        atom_auroc_rows,
        atom_group_rows,
        count_rows,
        score_summary,
        metric_rows,
    )

    backbone.close()
    print(f"[diagnose] wrote {output_dir}")


def _baseline_maps(baseline, patches, stats):
    scored = baseline.score_from_features(patches)
    raw = scored["anomaly_map"]
    if stats and "q_low" in stats and "q_high" in stats:
        maps = np.stack(
            [normalize_map(raw[index], stats["q_low"], stats["q_high"]) for index in range(raw.shape[0])],
            axis=0,
        )
    else:
        maps = np.clip(raw, 0.0, 1.0)
    return maps, scored["image_score"]


def _scene_graph(offset, rule_kb, component_maps, support_memory, patches, config):
    scene_graph = SceneGraph(component_maps={}, instances={}, patch_features=patches)
    for symbol in rule_kb.components:
        name = symbol.name
        raw = component_maps[name][offset]
        memory = support_memory.components[name]
        normalized = normalize_with_support(raw, memory.score_q_low, memory.score_q_high)
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


def _plot_atom_truths(path: Path, rows: List[Dict[str, object]]) -> None:
    labels = [row["atom_id"] for row in rows]
    good = [row["good_mean"] for row in rows]
    logical = [row["logical_mean"] for row in rows]
    structural = [row["structural_mean"] for row in rows]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
    width = 0.27
    axis.bar(x - width, good, width, label="good")
    axis.bar(x, logical, width, label="logical")
    axis.bar(x + width, structural, width, label="structural")
    axis.set_xticks(x, labels, rotation=45, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("mean truth")
    axis.legend()
    axis.set_title("Atom truth by group")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _plot_atom_auroc(path: Path, rows: List[Dict[str, object]]) -> None:
    labels = [row["atom_id"] for row in rows]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
    width = 0.27
    axis.bar(x - width, [row["auroc_logical"] for row in rows], width, label="logical")
    axis.bar(x, [row["auroc_structural"] for row in rows], width, label="structural")
    axis.bar(x + width, [row["auroc_all"] for row in rows], width, label="all")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(x, labels, rotation=45, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("AUROC (violation)")
    axis.legend()
    axis.set_title("Atom discrimination")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _plot_counts(path: Path, rows: List[Dict[str, object]]) -> None:
    names = [row["component"] for row in rows]
    figure, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5), squeeze=False)
    for axis, row in zip(axes[0], rows):
        data = []
        labels = []
        for group in GROUPS:
            distribution = row[f"{group}_distribution"]
            values = []
            for count, frequency in distribution.items():
                values.extend([int(count)] * int(frequency))
            if values:
                data.append(values)
                labels.append(group)
        if data:
            axis.boxplot(data, tick_labels=labels)
        axis.set_title(row["component"])
        axis.set_ylabel("instance count")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _plot_scores(path: Path, records: List[Dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for axis, key in zip(axes, ["base_score", "logic_score", "final_score"]):
        for group in GROUPS:
            values = [record[key] for record in records if record["group"] == group]
            if values:
                axis.hist(values, bins=20, alpha=0.5, label=group)
        axis.set_title(key)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _plot_alignment(path: Path, support: Dict[str, dict], query: Dict[str, dict]) -> None:
    names = list(support)
    x = np.arange(len(names))
    width = 0.25
    figure, axis = plt.subplots(figsize=(max(8, len(names) * 1.6), 4))
    axis.bar(
        x - width,
        [support[name]["text_proto_max_cos"] for name in names],
        width,
        label="text-prototype max cos (support)",
    )
    axis.bar(
        x,
        [support[name]["text_topk_in_expected_support"] for name in names],
        width,
        label="top-k in expected mask (support)",
    )
    axis.bar(
        x + width,
        [query.get(name, {}).get("query_text_topk_in_expected", 0.0) for name in names],
        width,
        label="top-k in expected mask (query)",
    )
    axis.set_xticks(x, names, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.legend()
    axis.set_title("CLIP rule-semantic alignment")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _select_examples(records: List[Dict[str, object]], per_group: int):
    selected = []
    for group in GROUPS:
        subset = [record for record in records if record["group"] == group]
        subset.sort(key=lambda item: float(item["logic_score"]), reverse=True)
        selected.extend(subset[:per_group])
    return selected


def _render_examples(
    output_dir: Path,
    records: List[Dict[str, object]],
    stored_maps: Dict[str, Dict[str, object]],
    rule_kb: RuleKB,
    support_memory,
    config,
    per_group: int = 3,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    components = [symbol.name for symbol in rule_kb.components]
    for record in _select_examples(records, per_group):
        image_id = record["image_id"]
        image = load_image(record["path"])
        rgb = np.asarray(image)
        gt_mask = load_gt_mask(record.get("mask_path"), (rgb.shape[0], rgb.shape[1]))
        maps = stored_maps[image_id]
        panels = 4 + len(components)
        figure, axes = plt.subplots(1, panels, figsize=(3.2 * panels, 3.4))
        axes[0].imshow(rgb)
        axes[0].set_title(Path(record["path"]).name)
        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("GT mask")
        for axis, name in zip(axes[2:2 + len(components)], components):
            memory = support_memory.components[name]
            raw = maps["component_maps"][name]
            normalized = normalize_with_support(raw, memory.score_q_low, memory.score_q_high)
            axis.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
            axis.set_title(name)
        axes[2 + len(components)].imshow(maps["logic"], cmap="jet", vmin=0, vmax=1)
        axes[2 + len(components)].set_title(
            f"logic (score={record['logic_score']:.2f})"
        )
        axes[3 + len(components)].imshow(rgb)
        axes[3 + len(components)].imshow(maps["final"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[3 + len(components)].set_title("final overlay")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_dir / f"{image_id}.png", dpi=110)
        plt.close(figure)


def _write_summary(
    path: Path,
    category: str,
    rule_source: str,
    rule_kb: RuleKB,
    structural_report,
    support_report,
    alignment_summary,
    query_alignment,
    atom_auroc_rows,
    atom_group_rows,
    count_rows,
    score_summary,
    metric_rows,
) -> None:
    lines = [
        f"# {category} intermediate diagnosis",
        "",
        "## 1. Rule source",
        "",
        f"- Generator: **{rule_source}**",
        "- Qwen3-VL inspected the normal support images and emitted a structured",
        "  description; the deterministic builder converted it into atoms.",
        "- No category-specific rules were hand-written for this run.",
        f"- Disabled atoms after support validation: `{support_report.to_dict()['dropped_atoms']}`",
        "",
        "## 2. Final static rule bank",
        "",
        "| atom | expression | weight | enabled |",
        "|---|---|---:|---|",
    ]
    for atom in rule_kb.atoms:
        lines.append(
            f"| {atom.atom_id} | `{atom.expression}` | {atom.weight:.2f} | {atom.weight > 0} |"
        )
    lines.extend(
        [
            "",
            f"- Structural issues: `{structural_report.to_dict()['issues']}`",
            f"- Support issues: `{support_report.to_dict()['issues']}`",
            "",
            "## 3. Does CLIP align with rule semantics?",
            "",
            "| component | text-proto max cos | text-proto centroid cos | "
            "support text top-k in expected | query text top-k in expected |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, value in alignment_summary.items():
        query = query_alignment.get(name, {})
        lines.append(
            f"| {name} | {value['text_proto_max_cos']:.3f} | "
            f"{value['text_proto_centroid_cos']:.3f} | "
            f"{value['text_topk_in_expected_support']:.3f} | "
            f"{query.get('query_text_topk_in_expected', float('nan')):.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: text-proto cosine measures whether CLIP text embeddings",
            "and normal visual prototypes live in the same direction; top-k-in-expected",
            "measures whether the text response actually lands on the support component",
            "region. Both should be high if the rule vocabulary is grounded.",
            "",
            "## 4. Rule discrimination",
            "",
            "| atom | AUROC logical | AUROC structural | AUROC all | good mean truth | logical mean truth |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    group_lookup = {row["atom_id"]: row for row in atom_group_rows}
    for row in atom_auroc_rows:
        group_row = group_lookup.get(row["atom_id"], {})
        lines.append(
            f"| {row['atom_id']} | {row['auroc_logical']:.3f} | "
            f"{row['auroc_structural']:.3f} | {row['auroc_all']:.3f} | "
            f"{group_row.get('good_mean', float('nan')):.3f} | "
            f"{group_row.get('logical_mean', float('nan')):.3f} |"
        )
    lines.extend(
        [
            "",
            "AUROC is computed from the violation score `1 - truth`; values near 0.5",
            "mean the atom does not separate normal from logical anomalies. Values",
            "below 0.5 mean the atom fires more on normal images than on anomalies.",
            "",
            "## 5. Instance counts by group",
            "",
            "| component | good mean | logical mean | structural mean |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in count_rows:
        lines.append(
            f"| {row['component']} | {row['good_mean']:.2f} | "
            f"{row['logical_mean']:.2f} | {row['structural_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Compare these with the expected counts in the rule bank. Large stable",
            "offsets mean CLIP connected-component extraction is over/under-splitting.",
            "",
            "## 6. Score summary",
            "",
            "| group | n | base | logic | final | root truth | logic mass on GT |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in score_summary:
        lines.append(
            f"| {row['group']} | {row['n']} | {row['base_score_mean']:.3f} | "
            f"{row['logic_score_mean']:.3f} | {row['final_score_mean']:.3f} | "
            f"{row['root_truth_mean']:.3f} | {row['logic_mass_on_gt_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 7. Metrics",
            "",
            "| group | variant | image AUROC | image AP | pixel AUROC | pixel AP |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        lines.append(
            f"| {row['group']} | {row['variant']} | {row['image_auroc']:.3f} | "
            f"{row['image_ap']:.3f} | {row['pixel_auroc']:.3f} | {row['pixel_ap']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
