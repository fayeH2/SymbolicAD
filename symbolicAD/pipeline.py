"""End-to-end staged pipeline: support -> baseline -> grounding -> logic -> metrics."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
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
from .grounding.component_maps import build_component_maps
from .grounding.relation_grounder import calibrate_atom_results, evaluate_atoms
from .grounding.scene_builder import build_scene_graph
from .grounding.support_memory import (
    LOGIC_CALIBRATION_VERSION,
    SUPPORT_MEMORY_VERSION,
    SupportMemory,
    baseline_signature,
    rule_grounding_signature,
    support_image_signature,
)
from .localization.fusion import (
    final_image_score,
    fuse_maps,
    logic_image_score,
    normalize_logic_map,
    normalize_map,
)
from .localization.responsibility import build_logic_heatmap
from .models.baseline_adapter import build_baseline
from .models.clip_backbone import CLIPBackbone, IMAGE_PREPROCESS_VERSION
from .rules.executor import execute_formula
from .rules.schema import RuleKB
from .static_program import (
    StaticProgramPaths,
    load_grounding_memory,
    load_manifest,
    load_rule_program,
    load_symbolicdet_program,
)
from .visualization.render_results import ResultRenderer
from .visualization.result_layout import (
    RunLayout,
    save_run_checkpoint,
    write_metrics_csv,
    write_reference_csv,
)


def _support_memory_path(config: Config, category: str) -> Path:
    return StaticProgramPaths.for_category(config, category).support_memory


def _support_memory_is_compatible(
    memory: SupportMemory,
    rule_kb: RuleKB,
    backbone: CLIPBackbone,
    config: Config,
    support_images: Sequence,
) -> bool:
    """Reject stale memories after support, model, rule or grounding changes."""
    metadata = memory.metadata
    expected = {
        "shot": int(config.shot),
        "seed": int(config.support_seed),
        "threshold": float(config.grounding.component_threshold),
        "min_instance_area": int(config.grounding.min_instance_area),
        "morphology_kernel": int(config.grounding.morphology_kernel),
        "support_quantile_low": float(config.grounding.support_quantile_low),
        "support_quantile_high": float(config.grounding.support_quantile_high),
        "max_prototypes_per_component": int(
            config.grounding.max_prototypes_per_component
        ),
        "clip_model": str(backbone.model_name),
        "image_size": int(backbone.image_size),
        "grid_size": [int(backbone.grid_size), int(backbone.grid_size)],
        "feature_dim": int(backbone.embed_dim),
        "preprocess_version": IMAGE_PREPROCESS_VERSION,
        "support_memory_version": SUPPORT_MEMORY_VERSION,
        "support_signature": support_image_signature(support_images),
        "rule_grounding_signature": rule_grounding_signature(rule_kb),
        "logic_calibration_version": LOGIC_CALIBRATION_VERSION,
        "baseline_signature": baseline_signature(config),
        "component_names": [symbol.name for symbol in rule_kb.components],
        "attribute_atom_ids": [
            atom.atom_id for atom in rule_kb.atoms if atom.kind == "AttributeMatch"
        ],
    }
    for key, value in expected.items():
        cached = metadata.get(key)
        if isinstance(value, float):
            try:
                if not np.isclose(float(cached), value, rtol=0.0, atol=1e-8):
                    return False
            except (TypeError, ValueError):
                return False
        elif cached != value:
            return False
    return (
        memory.category == rule_kb.category
        and tuple(memory.grid_size) == tuple(expected["grid_size"])
        and int(memory.feature_dim) == int(expected["feature_dim"])
    )


def _fusion_weights(
    rule_kb: RuleKB,
    support_memory: Optional[SupportMemory],
    config: Config,
) -> tuple[float, float, Dict[str, float]]:
    """Scale fusion by support-estimated rule reliability, or abstain."""
    configured_lambda = float(config.fusion.lambda_logic)
    configured_beta = float(config.fusion.beta)
    info = {
        "lambda_logic": configured_lambda,
        "beta": configured_beta,
        "reliability": 1.0,
        "coverage": 1.0,
        "abstain": 0.0,
    }
    if not config.fusion.auto_from_support:
        return configured_lambda, configured_beta, info
    logic_noise = float(
        support_memory.logic_stats.get("q_high", 0.0)
        if support_memory is not None
        else 0.0
    )
    info["support_logic_noise"] = logic_noise
    if logic_noise > float(config.fusion.max_support_logic_noise):
        info.update({"lambda_logic": 0.0, "beta": 0.0, "abstain": 1.0})
        return 0.0, 0.0, info
    atoms = [atom for atom in rule_kb.atoms if atom.weight > 0]
    total = max(len(rule_kb.atoms), 1)
    info["coverage"] = len(atoms) / float(total)
    if not atoms or support_memory is None:
        info.update({"lambda_logic": 0.0, "beta": 0.0, "reliability": 0.0, "abstain": 1.0})
        return 0.0, 0.0, info
    values = []
    for atom in atoms:
        stats = support_memory.atom_stats.get(atom.atom_id)
        if stats:
            values.append(float(stats.get("q50", 1.0)))
        else:
            values.append(1.0)
    reliability = float(np.mean(values)) if values else 0.0
    info["reliability"] = reliability
    if (
        reliability < float(config.fusion.min_reliability)
        or info["coverage"] < float(config.fusion.min_coverage)
    ):
        info.update({"lambda_logic": 0.0, "beta": 0.0, "abstain": 1.0})
        return 0.0, 0.0, info
    info["lambda_logic"] = configured_lambda * reliability
    info["beta"] = configured_beta * reliability
    return info["lambda_logic"], info["beta"], info


def _baseline_scores_and_maps(
    baseline,
    encoded: Dict[str, np.ndarray],
    baseline_stats: Dict[str, float],
    images=None,
) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(baseline, "score_encoded"):
        scored = baseline.score_encoded(encoded, images=images)
    else:
        scored = baseline.score_from_features(encoded["patches"])
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
    raw_scores = np.asarray(scored["image_score"], dtype=np.float32)
    # Preserve the baseline ranking for image-level metrics.  Clipping scores
    # to support quantiles creates many exact 0/1 ties on anomalies and can
    # substantially reduce AUROC/AP.  Support calibration is still used for
    # the spatial map, where a shared visual scale is required for fusion.
    return maps, raw_scores


def run_category(
    config: Config,
    category: str,
    backbone: Optional[CLIPBackbone] = None,
) -> Dict[str, object]:
    config.category = category
    original_feature_source = config.model.feature_source
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
            feature_layers=config.model.feature_layers,
        )

    samples = collect_test_samples(class_root)
    samples = limit_per_group(samples, config.max_images_per_group, seed=config.seed)

    validation: Dict[str, object] = {}
    baseline = build_baseline(backbone, support_images, category, config)

    supplement_mode = config.mode in {"baseline_logic", "logic"}
    use_semantic_rules = bool(config.logic.enabled) and supplement_mode
    use_symbolicdet = bool(config.symbolicdet.enabled) and supplement_mode
    lambda_logic = 0.0
    beta = 0.0
    symbolicdet = None
    support_memory: Optional[SupportMemory] = None
    rule_kb: Optional[RuleKB] = None
    program_paths = StaticProgramPaths.for_category(config, category)
    program_manifest = None
    if use_semantic_rules or use_symbolicdet:
        program_manifest = load_manifest(program_paths)
        expected_manifest = {
            "category": category,
            "shot": int(config.shot),
            "support_seed": int(config.support_seed),
            "clip_model": str(config.model.clip_model),
            "image_size": int(config.model.image_size),
            "support_signature": support_image_signature(support_images),
        }
        mismatches = {
            key: (program_manifest.get(key), value)
            for key, value in expected_manifest.items()
            if program_manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"static program {program_paths.root} is incompatible: {mismatches}. "
                "Re-run scripts/compile_static_program.py"
            )
        validation["static_program"] = {
            "path": str(program_paths.root),
            "version": program_manifest.get("version"),
            "runtime_contract": program_manifest.get("runtime_contract"),
        }
    if use_symbolicdet:
        symbolicdet = load_symbolicdet_program(program_paths, config)
        symbolic_payload = symbolicdet.to_dict()
        validation["symbolicdet"] = {
            "enabled": True,
            "protocol": symbolic_payload["protocol"],
            "feature_source": symbolic_payload["feature_source"],
            "n_expressions": len(symbolicdet.expressions),
            "expert_weights": dict(symbolicdet.expert_weights),
        }
        symbolic_path = (
            Path(config.results_dir) / "checkpoint" / f"{category}_symbolicdet.json"
        )
        symbolic_path.parent.mkdir(parents=True, exist_ok=True)
        symbolic_path.write_text(
            json.dumps(symbolic_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    memory_path = _support_memory_path(config, category)
    if use_semantic_rules:
        rule_kb = load_rule_program(program_paths)
        support_memory = load_grounding_memory(program_paths)
        if not _support_memory_is_compatible(
            support_memory, rule_kb, backbone, config, support_images
        ):
            raise RuntimeError(
                f"compiled grounding memory is incompatible: {program_paths.root}. "
                "Inference never rebuilds it; re-run scripts/compile_static_program.py"
            )
        lambda_logic, beta, fusion_info = _fusion_weights(
            rule_kb, support_memory, config
        )
        fusion_info["source"] = "compiled_support_statistics"
        validation["fusion"] = fusion_info
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

    evaluator = CategoryEvaluator(config)
    renderer = ResultRenderer(config)
    batch_size = max(1, int(config.model.batch_size))

    # ---------------------------------------------------------------- pass 1
    records: list[Dict[str, object]] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        images = [load_image(sample.path) for sample in batch]
        # Baseline and logic share the exact same CLIP patch features.  This is
        # what keeps the semantic rule space and the visual anomaly map aligned.

        encoded = backbone.encode_images(images) # 视觉特征输出

        patches = encoded["patches"]
        base_maps, base_scores = _baseline_scores_and_maps(
            baseline, encoded, support_memory.baseline_stats, images=images
        )
        component_maps: Dict[str, np.ndarray] = {}
        if use_semantic_rules and rule_kb is not None and support_memory is not None:
            component_maps = build_component_maps(
                patches,
                rule_kb,
                support_memory,
            )
        for offset, sample in enumerate(batch):
            image = images[offset]
            atom_results = {}
            scene_summary: Dict[str, object] = {}
            symbolic_output = (
                symbolicdet.infer(image, patches[offset])
                if symbolicdet is not None
                else None
            )
            if use_semantic_rules and rule_kb is not None and support_memory is not None:
                scene_graph = build_scene_graph(
                    offset,
                    rule_kb,
                    component_maps,
                    support_memory,
                    patches[offset],
                    config,
                    image=image,
                )
                atom_results = evaluate_atoms(scene_graph, rule_kb, support_memory, config)
                atom_results = calibrate_atom_results(
                    atom_results, support_memory, config
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
            records.append(
                {
                    "sample": sample,
                    "base_map": base_maps[offset],
                    "base_score": float(base_scores[offset]),
                    "atom_results": atom_results,
                    "scene_summary": scene_summary,
                    "symbolicdet": symbolic_output,
                }
            )

    if use_semantic_rules:
        # All weights and abstention decisions come from the compiled normal
        # support program. Query images can never mutate the expression tree.
        validation["atom_reliability"] = {
            "mode": "compiled_support_only",
            "query_adaptation": False,
        }
        fusion_info = dict(validation.get("fusion", {}))
        if (
            float(fusion_info.get("abstain", 0.0)) < 0.5
            and float(fusion_info.get("support_logic_noise", 0.0))
            <= float(config.fusion.max_support_image_noise)
        ):
            beta = float(config.fusion.image_logic_weight) * float(
                fusion_info.get("reliability", 1.0)
            )
        else:
            beta = 0.0
        fusion_info["beta"] = float(beta)
        fusion_info["image_logic_weight"] = float(config.fusion.image_logic_weight)
        fusion_info["query_adaptation"] = False
        validation["fusion"] = fusion_info

    # ---------------------------------------------------------------- pass 2
    for record in records:
        sample = record["sample"]  # type: ignore[assignment]
        image = load_image(sample.path)  # type: ignore[union-attr]
        gt_mask = load_gt_mask(
            sample.mask_path,  # type: ignore[union-attr]
            (image.height, image.width),
        )
        base_map = np.asarray(record["base_map"], dtype=np.float32)
        base_score = float(record["base_score"])
        atom_results = record["atom_results"]  # type: ignore[assignment]
        scene_summary = record["scene_summary"]  # type: ignore[assignment]
        semantic_rule_map = np.zeros_like(base_map, dtype=np.float32)
        logic_map = np.zeros_like(base_map, dtype=np.float32)
        logic_result = None
        symbolic_output = record.get("symbolicdet")
        if use_semantic_rules and rule_kb is not None and support_memory is not None:
            logic_result = execute_formula(
                rule_kb.formula,
                atom_results,
                fail_threshold=config.logic.fail_threshold,
                epsilon=config.logic.and_epsilon,
            )
            semantic_rule_map, _ = build_logic_heatmap(
                atom_results,
                config,
                logic_result=logic_result,
            )
            semantic_rule_map = normalize_logic_map(
                semantic_rule_map,
                support_memory.logic_stats,
                config.fusion.map_epsilon,
            )
            score_size = int(config.evaluation.pixel_eval_size)
            if logic_result.active_failed_atom_ids:
                semantic_rule_score = logic_image_score(
                    cv2.resize(semantic_rule_map, (score_size, score_size)),
                    logic_result.root_truth,
                    config.fusion.topk_percent,
                )
            else:
                semantic_rule_score = 0.0
            final_map = fuse_maps(base_map, semantic_rule_map, lambda_logic)
            final_score = final_image_score(
                base_score,
                semantic_rule_score,
                beta,
                baseline_scale=float(config.fusion.image_logic_scale),
            )
        else:
            final_map = base_map
            semantic_rule_score = 0.0
            final_score = base_score

        symbolic_score = 0.0
        symbolic_trace = None
        if symbolic_output is not None:
            symbolic_map = cv2.resize(
                np.asarray(symbolic_output.anomaly_map, dtype=np.float32),
                (base_map.shape[1], base_map.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            symbolic_score = float(symbolic_output.score)
            symbolic_confidence = float(symbolic_output.confidence)
            symbolic_lambda = float(config.symbolicdet.fusion_weight) * symbolic_confidence
            final_map = fuse_maps(final_map, symbolic_map, symbolic_lambda)
            final_score += (
                float(config.symbolicdet.image_weight)
                * symbolic_confidence
                * symbolic_score
            )
            # The public logic panel/metric represents evidence that actually
            # participates in fusion. Raw abstained semantic evidence remains in
            # the trace, but cannot masquerade as an active localisation map.
            active_semantic_map = (
                semantic_rule_map
                if float(lambda_logic) > 0.0
                else np.zeros_like(semantic_rule_map)
            )
            active_semantic_score = semantic_rule_score if float(beta) > 0.0 else 0.0
            logic_map = 1.0 - (1.0 - active_semantic_map) * (1.0 - symbolic_map)
            logic_score = 1.0 - (1.0 - active_semantic_score) * (1.0 - symbolic_score)
            symbolic_trace = symbolic_output.trace()
        else:
            if use_semantic_rules:
                logic_map = (
                    semantic_rule_map
                    if float(lambda_logic) > 0.0
                    else np.zeros_like(semantic_rule_map)
                )
                logic_score = semantic_rule_score if float(beta) > 0.0 else 0.0
            else:
                logic_map = np.zeros_like(semantic_rule_map)
                logic_score = 0.0

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
            scores,
            maps,
            label=sample.label,  # type: ignore[union-attr]
            group=sample.group,  # type: ignore[union-attr]
            gt_mask=gt_mask,
        )
        image_id = f"{sample.group}_{sample.image_id}"  # type: ignore[union-attr]
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
                "image_path": str(sample.path),  # type: ignore[union-attr]
                "sample_id": sample.image_id,  # type: ignore[union-attr]
                "label": int(sample.label),  # type: ignore[union-attr]
                "group": sample.group,  # type: ignore[union-attr]
                "defect_type": sample.defect_type,  # type: ignore[union-attr]
                "baseline_raw_score": base_score,
                "semantic_rule_score": float(semantic_rule_score),
                "semantic_rules_active": bool(
                    float(lambda_logic) > 0.0 or float(beta) > 0.0
                ) if use_semantic_rules else False,
                "symbolicdet": symbolic_trace,
                **scene_summary,
            },
        )

    metrics = evaluator.compute()
    metrics["category"] = category
    metrics["mode"] = config.mode
    metrics["support"] = [str(path) for path in support_paths]
    metrics["validation"] = validation
    metrics["support_memory"] = str(memory_path) if use_semantic_rules else None
    metrics["protocol"] = {
        "name": "static_neurosymbolic_2shot",
        "shot": int(config.shot),
        "llm_at_inference": False,
        "query_adaptation": False,
        "visual_encoder": "CLIP",
        "static_program": str(program_paths.root) if supplement_mode else None,
        "limited_test_subset": bool(config.max_images_per_group > 0),
        "max_images_per_group": int(config.max_images_per_group),
    }
    all_group = metrics.get("groups", {}).get("all", {})
    baseline_image = all_group.get("baseline", {}).get("image", {}).get("auroc")
    logic_image = all_group.get("logic", {}).get("image", {}).get("auroc")
    final_image = all_group.get("final", {}).get("image", {}).get("auroc")
    baseline_pixel = all_group.get("baseline", {}).get("pixel", {}).get("auroc")
    final_pixel = all_group.get("final", {}).get("pixel", {}).get("auroc")
    metrics["logic_contribution"] = {
        "baseline_image_auroc": baseline_image,
        "logic_image_auroc": logic_image,
        "final_image_auroc": final_image,
        "image_auroc_delta": (
            float(final_image) - float(baseline_image)
            if final_image is not None and baseline_image is not None
            else None
        ),
        "pixel_auroc_delta": (
            float(final_pixel) - float(baseline_pixel)
            if final_pixel is not None and baseline_pixel is not None
            else None
        ),
        "effective_lambda_logic": float(
            validation.get("fusion", {}).get("lambda_logic", 0.0)
        ),
        "effective_image_logic_weight": float(
            validation.get("fusion", {}).get("beta", 0.0)
        ),
        "symbolicdet_enabled": bool(use_symbolicdet),
        "symbolicdet_fusion_weight": (
            float(config.symbolicdet.fusion_weight) if use_symbolicdet else 0.0
        ),
        "symbolicdet_image_weight": (
            float(config.symbolicdet.image_weight) if use_symbolicdet else 0.0
        ),
        "logic_abstained": bool(
            validation.get("fusion", {}).get("abstain", 0.0) >= 0.5
            and not use_symbolicdet
        ),
    }

    if config.output.save_predictions:
        prediction_dir = config.results_dir / "predictions"
        evaluator.dump_predictions(prediction_dir / category)

    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"metrics_{category}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if owns_backbone:
        backbone.close()
    config.model.feature_source = original_feature_source
    return metrics


def run_categories(
    config: Config,
    categories: Sequence[str],
) -> Dict[str, object]:
    # Hard runtime boundary: the deployment entry point cannot enable Qwen,
    # even when the shared offline configuration file has qwen.enabled=true.
    config.qwen.enabled = False
    layout = RunLayout.create(config.results_dir, config)
    config.paths.results_dir = str(layout.root)
    print(f"[SymbolicAD] run directory: {layout.root}", flush=True)

    backbone = CLIPBackbone(
        model_name=config.model.clip_model,
        device=config.model.device,
        image_size=config.model.image_size,
        feature_layers=config.model.feature_layers,
    )
    all_metrics: Dict[str, object] = {}
    try:
        for category in categories:
            print(f"[SymbolicAD] running {category} ({config.mode})", flush=True)
            all_metrics[category] = run_category(
                config,
                category,
                backbone=backbone,
            )
    finally:
        backbone.close()

    results_dir = layout.root
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n"
    (results_dir / "metrics_summary.json").write_text(metrics_json, encoding="utf-8")
    (layout.checkpoint_dir / "metrics_summary.json").write_text(
        metrics_json, encoding="utf-8"
    )
    write_reference_csv(all_metrics, layout.csv_dir / "results.csv")
    write_metrics_csv(all_metrics, layout.csv_dir / "metrics_by_variant.csv")
    _write_csv(all_metrics, layout.csv_dir / "metrics_per_class.csv")
    save_run_checkpoint(layout, config, categories)
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
