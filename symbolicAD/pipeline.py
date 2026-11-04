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
from .grounding.component_maps import build_component_maps, normalize_with_support
from .grounding.instance_extractor import (
    component_spatial_prior,
    extract_instances,
    image_to_lab_grid,
    select_instances,
)
from .grounding.relation_grounder import calibrate_atom_results, evaluate_atoms
from .grounding.scene_graph import SceneGraph
from .grounding.support_memory import (
    LOGIC_CALIBRATION_VERSION,
    SUPPORT_MEMORY_VERSION,
    SupportMemory,
    baseline_signature,
    build_support_memory,
    load_support_memory,
    rule_grounding_signature,
    save_support_memory,
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
from .models.baseline_adapter import build_baseline, otsu_separability
from .models.clip_backbone import CLIPBackbone, IMAGE_PREPROCESS_VERSION
from .rules.compiler import compile_rule_kb
from .rules.executor import execute_formula
from .rules.qwen_rule_generator import load_or_generate_rules
from .rules.schema import RuleKB
from .rules.validator import validate_against_support, validate_rule_kb
from .symbolicdet import SegmentFirstSymbolicDet
from .visualization.render_results import ResultRenderer
from .visualization.result_layout import (
    RunLayout,
    save_run_checkpoint,
    write_metrics_csv,
    write_reference_csv,
)


def _support_memory_path(config: Config, category: str) -> Path:
    return config.support_memory_dir / f"{category}_k{config.shot}.npz"


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
        "alpha": float(config.grounding.alpha),
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


def _build_scene_graph(
    image_index: int,
    rule_kb: RuleKB,
    component_maps: Dict[str, np.ndarray],
    support_memory: SupportMemory,
    patches: np.ndarray,
    config: Config,
    image=None,
) -> SceneGraph:
    scene_graph = SceneGraph(component_maps={}, instances={}, patch_features=patches)
    color_grid = (
        image_to_lab_grid(image, scene_graph.grid_size()) if image is not None else None
    )
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
        if image is not None:
            prior = component_spatial_prior(
                image,
                scene_graph.grid_size(),
                position=rule_kb.position_for(name),
                role=symbol.role,
            )
            normalized *= prior
            if "container" in symbol.role.lower():
                normalized = np.maximum(normalized, 0.6 * prior)
        scene_graph.component_maps[name] = normalized
        instances = extract_instances(
            component_map=normalized,
            patch_features=patches,
            concept=name,
            threshold=config.grounding.component_threshold,
            min_area=config.grounding.min_instance_area,
            morphology_kernel=config.grounding.morphology_kernel,
            color_grid=color_grid,
        )
        scene_graph.instances[name] = select_instances(
            instances, rule_kb.count_for(name)
        )
    return scene_graph


def _calibrate_logic_on_support(
    backbone: CLIPBackbone,
    support_images: Sequence,
    support_memory: SupportMemory,
    rule_kb: RuleKB,
    config: Config,
) -> SupportMemory:
    """Estimate every atom's normal-support truth distribution.

    A leave-one-out protocol is essential with 2-shot support: evaluating a
    support image with a memory that already contains the same image gives
    artificially perfect truths and hides the real query-time grounding noise.
    We therefore rebuild a temporary memory from the other support images and
    score the held-out normal image against it.
    """
    collected: Dict[str, list] = {}
    held_records: list[Dict[str, object]] = []
    count = len(support_images)
    for held in range(count):
        keep = [index for index in range(count) if index != held] or [held]
        calibration_memory = build_support_memory(
            backbone,
            [support_images[index] for index in keep],
            rule_kb,
            config,
            baseline=None,
        )
        encoded = backbone.encode_images([support_images[held]])
        patches = encoded["patches"]
        component_maps = build_component_maps(
            backbone,
            patches,
            rule_kb,
            support_memory=calibration_memory,
            alpha=config.grounding.alpha,
            cache_dir=config.text_embedding_dir,
            text_cache_enabled=config.grounding.text_embedding_cache,
        )
        scene_graph = _build_scene_graph(
            0,
            rule_kb,
            component_maps,
            calibration_memory,
            patches[0],
            config,
            image=support_images[held],
        )
        atom_results = evaluate_atoms(
            scene_graph, rule_kb, calibration_memory, config, backbone=backbone
        )
        for atom_id, result in atom_results.items():
            if result.valid and result.weight > 0:
                collected.setdefault(atom_id, []).append(float(result.truth))
        held_records.append(
            {"atom_results": atom_results, "patches": patches[0]}
        )

    atom_stats: Dict[str, Dict[str, float]] = {}
    for atom_id, values in collected.items():
        array = np.asarray(values, dtype=np.float64)
        atom_stats[atom_id] = {
            "q05": float(np.quantile(array, 0.05)),
            "q50": float(np.quantile(array, 0.50)),
            "q95": float(np.quantile(array, 0.95)),
            "mean": float(array.mean()),
            "std": float(array.std()),
            "n": int(array.size),
            "support_failure_rate": float(
                np.mean(array < float(config.logic.fail_threshold))
            ),
        }
    support_memory.atom_stats = atom_stats
    logic_maps = []
    for record in held_records:
        atom_results = record["atom_results"]  # type: ignore[assignment]
        for atom_id, result in atom_results.items():  # type: ignore[union-attr]
            stats = atom_stats.get(atom_id, {})
            if float(stats.get("support_failure_rate", 0.0)) > 0.5:
                result.weight = 0.0
                result.valid = False
        atom_results = calibrate_atom_results(
            atom_results, support_memory, config
        )
        logic_result = execute_formula(
            rule_kb.formula,
            atom_results,  # type: ignore[arg-type]
            fail_threshold=config.logic.fail_threshold,
            epsilon=config.logic.and_epsilon,
        )
        logic_map, _ = build_logic_heatmap(
            atom_results,  # type: ignore[arg-type]
            rule_kb,
            record["patches"],  # type: ignore[arg-type]
            backbone,
            config,
            logic_result=logic_result,
        )
        logic_maps.append(logic_map)
    if logic_maps:
        flat = np.concatenate([item.reshape(-1) for item in logic_maps])
        support_memory.logic_stats = {
            "q_low": float(np.quantile(flat, 0.50)),
            "q_high": float(max(np.quantile(flat, 0.99), 0.02)),
        }
    support_memory.metadata["logic_calibration_version"] = LOGIC_CALIBRATION_VERSION
    return support_memory


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


def _prepare_rule_bank(
    category: str,
    config: Config,
    support_memory: Optional[SupportMemory],
    force_rules: bool,
    support_images=None,
    rule_kb: Optional[RuleKB] = None,
) -> tuple[RuleKB, Dict[str, object]]:
    if rule_kb is None:
        rule_kb = load_or_generate_rules(
            category,
            config.rule_cache_dir,
            force=force_rules,
            support_images=support_images,
            config=config,
        )
    rule_kb, structural_report = validate_rule_kb(rule_kb)
    compile_report = None
    support_report = None
    if support_memory is not None and config.compiler.enabled:
        rule_kb, compile_report = compile_rule_kb(rule_kb, support_memory, config)
        support_report = validate_against_support(
            rule_kb,
            support_memory,
            disable_on_mismatch=bool(config.compiler.disable_inconsistent_atoms),
            min_attribute_support_score=float(
                config.compiler.min_attribute_support_score
            ),
            min_atom_support_truth=float(config.logic.min_atom_support_truth),
        )
        compiled_path = config.rule_cache_dir / f"{category}.compiled.json"
        rule_kb.save(compiled_path)
        if compile_report is not None:
            (compiled_path.with_suffix(".compile.json")).write_text(
                json.dumps(compile_report.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return rule_kb, {
        "structural": structural_report.to_dict(),
        "compile": compile_report.to_dict() if compile_report is not None else None,
        "support": support_report.to_dict() if support_report is not None else None,
    }


def _transductive_atom_reliability(
    records: Sequence[Dict[str, object]],
    fail_threshold: float,
) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Detect rule atoms that fire on almost every query image.

    Such atoms cannot discriminate anomalies from normal variation and are
    usually grounding failures (for example two labels whose CLIP masks are
    swapped).  The criterion uses no labels and no category-specific knowledge;
    it only asks whether an atom's violation rate is close to a constant.
    """
    counts: Dict[str, list] = {}
    for record in records:
        atom_results = record.get("atom_results", {})
        for atom_id, result in atom_results.items():  # type: ignore[union-attr]
            if not result.valid or result.weight <= 0:
                continue
            fail, total = counts.get(atom_id, [0, 0])
            total += 1
            if result.truth < fail_threshold:
                fail += 1
            counts[atom_id] = [fail, total]
    scales: Dict[str, float] = {}
    info: Dict[str, Dict[str, float]] = {}
    for atom_id, (fail, total) in counts.items():
        rate = float(fail) / max(int(total), 1)
        if rate >= 0.90:
            scale = 0.0
        elif rate >= 0.75:
            scale = (0.90 - rate) / 0.15
        else:
            scale = 1.0
        scales[atom_id] = float(scale)
        info[atom_id] = {
            "failure_rate": rate,
            "reliability_scale": float(scale),
            "n": int(total),
        }
    return scales, info


def _auto_select_feature_source(
    backbone: CLIPBackbone,
    support_images: Sequence,
    category: str,
    config: Config,
    samples: Sequence,
) -> tuple[str, Dict[str, object]]:
    """Choose native / projected / ensemble features on unlabeled queries.

    The criterion is Otsu separability of the image-score distribution. It uses
    no labels or category names, but it is still transductive test-time model
    selection and must never be used in the strict FSAD protocol.
    """
    candidates = ("native", "projected", "both")
    original = config.model.feature_source
    subset_size = max(12, int(config.model.auto_select_samples))
    step = max(1, len(samples) // subset_size)
    subset = list(samples[::step])[:subset_size]
    if not subset:
        return "both", {"selected": "both", "scores": {}, "n": 0}
    images = [load_image(sample.path) for sample in subset]
    encoded = backbone.encode_images(images)
    scores: Dict[str, float] = {}
    try:
        for candidate in candidates:
            config.model.feature_source = candidate
            candidate_baseline = build_baseline(
                backbone, support_images, category, config
            )
            if hasattr(candidate_baseline, "score_encoded"):
                scored = candidate_baseline.score_encoded(encoded, images=images)
            else:
                scored = candidate_baseline.score_from_features(encoded["patches"])
            scores[candidate] = otsu_separability(scored["image_score"])
    finally:
        config.model.feature_source = original
    best = max(scores, key=scores.get)
    # If native and the ensemble are statistically tied, keep the ensemble for
    # robustness.  If projected clearly wins (small-object categories), do not
    # override it with a mixed space.
    if best == "native" and scores.get("both", -1.0) >= scores["native"] - 0.02:
        best = "both"
    # Only leave the simple projected baseline when a more complex candidate is
    # clearly more separable.  The relative margin is stable across subset
    # sizes, unlike an absolute Otsu threshold.
    if scores[best] - scores["projected"] < float(config.model.auto_margin):
        best = "projected"
    info = {
        "selected": best,
        "scores": {key: float(value) for key, value in scores.items()},
        "n": len(subset),
    }
    print(
        f"[SymbolicAD] {category}: baseline feature source -> {best} "
        f"{info['scores']}",
        flush=True,
    )
    return best, info


def run_category(
    config: Config,
    category: str,
    backbone: Optional[CLIPBackbone] = None,
    force_support: bool = False,
    force_rules: bool = False,
) -> Dict[str, object]:
    config.category = category
    original_feature_source = config.model.feature_source
    if config.mode not in {"baseline", "baseline_logic", "logic"}:
        raise ValueError(f"unsupported mode: {config.mode}")
    if (
        str(config.model.feature_source).lower() == "auto"
        and not bool(config.evaluation.allow_transductive)
    ):
        raise ValueError(
            "model.feature_source='auto' inspects query images and is not "
            "allowed in strict FSAD; choose native/projected/both or enable "
            "evaluation.allow_transductive for an explicitly labelled ablation"
        )

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
    baseline_selection = None
    if (
        config.model.baseline == "clip_adaptive"
        and str(config.model.feature_source).lower() == "auto"
        and bool(config.evaluation.allow_transductive)
    ):
        selected, baseline_selection = _auto_select_feature_source(
            backbone, support_images, category, config, samples
        )
        config.model.feature_source = selected
    baseline = build_baseline(backbone, support_images, category, config)
    if baseline_selection is not None:
        validation["baseline_selection"] = baseline_selection

    use_logic = config.mode in {"baseline_logic", "logic"}
    use_symbolicdet = bool(config.symbolicdet.enabled) and use_logic
    symbolicdet = None
    if use_symbolicdet:
        symbolicdet = SegmentFirstSymbolicDet(config.symbolicdet).fit(support_images)
        symbolic_payload = symbolicdet.to_dict()
        validation["symbolicdet"] = {
            "enabled": True,
            "protocol": "normal_support_only",
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
    support_memory: Optional[SupportMemory] = None
    rule_kb: Optional[RuleKB] = None

    memory_path = _support_memory_path(config, category)
    if use_logic:
        rule_kb = load_or_generate_rules(
            category,
            config.rule_cache_dir,
            force=force_rules,
            support_images=support_images,
            config=config,
        )
        rule_kb, _ = validate_rule_kb(rule_kb)
        if memory_path.is_file() and not force_support:
            support_memory = load_support_memory(memory_path)
            if not _support_memory_is_compatible(
                support_memory, rule_kb, backbone, config, support_images
            ):
                print(
                    f"[SymbolicAD] support memory incompatible for {category}; rebuilding",
                    flush=True,
                )
                support_memory = None
            elif not support_memory.baseline_stats:
                support_memory.baseline_stats = baseline.support_statistics(
                    support_images,
                    low_quantile=config.grounding.support_quantile_low,
                    high_quantile=config.grounding.support_quantile_high,
                )
                save_support_memory(support_memory, memory_path)
        else:
            support_memory = None
        if support_memory is None:
            support_memory = build_support_memory(
                backbone, support_images, rule_kb, config, baseline=baseline
            )
            save_support_memory(support_memory, memory_path)
        if config.logic.support_calibration and not support_memory.atom_stats:
            support_memory = _calibrate_logic_on_support(
                backbone,
                support_images,
                support_memory,
                rule_kb,
                config,
            )
            save_support_memory(support_memory, memory_path)
        rule_kb, rule_validation = _prepare_rule_bank(
            category,
            config,
            support_memory,
            force_rules,
            support_images=support_images,
            rule_kb=rule_kb,
        )
        validation.update(rule_validation)
        if baseline_selection is not None:
            validation["baseline_selection"] = baseline_selection
        lambda_logic, beta, fusion_info = _fusion_weights(
            rule_kb, support_memory, config
        )
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
        encoded = backbone.encode_images(images)
        patches = encoded["patches"]
        base_maps, base_scores = _baseline_scores_and_maps(
            baseline, encoded, support_memory.baseline_stats, images=images
        )
        component_maps: Dict[str, np.ndarray] = {}
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
        for offset, sample in enumerate(batch):
            image = images[offset]
            atom_results = {}
            scene_summary: Dict[str, object] = {}
            symbolic_output = symbolicdet.infer(image) if symbolicdet is not None else None
            if use_logic and rule_kb is not None and support_memory is not None:
                scene_graph = _build_scene_graph(
                    offset,
                    rule_kb,
                    component_maps,
                    support_memory,
                    patches[offset],
                    config,
                    image=image,
                )
                atom_results = evaluate_atoms(
                    scene_graph, rule_kb, support_memory, config, backbone=backbone
                )
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

    reliability: Dict[str, float] = {}
    if use_logic and bool(config.evaluation.allow_transductive):
        reliability, reliability_info = _transductive_atom_reliability(
            records, float(config.logic.fail_threshold)
        )
        validation["atom_reliability"] = reliability_info
        disabled = [atom_id for atom_id, scale in reliability.items() if scale <= 0.0]
        if disabled:
            print(
                f"[SymbolicAD] {category}: disabled {len(disabled)} unstable "
                f"query-side atoms {disabled}",
                flush=True,
            )
        retained = [
            info
            for atom_id, info in reliability_info.items()
            if reliability.get(atom_id, 0.0) > 0.0
        ]
        effective_coverage = len(retained) / max(len(rule_kb.atoms), 1)
        max_failure = max(
            (float(info.get("failure_rate", 0.0)) for info in retained),
            default=0.0,
        )
        fusion_info = dict(validation.get("fusion", {}))
        fusion_info["effective_coverage"] = effective_coverage
        fusion_info["max_query_failure_rate"] = max_failure
        # If all atoms are constant on the query set, or too few survive, the
        # symbolic branch cannot add information and must abstain cleanly.
        image_beta = 0.0
        support_logic_noise = float(fusion_info.get("support_logic_noise", 0.0))
        if (
            effective_coverage >= float(config.fusion.min_coverage)
            and max_failure >= 0.02
            and retained
            and support_logic_noise <= float(config.fusion.max_support_image_noise)
        ):
            image_beta = float(config.fusion.image_logic_weight)
        if (
            effective_coverage < float(config.fusion.min_coverage)
            or max_failure < 0.02
            or not retained
        ):
            fusion_info.update({"lambda_logic": 0.0, "beta": 0.0, "abstain": 1.0})
            lambda_logic = 0.0
        else:
            fusion_info["lambda_logic"] = float(lambda_logic)
        beta = image_beta
        fusion_info["beta"] = float(beta)
        fusion_info["image_logic_weight"] = float(config.fusion.image_logic_weight)
        validation["fusion"] = fusion_info
    elif use_logic:
        # Strict FSAD: only normal support images may calibrate rules/fusion.
        # A rule violated by many query images may be the desired anomaly cue,
        # so query frequency must not be used to disable it.
        validation["atom_reliability"] = {
            "mode": "support_only",
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
    patch_shape = (
        int(support_memory.grid_size[0]),
        int(support_memory.grid_size[1]),
    )
    dummy_patches = np.zeros((1,) + patch_shape, dtype=np.float32)
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
        qwen_logic_map = np.zeros_like(base_map, dtype=np.float32)
        logic_map = np.zeros_like(base_map, dtype=np.float32)
        logic_result = None
        symbolic_output = record.get("symbolicdet")
        if use_logic and rule_kb is not None and support_memory is not None:
            for atom_id, result in atom_results.items():
                scale = reliability.get(atom_id, 1.0)
                if scale <= 0.0:
                    result.weight = 0.0
                    result.valid = False
                elif scale < 1.0:
                    result.weight = float(result.weight) * float(scale)
            logic_result = execute_formula(
                rule_kb.formula,
                atom_results,
                fail_threshold=config.logic.fail_threshold,
                epsilon=config.logic.and_epsilon,
            )
            qwen_logic_map, _ = build_logic_heatmap(
                atom_results,
                rule_kb,
                dummy_patches,
                backbone,
                config,
                logic_result=logic_result,
            )
            qwen_logic_map = normalize_logic_map(
                qwen_logic_map,
                support_memory.logic_stats,
                config.fusion.map_epsilon,
            )
            score_size = int(config.evaluation.pixel_eval_size)
            if logic_result.active_failed_atom_ids:
                qwen_logic_score = logic_image_score(
                    cv2.resize(qwen_logic_map, (score_size, score_size)),
                    logic_result.root_truth,
                    config.fusion.topk_percent,
                )
            else:
                qwen_logic_score = 0.0
            final_map = fuse_maps(base_map, qwen_logic_map, lambda_logic)
            final_score = final_image_score(
                base_score,
                qwen_logic_score,
                beta,
                baseline_scale=float(config.fusion.image_logic_scale),
            )
        else:
            final_map = base_map
            qwen_logic_score = 0.0
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
            # participates in fusion. Raw abstained Qwen evidence remains in
            # the trace, but cannot masquerade as an active localisation map.
            active_qwen_map = (
                qwen_logic_map if float(lambda_logic) > 0.0 else np.zeros_like(qwen_logic_map)
            )
            active_qwen_score = qwen_logic_score if float(beta) > 0.0 else 0.0
            logic_map = 1.0 - (1.0 - active_qwen_map) * (1.0 - symbolic_map)
            logic_score = 1.0 - (1.0 - active_qwen_score) * (1.0 - symbolic_score)
            symbolic_trace = symbolic_output.trace()
        else:
            if use_logic:
                logic_map = (
                    qwen_logic_map
                    if float(lambda_logic) > 0.0
                    else np.zeros_like(qwen_logic_map)
                )
                logic_score = qwen_logic_score if float(beta) > 0.0 else 0.0
            else:
                logic_map = np.zeros_like(qwen_logic_map)
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
                "qwen_logic_score": float(qwen_logic_score),
                "qwen_logic_active": bool(
                    float(lambda_logic) > 0.0 or float(beta) > 0.0
                ) if use_logic else False,
                "symbolicdet": symbolic_trace,
                **scene_summary,
            },
        )

    metrics = evaluator.compute()
    metrics["category"] = category
    metrics["mode"] = config.mode
    metrics["support"] = [str(path) for path in support_paths]
    metrics["validation"] = validation
    metrics["support_memory"] = str(memory_path) if use_logic else None
    metrics["protocol"] = {
        "name": (
            "transductive_unlabelled_query"
            if bool(config.evaluation.allow_transductive)
            else "strict_2shot_support_only"
        ),
        "shot": int(config.shot),
        "query_adaptation": bool(config.evaluation.allow_transductive),
        "feature_source": str(config.model.feature_source),
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
        evaluator.dump_predictions(prediction_dir / f"{category}.npz")

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
    force_support: bool = False,
    force_rules: bool = False,
) -> Dict[str, object]:
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
                force_support=force_support,
                force_rules=force_rules,
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
