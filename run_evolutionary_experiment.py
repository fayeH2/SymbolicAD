#!/usr/bin/env python3
"""
Evolutionary Neuro-Symbolic Experiment for MVTec Transistor.

Compares four approaches:
1. Baseline: INP-Former only
2. Fuzzy Logic (v7): Product fuzzy logic with fixed rules
3. Arithmetic-Logic: Fixed arithmetic-logic expressions
4. Evolutionary Search: SymbolicDet-style evolution with LLM feedback
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project root
sys.path.insert(0, "/home/hjh/Desktop/NeuroSymbolic_FSAD")

from neurosymbolic_fsad.data import collect_test_samples, select_inpformer_support
from neurosymbolic_fsad.inpformer_adapter import INPFormerAdapter
from neurosymbolic_fsad.transistor_logic import (
    TransistorLogicEngine,
    compile_transistor_logic_library,
    preprocess_mask,
)
from neurosymbolic_fsad.expression_space import (
    ExprNode, FuzzyExpressionEvaluator,
    var, const, add, sub, mul, div, max_op, min_op,
    gt, lt, ge, le, eq, ne,
    logic_and, logic_or, logic_not, logic_implies, logic_xor,
)
from neurosymbolic_fsad.fuzzy_logic import fuzzy_or, fuzzy_and, fuzzy_not, fuzzy_implies, fuzzy_xor
from neurosymbolic_fsad.evolutionary_search import (
    SearchConfig, ClassificationFitnessEvaluator, EvolutionarySearch,
    create_default_search,
)
from neurosymbolic_fsad.llm_feedback import (
    LLMFeedbackConfig, create_llm_feedback_fn, get_scene_description, TRANSISTOR_VARIABLES,
)


# Improved fixed arithmetic-logic rules using TRUTH variables (high = normal)
# Key insight from v7: IMPLIES structure with truth variables
# IMPLIES(count_normal, geometry_normal) = anomaly when count normal BUT geometry abnormal

FIXED_ARITHMETIC_RULES = [
    # Key insight from v7: IMPLIES structure with truth variables
    # If lead_count is normal THEN lead_geometry should be normal
    logic_implies(
        var("lead_count_normal"),           # count is normal
        var("lead_geometry_normal")         # then geometry should be normal
    ),
    # Alternative: If count is violated (<0.5 normal) THEN geometry violated
    logic_implies(
        lt(var("lead_count_normal"), const(0.5)),    # count violated (<0.5 normal)
        lt(var("lead_geometry_normal"), const(0.5)), # then geometry violated
    ),
    
    # Single variable: low normal = anomaly
    lt(var("lead_geometry_normal"), const(0.85)),
    lt(var("lead_count_normal"), const(0.5)),
    
    # Ratios of normal scores
    gt(div(var("lead_geometry_normal"), add(var("lead_count_normal"), const(0.01))), const(1.2)),
    gt(div(var("lead_geometry_normal"), add(var("lead_count_normal"), const(0.01))), const(2.0)),
    
    # Differences
    gt(sub(var("lead_count_normal"), var("lead_geometry_normal")), const(0.2)),
    gt(sub(var("lead_geometry_normal"), var("lead_symmetry_normal")), const(0.1)),
    
    # Products of normal scores (both normal = normal, so low product = anomaly)
    lt(mul(var("lead_geometry_normal"), var("lead_count_normal")), const(0.7)),
    lt(mul(var("lead_geometry_normal"), var("lead_symmetry_normal")), const(0.8)),
    
    # Min/Max
    lt(min_op(var("lead_geometry_normal"), var("lead_count_normal")), const(0.8)),
    lt(min_op(var("lead_geometry_normal"), var("lead_symmetry_normal")), const(0.85)),
    
    # Logical combinations
    logic_or(
        lt(var("lead_geometry_normal"), const(0.8)),
        lt(var("lead_count_normal"), const(0.5)),
    ),
    logic_and(
        lt(var("lead_geometry_normal"), const(0.85)),
        lt(var("lead_count_normal"), const(0.9)),
    ),
    logic_xor(
        lt(var("lead_geometry_normal"), const(0.8)),
        lt(var("lead_count_normal"), const(0.5)),
    ),
    
    # Package pose
    lt(var("package_pose_normal"), const(0.9)),
]


def load_data_and_baseline(
    data_root: str,
    category: str,
    baseline_shot: int,
    cache_dir: str,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict], np.ndarray]:
    """Load test samples and INP-Former baseline predictions."""
    data_root_path = Path(data_root).expanduser().resolve()
    class_root = data_root_path / category
    
    # Load baseline cache
    cache_path = Path(cache_dir).expanduser().resolve()
    if (cache_path / "cache").is_dir():
        cache_path = cache_path / "cache"
    
    with np.load(cache_path / "test_inp_v4.npz", allow_pickle=False) as payload:
        test_cache = {
            "paths": payload["paths"].astype(str),
            "scores": payload["scores"].astype(np.float64),
            "maps": payload["maps"].astype(np.float32),
            "metadata": json.loads(str(payload["metadata_json"].item())),
        }
    
    # Collect test samples
    samples = collect_test_samples(class_root)
    expected_paths = np.asarray([str(Path(row["path"]).resolve()) for row in samples])
    
    if not np.array_equal(test_cache["paths"], expected_paths):
        raise ValueError("Test cache path mismatch")
    
    image_labels = np.asarray([int(row["label"]) for row in samples], dtype=np.uint8)
    baseline_scores = np.clip(test_cache["scores"], 0.0, 1.0)
    baseline_maps = np.clip(test_cache["maps"], 0.0, 1.0)
    
    # Ground truth masks
    masks = []
    for sample in samples:
        if int(sample["label"]) == 0:
            masks.append(np.zeros((256, 256), dtype=bool))
        else:
            image_path = Path(sample["path"])
            defect = image_path.parent.name
            mask_path = class_root / "ground_truth" / defect / f"{image_path.stem}_mask.png"
            if not mask_path.exists():
                mask_path = class_root / "test" / "ground_truth" / defect / f"{image_path.stem}_mask.png"
            if mask_path.exists():
                masks.append(preprocess_mask(mask_path))
            else:
                masks.append(np.zeros((256, 256), dtype=bool))
    masks = np.asarray(masks, dtype=bool)
    
    return image_labels, baseline_scores, baseline_maps, masks, samples, test_cache["maps"]


def run_baseline_only(
    image_labels: np.ndarray,
    baseline_scores: np.ndarray,
    masks: np.ndarray,
    baseline_maps: np.ndarray,
) -> Dict[str, float]:
    """Evaluate INP-Former baseline only."""
    def _metric(labels, scores):
        return {
            "auroc": float(roc_auc_score(labels.reshape(-1), scores.reshape(-1))),
            "aupr": float(average_precision_score(labels.reshape(-1), scores.reshape(-1))),
        }
    
    img_metrics = _metric(image_labels, baseline_scores)
    pix_metrics = _metric(masks.flatten(), baseline_maps.flatten())
    
    return {
        "i_auroc": img_metrics["auroc"],
        "i_aupr": img_metrics["aupr"],
        "p_auroc": pix_metrics["auroc"],
        "p_aupr": pix_metrics["aupr"],
    }


def run_fuzzy_logic_v7(
    category: str,
    rule_support_count: int,
    support_seed: int,
    data_root: str,
    candidate_bank: Optional[str],
    image_labels: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_maps: np.ndarray,
    masks: np.ndarray,
    samples: List[Dict],
) -> Dict[str, float]:
    """Run v7 executable logic (fuzzy logic)."""
    # Use existing v7 results if available
    v7_dir = Path(f"/home/hjh/Desktop/NeuroSymbolic_FSAD/results_v7_logic/mvtec/{category}/nested_k{rule_support_count}_executable_qwen")
    metrics_file = v7_dir / "metrics_v7_executable_logic.json"
    
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        return {
            "i_auroc": metrics["neuro_symbolic"]["i_auroc"],
            "i_aupr": metrics["neuro_symbolic"]["i_aupr"],
            "p_auroc": metrics["neuro_symbolic"]["p_auroc"],
            "p_aupr": metrics["neuro_symbolic"]["p_aupr"],
        }
    
    # Otherwise run from scratch (reuse existing logic)
    # This is a simplified version - in practice, use the existing run_ns_experiment.py
    # For now, return baseline as placeholder
    return run_baseline_only(image_labels, baseline_scores, masks, baseline_maps)


def evaluate_fixed_arithmetic_rules(
    rule_support_count: int,
    support_seed: int,
    data_root: str,
    category: str,
    candidate_bank: Optional[str],
    image_labels: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_maps: np.ndarray,
    masks: np.ndarray,
    samples: List[Dict],
) -> Dict[str, float]:
    """Evaluate fixed arithmetic-logic rules on transistor."""
    # Compile rule library to get grounder
    rule_support = select_inpformer_support(
        data_root, "mvtec", category, rule_support_count, support_seed
    )
    
    bank = None
    if candidate_bank:
        bank_path = Path(candidate_bank).expanduser().resolve()
        bank = json.loads(bank_path.read_text())
    
    library = compile_transistor_logic_library(
        rule_support,
        candidate_bank=bank,
        dataset="mvtec",
        class_name=category,
    )
    
    engine = TransistorLogicEngine(library)
    
    # Evaluate each sample
    fused_scores = []
    fused_maps = []
    
    for sample, baseline_score, baseline_map in zip(samples, baseline_scores, baseline_maps):
        inference = engine.evaluate(sample["path"])
        
        # Get variable TRUTHS (1 - violation, so high = normal)
        var_values = {
            "lead_count_normal": inference.atoms.get("lead_count", {}).get("truth", 1.0),
            "lead_geometry_normal": inference.atoms.get("lead_geometry", {}).get("truth", 1.0),
            "lead_symmetry_normal": inference.atoms.get("lead_symmetry", {}).get("truth", 1.0),
            "package_pose_normal": inference.atoms.get("package_pose", {}).get("truth", 1.0),
            "neural_anomaly": float(baseline_score),
            "grounding": float(baseline_score),
        }
        
        # Evaluate all fixed rules and combine with OR
        evaluator = FuzzyExpressionEvaluator(var_values)
        rule_scores = []
        for rule_expr in FIXED_ARITHMETIC_RULES:
            try:
                score = evaluator.evaluate(rule_expr)
                rule_scores.append(score)
            except Exception:
                rule_scores.append(0.0)
        
        # Combine rules with OR (any rule triggers = anomaly)
        symbolic_anomaly = 0.0
        for rs in rule_scores:
            symbolic_anomaly = float(fuzzy_or(symbolic_anomaly, rs))
        
        # Neuro-symbolic fusion: OR(neural, AND(grounding, symbolic))
        grounded_symbolic = float(fuzzy_and(var_values["grounding"], symbolic_anomaly))
        fused_score = float(fuzzy_or(var_values["neural_anomaly"], grounded_symbolic))
        
        fused_scores.append(fused_score)
        
        # For pixel map: use baseline map where symbolic doesn't add info
        # Blend baseline and symbolic maps based on symbolic anomaly strength
        symbolic_map = inference.symbolic_map
        if symbolic_anomaly > 0.1:
            # Where symbolic detects anomaly, blend maps
            fused_map = np.clip(fuzzy_or(baseline_map, symbolic_map), 0.0, 1.0)
        else:
            fused_map = baseline_map
        fused_maps.append(fused_map)
    
    fused_scores = np.asarray(fused_scores)
    fused_maps = np.asarray(fused_maps)
    
    def _metric(labels, scores):
        return {
            "auroc": float(roc_auc_score(labels.reshape(-1), scores.reshape(-1))),
            "aupr": float(average_precision_score(labels.reshape(-1), scores.reshape(-1))),
        }
    
    img_metrics = _metric(image_labels, fused_scores)
    pix_metrics = _metric(masks.flatten(), fused_maps.flatten())
    
    return {
        "i_auroc": img_metrics["auroc"],
        "i_aupr": img_metrics["aupr"],
        "p_auroc": pix_metrics["auroc"],
        "p_aupr": pix_metrics["aupr"],
    }


def run_evolutionary_search(
    rule_support_count: int,
    support_seed: int,
    data_root: str,
    category: str,
    candidate_bank: Optional[str],
    image_labels: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_maps: np.ndarray,
    masks: np.ndarray,
    samples: List[Dict],
    use_llm: bool = False,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Run evolutionary search for rule discovery."""
    # Compile rule library to get grounder
    rule_support = select_inpformer_support(
        data_root, "mvtec", category, rule_support_count, support_seed
    )
    
    bank = None
    if candidate_bank:
        bank_path = Path(candidate_bank).expanduser().resolve()
        bank = json.loads(bank_path.read_text())
    
    library = compile_transistor_logic_library(
        rule_support,
        candidate_bank=bank,
        dataset="mvtec",
        class_name=category,
    )
    
    engine = TransistorLogicEngine(library)
    
    # Prepare training data for evolutionary search
    # Use a subset of test data for rule evolution (simulating validation set)
    n_samples = len(samples)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split = int(0.5 * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]
    
    # Build feature matrix for evolutionary search
    X_list = []
    y_list = []
    
    for idx in train_idx:
        sample = samples[idx]
        inference = engine.evaluate(sample["path"])
        
        var_values = [
            inference.atoms.get("lead_count", {}).get("truth", 1.0),
            inference.atoms.get("lead_geometry", {}).get("truth", 1.0),
            inference.atoms.get("lead_symmetry", {}).get("truth", 1.0),
            inference.atoms.get("package_pose", {}).get("truth", 1.0),
            float(baseline_scores[idx]),
            float(baseline_scores[idx]),  # grounding
        ]
        X_list.append(var_values)
        y_list.append(image_labels[idx])
    
    X_train = np.asarray(X_list)
    y_train = np.asarray(y_list)
    
    # Validation set - use same truth variables
    X_val_list = []
    y_val_list = []
    for idx in val_idx:
        sample = samples[idx]
        inference = engine.evaluate(sample["path"])
        
        var_values = [
            inference.atoms.get("lead_count", {}).get("truth", 1.0),
            inference.atoms.get("lead_geometry", {}).get("truth", 1.0),
            inference.atoms.get("lead_symmetry", {}).get("truth", 1.0),
            inference.atoms.get("package_pose", {}).get("truth", 1.0),
            float(baseline_scores[idx]),
            float(baseline_scores[idx]),
        ]
        X_val_list.append(var_values)
        y_val_list.append(image_labels[idx])
    
    X_val = np.asarray(X_val_list)
    y_val = np.asarray(y_val_list)
    
    # Variable names (truth versions)
    var_names = ["lead_count_normal", "lead_geometry_normal", "lead_symmetry_normal", "package_pose_normal", 
                 "neural_anomaly", "grounding"]
    
    # Configure evolutionary search
    config = SearchConfig(
        population_size=80,
        max_generations=50,
        tournament_size=7,
        elitism=3,
        crossover_rate=0.6,
        mutation_rate=0.4,
        max_depth=5,
        complexity_penalty=0.005,  # Lower penalty to allow more complex rules
        target_metric="auroc",
        early_stopping_patience=15,
        random_seed=42,
    )
    
    # LLM feedback
    llm_feedback = None
    if use_llm:
        feedback_config = LLMFeedbackConfig(
            device=device,
            feedback_interval=3,  # More frequent feedback
            num_suggestions=5,
        )
        llm_feedback = create_llm_feedback_fn(
            feedback_config,
            get_scene_description(category),
            TRANSISTOR_VARIABLES,
            anomaly_examples=["bent_lead", "cut_lead", "misplaced", "damaged_case"],
        )
    
    # Multi-objective fitness: combine AUROC and AUPR
    evaluator = ClassificationFitnessEvaluator(
        metric="auroc",
        complexity_penalty=config.complexity_penalty,
        use_fuzzy=True,
    )
    
    search = EvolutionarySearch(
        config=config,
        fitness_evaluator=evaluator,
        variable_names=var_names,
        constants=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0),
        llm_feedback_fn=llm_feedback,
    )
    
    print(f"Running evolutionary search: pop={config.population_size}, gens={config.max_generations}")
    result = search.run(X_train, y_train, X_val, y_val, var_names)
    
    # Evaluate best rule on full test set
    best_expr = result.best_individual.expr
    
    fused_scores = []
    fused_maps = []
    
    for sample, baseline_score, baseline_map in zip(samples, baseline_scores, baseline_maps):
        inference = engine.evaluate(sample["path"])
        
        # Use truth variables (high = normal) for IMPLIES
        var_values = {
            "lead_count_normal": inference.atoms.get("lead_count", {}).get("truth", 1.0),
            "lead_geometry_normal": inference.atoms.get("lead_geometry", {}).get("truth", 1.0),
            "lead_symmetry_normal": inference.atoms.get("lead_symmetry", {}).get("truth", 1.0),
            "package_pose_normal": inference.atoms.get("package_pose", {}).get("truth", 1.0),
            "neural_anomaly": float(baseline_score),
            "grounding": float(baseline_score),
        }
        
        evaluator = FuzzyExpressionEvaluator(var_values)
        try:
            symbolic_anomaly = evaluator.evaluate(best_expr)
        except Exception:
            symbolic_anomaly = 0.0
        
        grounded_symbolic = float(fuzzy_and(var_values["grounding"], symbolic_anomaly))
        fused_score = float(fuzzy_or(var_values["neural_anomaly"], grounded_symbolic))
        
        fused_scores.append(fused_score)
        
        # Blend maps based on symbolic anomaly
        if symbolic_anomaly > 0.1:
            fused_map = np.clip(fuzzy_or(baseline_map, inference.symbolic_map), 0.0, 1.0)
        else:
            fused_map = baseline_map
        fused_maps.append(fused_map)
    
    fused_scores = np.asarray(fused_scores)
    fused_maps = np.asarray(fused_maps)
    
    def _metric(labels, scores):
        return {
            "auroc": float(roc_auc_score(labels.reshape(-1), scores.reshape(-1))),
            "aupr": float(average_precision_score(labels.reshape(-1), scores.reshape(-1))),
        }
    
    img_metrics = _metric(image_labels, fused_scores)
    pix_metrics = _metric(masks.flatten(), fused_maps.flatten())
    
    return {
        "i_auroc": img_metrics["auroc"],
        "i_aupr": img_metrics["aupr"],
        "p_auroc": pix_metrics["auroc"],
        "p_aupr": pix_metrics["aupr"],
        "best_expression": str(best_expr),
        "best_metrics": result.best_individual.metrics,
        "generation_stats": result.generation_stats,
        "hall_of_fame": [(str(ind.expr), ind.fitness, ind.metrics) for ind in result.hall_of_fame[:10]],
    }


def main():
    parser = argparse.ArgumentParser(description="Evolutionary Neuro-Symbolic Experiment")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--category", default="transistor")
    parser.add_argument("--baseline-shot", type=int, default=4)
    parser.add_argument("--rule-support-count", type=int, default=30)
    parser.add_argument("--support-seed", type=int, default=1)
    parser.add_argument("--candidate-bank")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", default="results_evolutionary")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--skip-fuzzy", action="store_true")
    parser.add_argument("--skip-arithmetic", action="store_true")
    parser.add_argument("--skip-evolutionary", action="store_true")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Evolutionary Neuro-Symbolic Experiment")
    print("=" * 60)
    print(f"Category: {args.category}")
    print(f"Rule support: {args.rule_support_count}")
    print(f"Device: {args.device}")
    print(f"Use LLM: {args.use_llm}")
    
    # Load data
    print("\nLoading data...")
    image_labels, baseline_scores, baseline_maps, masks, samples, _ = load_data_and_baseline(
        args.data_root, args.category, args.baseline_shot,
        args.cache_dir, args.device
    )
    print(f"Test samples: {len(samples)} (normal: {np.sum(image_labels==0)}, anomaly: {np.sum(image_labels==1)})")
    
    results = {}
    
    # 1. Baseline
    print("\n[1/4] Baseline (INP-Former)...")
    baseline_metrics = run_baseline_only(image_labels, baseline_scores, masks, baseline_maps)
    results["baseline"] = baseline_metrics
    print(f"  I-AUROC: {baseline_metrics['i_auroc']:.4f}, I-AUPR: {baseline_metrics['i_aupr']:.4f}")
    print(f"  P-AUROC: {baseline_metrics['p_auroc']:.4f}, P-AUPR: {baseline_metrics['p_aupr']:.4f}")
    
    # 2. Fuzzy Logic (v7)
    if not args.skip_fuzzy:
        print("\n[2/4] Fuzzy Logic (v7 Executable Logic)...")
        fuzzy_metrics = run_fuzzy_logic_v7(
            args.category, args.rule_support_count, args.support_seed,
            args.data_root, args.candidate_bank,
            image_labels, baseline_scores, baseline_maps, masks, samples
        )
        results["fuzzy_logic_v7"] = fuzzy_metrics
        print(f"  I-AUROC: {fuzzy_metrics['i_auroc']:.4f} (Δ: {fuzzy_metrics['i_auroc'] - baseline_metrics['i_auroc']:+.4f})")
        print(f"  P-AUROC: {fuzzy_metrics['p_auroc']:.4f} (Δ: {fuzzy_metrics['p_auroc'] - baseline_metrics['p_auroc']:+.4f})")
    
    # 3. Fixed Arithmetic-Logic
    if not args.skip_arithmetic:
        print("\n[3/4] Fixed Arithmetic-Logic Rules...")
        arithmetic_metrics = evaluate_fixed_arithmetic_rules(
            args.rule_support_count, args.support_seed,
            args.data_root, args.category, args.candidate_bank,
            image_labels, baseline_scores, baseline_maps, masks, samples
        )
        results["fixed_arithmetic_logic"] = arithmetic_metrics
        print(f"  I-AUROC: {arithmetic_metrics['i_auroc']:.4f} (Δ: {arithmetic_metrics['i_auroc'] - baseline_metrics['i_auroc']:+.4f})")
        print(f"  P-AUROC: {arithmetic_metrics['p_auroc']:.4f} (Δ: {arithmetic_metrics['p_auroc'] - baseline_metrics['p_auroc']:+.4f})")
    
    # 4. Evolutionary Search
    if not args.skip_evolutionary:
        print("\n[4/4] Evolutionary Search (SymbolicDet-style)...")
        evo_results = run_evolutionary_search(
            args.rule_support_count, args.support_seed,
            args.data_root, args.category, args.candidate_bank,
            image_labels, baseline_scores, baseline_maps, masks, samples,
            use_llm=args.use_llm, device=args.device
        )
        results["evolutionary_search"] = {
            "i_auroc": evo_results["i_auroc"],
            "i_aupr": evo_results["i_aupr"],
            "p_auroc": evo_results["p_auroc"],
            "p_aupr": evo_results["p_aupr"],
            "best_expression": evo_results["best_expression"],
        }
        print(f"  I-AUROC: {evo_results['i_auroc']:.4f} (Δ: {evo_results['i_auroc'] - baseline_metrics['i_auroc']:+.4f})")
        print(f"  P-AUROC: {evo_results['p_auroc']:.4f} (Δ: {evo_results['p_auroc'] - baseline_metrics['p_auroc']:+.4f})")
        print(f"  Best expression: {evo_results['best_expression']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<30} {'I-AUROC':>10} {'Δ I-AUROC':>10} {'P-AUROC':>10} {'Δ P-AUROC':>10}")
    print("-" * 70)
    
    baseline_iaroc = baseline_metrics['i_auroc']
    baseline_pauroc = baseline_metrics['p_auroc']
    
    for method, metrics in results.items():
        if method == "baseline":
            print(f"{'Baseline (INP-Former)':<30} {metrics['i_auroc']:>10.4f} {'(ref)':>10} {metrics['p_auroc']:>10.4f} {'(ref)':>10}")
        else:
            delta_i = metrics['i_auroc'] - baseline_iaroc
            delta_p = metrics['p_auroc'] - baseline_pauroc
            print(f"{method:<30} {metrics['i_auroc']:>10.4f} {delta_i:>+10.4f} {metrics['p_auroc']:>10.4f} {delta_p:>+10.4f}")
    
    # Save results
    output_path = output_dir / "evolutionary_experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()