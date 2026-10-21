"""Integration of explicit neuro-symbolic rule combinations with MVTec transistor experiment.

This script demonstrates how to use the new explicit rule combinations (XOR, AND, OR, IMPLIES)
on top of the existing executable logic baseline for anomaly detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from neurosymbolic_fsad.transistor_logic import (
    TransistorLogicEngine,
    compile_transistor_logic_library,
    execute_grounded_neural_symbolic,
    preprocess_rgb,
)
from neurosymbolic_fsad.neurosymbolic_rules import (
    NeuroSymbolicRule,
    RuleCombination,
    NeuroSymbolicEngine,
    execute_neuro_symbolic,
    create_xor_combinations,
    create_and_combinations,
    create_implication_chain,
)
from neurosymbolic_fsad.data import collect_test_samples, select_inpformer_support
from neurosymbolic_fsad.inpformer_adapter import INPFormerAdapter


def build_neuro_symbolic_rules(
    engine: TransistorLogicEngine,
    image_path: str,
) -> List[NeuroSymbolicRule]:
    """Build neuro-symbolic rules from TransistorLogicEngine evaluation."""
    inference = engine.evaluate(image_path)
    
    rules = []
    for atom_id, atom_data in inference.atoms.items():
        # truth_value: 1.0 = normal, 0.0 = violated
        truth = atom_data["truth"]
        violation = atom_data["violation"]
        
        rules.append(NeuroSymbolicRule(
            rule_id=atom_id,
            description=atom_data.get("predicate", atom_id),
            truth_value=truth,
            spatial_map=inference.symbolic_map if atom_id == "lead_geometry" else None,
            weight=atom_data.get("weight", 1.0),
        ))
    
    return rules


def run_neuro_symbolic_experiment(
    data_root: str,
    class_name: str = "transistor",
    baseline_shot: int = 4,
    rule_support_count: int = 30,
    support_seed: int = 1,
    candidate_bank: Optional[str] = None,
    prediction_cache_dir: Optional[str] = None,
    output_dir: str = "results_neuro_symbolic",
    combination_type: str = "xor_all",  # "xor_all", "and_all", "implication_chain", "custom"
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Run neuro-symbolic experiment with explicit rule combinations."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_root_path = Path(data_root).expanduser().resolve()
    class_root = data_root_path / class_name

    # Step 1: Compile the static rule library (same as v7)
    rule_support = select_inpformer_support(
        data_root, "mvtec", class_name,
        rule_support_count, support_seed,
    )
    
    bank = None
    if candidate_bank:
        bank_path = Path(candidate_bank).expanduser().resolve()
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
    
    library = compile_transistor_logic_library(
        rule_support,
        candidate_bank=bank,
        dataset="mvtec",
        class_name=class_name,
    )
    library_path = output / "normality_rules_v7_executable_logic.json"
    library_path.write_text(json.dumps(library, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Step 2: Load INP-Former predictions
    cache_dir = Path(prediction_cache_dir).expanduser().resolve()
    if (cache_dir / "cache").is_dir():
        cache_dir = cache_dir / "cache"
    
    with np.load(cache_dir / "test_inp_v4.npz", allow_pickle=False) as payload:
        test_cache = {
            "paths": payload["paths"].astype(str),
            "scores": payload["scores"].astype(np.float64),
            "maps": payload["maps"].astype(np.float32),
            "metadata": json.loads(str(payload["metadata_json"].item())),
        }

    # Step 3: Collect test samples
    samples = collect_test_samples(class_root)
    expected_paths = np.asarray([str(Path(row["path"]).resolve()) for row in samples])
    
    if not np.array_equal(test_cache["paths"], expected_paths):
        raise ValueError("Test cache path/order mismatch")

    # Step 4: Create neuro-symbolic engine
    engine = TransistorLogicEngine(library)
    
    # Get rule IDs from the compiled library
    rule_ids = [str(row["id"]) for row in library["rules"]]
    
    # Create combinations based on type
    if combination_type == "xor_all":
        combinations = create_xor_combinations(rule_ids)
    elif combination_type == "and_all":
        combinations = create_and_combinations(rule_ids)
    elif combination_type == "implication_chain":
        combinations = create_implication_chain(rule_ids)
    elif combination_type == "custom":
        # Custom combinations for transistor
        combinations = [
            RuleCombination("xor_count_geometry", "XOR", ["lead_count", "lead_geometry"], 
                          "Lead count XOR lead geometry"),
            RuleCombination("xor_geometry_symmetry", "XOR", ["lead_geometry", "lead_symmetry"],
                          "Lead geometry XOR lead symmetry"),
            RuleCombination("and_count_geometry", "AND", ["lead_count", "lead_geometry"],
                          "Lead count AND lead geometry"),
            RuleCombination("implies_count_geometry", "IMPLIES", ["lead_count", "lead_geometry"],
                          "Lead count implies lead geometry"),
        ]
    else:
        combinations = []

    # Step 5: Run evaluation
    neuro_symbolic_engine = NeuroSymbolicEngine(
        [],  # Rules will be built per image
        combinations,
    )
    
    image_labels = np.asarray([int(row["label"]) for row in samples], dtype=np.uint8)
    baseline_scores = np.clip(test_cache["scores"], 0.0, 1.0)
    baseline_maps = np.clip(test_cache["maps"], 0.0, 1.0)
    
    neuro_symbolic_scores = []
    neuro_symbolic_maps = []
    combination_scores_list = []
    
    for index, (sample, baseline_score, baseline_map) in enumerate(
        zip(samples, baseline_scores, baseline_maps)
    ):
        # Get rule truths from the static engine
        inference = engine.evaluate(sample["path"])
        
        # Build neuro-symbolic rules
        ns_rules = []
        rule_truths = {}
        rule_maps = {}
        for atom_id, atom_data in inference.atoms.items():
            truth = atom_data["truth"]
            ns_rules.append(NeuroSymbolicRule(
                rule_id=atom_id,
                description=atom_data.get("predicate", atom_id),
                truth_value=truth,
                spatial_map=None,  # We'll use the combined symbolic map
            ))
            rule_truths[atom_id] = truth
        
        # Use the symbolic map for spatial evaluation
        symbolic_map = inference.symbolic_map
        
        # Evaluate neuro-symbolic combinations
        result = neuro_symbolic_engine.evaluate(
            rule_truths,
            neural_anomaly=float(baseline_score),
            grounding_confidence=float(baseline_score),  # Use neural as grounding
            rule_maps={atom_id: symbolic_map for atom_id in rule_truths.keys()},
        )
        
        neuro_symbolic_scores.append(result["final_score"])
        neuro_symbolic_maps.append(result["spatial_map"] if result["spatial_map"] is not None else baseline_map)
        combination_scores_list.append(result["combination_scores"])

    neuro_symbolic_scores = np.asarray(neuro_symbolic_scores, dtype=np.float64)
    neuro_symbolic_maps = np.asarray(neuro_symbolic_maps, dtype=np.float32)
    
    # Compute metrics
    from sklearn.metrics import average_precision_score, roc_auc_score
    
    def _metric_pair(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        return {
            "auroc": float(roc_auc_score(labels, scores)),
            "aupr": float(average_precision_score(labels, scores)),
        }
    
    baseline_metrics = _metric_pair(image_labels, baseline_scores)
    baseline_pixel_metrics = _metric_pair(
        np.concatenate([m.flatten() for m in baseline_maps]),
        np.concatenate([m.flatten() for m in baseline_maps])
    ) if len(baseline_maps) > 0 else {"auroc": 0.0, "aupr": 0.0}
    
    neuro_symbolic_metrics = _metric_pair(image_labels, neuro_symbolic_scores)
    
    # Pixel-level metrics
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
                from neurosymbolic_fsad.transistor_logic import preprocess_mask
                masks.append(preprocess_mask(mask_path))
            else:
                masks.append(np.zeros((256, 256), dtype=bool))
    
    masks = np.asarray(masks, dtype=bool)
    
    neuro_symbolic_pixel_metrics = _metric_pair(
        masks.flatten(),
        neuro_symbolic_maps.flatten()
    )
    
    result = {
        "schema_version": "1.0-neuro-symbolic-explicit-combinations",
        "dataset": "mvtec",
        "class_name": class_name,
        "baseline_shot": baseline_shot,
        "rule_support_count": rule_support_count,
        "support_seed": support_seed,
        "combination_type": combination_type,
        "accepted_rule_ids": rule_ids,
        "combinations": [{"name": c.name, "operator": c.operator, "rule_ids": c.rule_ids} for c in combinations],
        "baseline_i_auroc": baseline_metrics["auroc"],
        "baseline_i_aupr": baseline_metrics["aupr"],
        "neuro_symbolic_i_auroc": neuro_symbolic_metrics["auroc"],
        "neuro_symbolic_i_aupr": neuro_symbolic_metrics["aupr"],
        "delta_i_auroc": neuro_symbolic_metrics["auroc"] - baseline_metrics["auroc"],
        "delta_i_aupr": neuro_symbolic_metrics["aupr"] - baseline_metrics["aupr"],
        "neuro_symbolic_p_auroc": neuro_symbolic_pixel_metrics["auroc"],
        "neuro_symbolic_p_aupr": neuro_symbolic_pixel_metrics["aupr"],
    }
    
    output_path = output / f"metrics_neuro_symbolic_{combination_type}.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Run neuro-symbolic experiment with explicit rule combinations")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--class-name", default="transistor")
    parser.add_argument("--baseline-shot", type=int, default=4)
    parser.add_argument("--rule-support-count", type=int, default=30)
    parser.add_argument("--support-seed", type=int, default=1)
    parser.add_argument("--candidate-bank")
    parser.add_argument("--prediction-cache-dir", required=True)
    parser.add_argument("--output-dir", default="results_neuro_symbolic")
    parser.add_argument("--combination-type", default="xor_all", 
                       choices=["xor_all", "and_all", "implication_chain", "custom"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    result = run_neuro_symbolic_experiment(
        data_root=args.data_root,
        class_name=args.class_name,
        baseline_shot=args.baseline_shot,
        rule_support_count=args.rule_support_count,
        support_seed=args.support_seed,
        candidate_bank=args.candidate_bank,
        prediction_cache_dir=args.prediction_cache_dir,
        output_dir=args.output_dir,
        combination_type=args.combination_type,
        device=args.device,
    )
    
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()