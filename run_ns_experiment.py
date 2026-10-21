#!/usr/bin/env python3
"""Run neuro-symbolic combinations experiment on MVTec transistor.

This runs explicit rule combinations (XOR, AND, IMPLIES) on top of the
existing v7 executable logic baseline.
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from neurosymbolic_fsad.transistor_logic import (
    TransistorLogicEngine,
    compile_transistor_logic_library,
    execute_grounded_neural_symbolic,
    preprocess_rgb,
    preprocess_mask,
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


def _metric_pair(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
    }


def run_experiment(
    data_root: str,
    class_name: str = "transistor",
    baseline_shot: int = 4,
    rule_support_count: int = 30,
    support_seed: int = 1,
    candidate_bank: Optional[str] = None,
    prediction_cache_dir: Optional[str] = None,
    output_dir: str = "results_neuro_symbolic",
    combination_type: str = "xor_all",
    device: str = "cuda:0",
) -> Dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_root_path = Path(data_root).expanduser().resolve()
    class_root = data_root_path / class_name

    # Step 1: Load or compile the static rule library
    if candidate_bank:
        bank_path = Path(candidate_bank).expanduser().resolve()
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        rule_support = []
        for row in sorted(bank.get("support", []), key=lambda r: int(r["sample_index"])):
            rule_support.append(Path(str(row["image_path"])).expanduser().resolve())
    else:
        rule_support = select_inpformer_support(
            data_root, "mvtec", class_name,
            rule_support_count, support_seed,
        )
        bank = None

    library = compile_transistor_logic_library(
        rule_support,
        candidate_bank=bank,
        dataset="mvtec",
        class_name=class_name,
    )
    rule_ids = [str(row["id"]) for row in library["rules"]]
    print(f"Accepted rules: {rule_ids}")

    # Step 2: Load INP-Former predictions from cache
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

    # Step 4: Create engine and combinations
    engine = TransistorLogicEngine(library)

    if combination_type == "xor_all":
        combinations = create_xor_combinations(rule_ids)
    elif combination_type == "and_all":
        combinations = create_and_combinations(rule_ids)
    elif combination_type == "implication_chain":
        combinations = create_implication_chain(rule_ids)
    elif combination_type == "custom":
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

    print(f"Combinations: {[f'{c.name}({c.operator})' for c in combinations]}")

    neuro_symbolic_engine = NeuroSymbolicEngine([], combinations)

    image_labels = np.asarray([int(row["label"]) for row in samples], dtype=np.uint8)
    baseline_scores = np.clip(test_cache["scores"], 0.0, 1.0)
    baseline_maps = np.clip(test_cache["maps"], 0.0, 1.0)

    neuro_symbolic_scores = []
    neuro_symbolic_maps = []
    combination_scores_list = []

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

    for index, (sample, baseline_score, baseline_map) in enumerate(
        zip(samples, baseline_scores, baseline_maps)
    ):
        inference = engine.evaluate(sample["path"])

        # Build rule truths and get individual spatial maps
        rule_truths = {}
        rule_maps = {}
        for atom_id, atom_data in inference.atoms.items():
            rule_truths[atom_id] = atom_data["truth"]
            if atom_id in inference.rule_maps:
                rule_maps[atom_id] = inference.rule_maps[atom_id]

        # Evaluate neuro-symbolic combinations
        result = neuro_symbolic_engine.evaluate(
            rule_truths,
            neural_anomaly=float(baseline_score),
            grounding_confidence=float(baseline_score),
            rule_maps=rule_maps,
        )

        neuro_symbolic_scores.append(result["final_score"])
        neuro_symbolic_maps.append(
            result["spatial_map"] if result["spatial_map"] is not None else baseline_map
        )
        combination_scores_list.append(result["combination_scores"])

    neuro_symbolic_scores = np.asarray(neuro_symbolic_scores, dtype=np.float64)
    neuro_symbolic_maps = np.asarray(neuro_symbolic_maps, dtype=np.float32)

    # Compute metrics
    baseline_metrics = _metric_pair(image_labels, baseline_scores)
    neuro_symbolic_metrics = _metric_pair(image_labels, neuro_symbolic_scores)
    neuro_symbolic_pixel_metrics = _metric_pair(masks.flatten(), neuro_symbolic_maps.flatten())

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
        "baseline_p_auroc": float(roc_auc_score(masks.flatten(), baseline_maps.flatten())),
        "baseline_p_aupr": float(average_precision_score(masks.flatten(), baseline_maps.flatten())),
        "delta_p_auroc": neuro_symbolic_pixel_metrics["auroc"] - float(roc_auc_score(masks.flatten(), baseline_maps.flatten())),
        "delta_p_aupr": neuro_symbolic_pixel_metrics["aupr"] - float(average_precision_score(masks.flatten(), baseline_maps.flatten())),
    }

    output_path = output / f"metrics_neuro_symbolic_{combination_type}.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run neuro-symbolic combinations experiment")
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

    result = run_experiment(
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

    # Print summary
    print("\n" + "="*60)
    print(f"COMBINATION TYPE: {args.combination_type}")
    print("="*60)
    print(f"Baseline I-AUROC:  {result['baseline_i_auroc']:.6f}")
    print(f"NS I-AUROC:        {result['neuro_symbolic_i_auroc']:.6f}")
    print(f"Delta I-AUROC:     {result['delta_i_auroc']:+.6f}")
    print(f"Baseline I-AUPR:   {result['baseline_i_aupr']:.6f}")
    print(f"NS I-AUPR:         {result['neuro_symbolic_i_aupr']:.6f}")
    print(f"Delta I-AUPR:      {result['delta_i_aupr']:+.6f}")
    print(f"Baseline P-AUROC:  {result['baseline_p_auroc']:.6f}")
    print(f"NS P-AUROC:        {result['neuro_symbolic_p_auroc']:.6f}")
    print(f"Delta P-AUROC:     {result['delta_p_auroc']:+.6f}")
    print(f"Baseline P-AUPR:   {result['baseline_p_aupr']:.6f}")
    print(f"NS P-AUPR:         {result['neuro_symbolic_p_aupr']:.6f}")
    print(f"Delta P-AUPR:      {result['delta_p_aupr']:+.6f}")


if __name__ == "__main__":
    main()