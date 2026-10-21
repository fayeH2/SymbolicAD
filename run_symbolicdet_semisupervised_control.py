#!/usr/bin/env python3
"""SymbolicDet-faithful labeled-event control for the FSAD supervision audit.

This is intentionally labelled semi-supervised: a deterministic few-shot set
of real anomalies is used for rule search and is permanently excluded from the
reported evaluation.  It tests whether the normal-only supervision gap, rather
than the evolutionary machinery, is the limiting factor.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from neurosymbolic_fsad.causal_evolution import GroupRobustEvolutionSearch
from neurosymbolic_fsad.data import collect_test_samples
from neurosymbolic_fsad.experiment_logic import _mask_for_sample, _save_visualization
from neurosymbolic_fsad.qwen_evolution_advisor import QwenEvolutionAdvisor
from neurosymbolic_fsad.symbolicdet_evolution import (
    EvolutionConfig, qwen_seed_asts, support_truth_matrix,
)
from neurosymbolic_fsad.transistor_logic import (
    ATOM_SPECS, SpatialTransistorGrounder, TransistorLogicEngine,
    execute_grounded_neural_symbolic,
)
from run_symbolicdet_evolution_fsad import (
    _build_evolved_library, _cache, _delta, _dump, _metrics, _per_defect,
)


def _truths(grounder, rules, image):
    facts, _, diagnostics, _ = grounder.extract_spatial(image)
    row = []
    for rule in rules:
        value = float(facts[rule["fact"]])
        excess = max(0.0, value - float(rule["upper"]))
        violation = float(np.tanh(excess / max(float(rule["scale"]), 1e-8)))
        row.append(1.0 - violation)
    return row, facts, diagnostics


def run(args):
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base_rule_library).expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    rules = [dict(row) for row in base["candidate_audit"]]
    atoms = [str(row["id"]) for row in rules]
    grounder = SpatialTransistorGrounder(state=base["grounder"])

    data_root = Path(args.data_root).expanduser().resolve()
    class_root = data_root / "transistor"
    all_samples = collect_test_samples(class_root)
    defect_counts = {}
    development, evaluation = [], []
    for sample in all_samples:
        defect = str(sample["defect_type"])
        if int(sample["label"]) == 1 and defect_counts.get(defect, 0) < args.anomaly_shot:
            development.append(sample)
            defect_counts[defect] = defect_counts.get(defect, 0) + 1
        else:
            evaluation.append(sample)
    if not development:
        raise ValueError("no labeled anomaly development samples were selected")

    # Original SymbolicDet uses labeled positive/negative events.  Here normal
    # negatives are support LOO truths and positive events are the disjoint
    # anomaly development split.
    normal = support_truth_matrix(rules, atoms)
    rows = [row for row in normal]
    labels = [0] * len(normal)
    groups = ["normal_support_loo"] * len(normal)
    dev_audit = []
    for sample in development:
        row, facts, diagnostics = _truths(grounder, rules, sample["path"])
        rows.append(row); labels.append(1); groups.append(str(sample["defect_type"]))
        dev_audit.append({
            "path": str(Path(sample["path"]).resolve()),
            "defect_type": str(sample["defect_type"]),
            "facts": facts, "diagnostics": diagnostics, "truths": dict(zip(atoms, row)),
        })
    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=np.uint8)
    groups_array = np.asarray(groups, dtype=str)

    advisor = None
    if not args.no_llm_feedback:
        advisor = QwenEvolutionAdvisor(
            atoms, {name: ATOM_SPECS[name]["description"] for name in atoms},
            model_path=args.llm_model, device=args.llm_device,
            max_new_tokens=args.llm_max_new_tokens,
            task_context=(
                "MVTec transistor semi-supervised anomaly detection with "
                "disjoint labeled anomaly development events"
            ),
        )
    searcher = GroupRobustEvolutionSearch(
        atoms,
        EvolutionConfig(
            population_size=args.population_size, generations=args.generations,
            tournament_size=5, elite_count=6, crossover_rate=0.5,
            mutation_rate=0.3, complexity_penalty=0.004,
            normal_false_positive_penalty=0.0, early_stop_patience=12,
            random_seed=args.search_seed, llm_feedback_interval=5,
        ),
        groups_array, atom_cost=args.atom_cost, worst_group_weight=0.30,
        normal_invariance_weight=0.25, llm_advisor=advisor,
        llm_seed_asts=qwen_seed_asts(atoms),
    )
    search = searcher.run(X, y)
    if advisor is not None:
        advisor.close()
    search["protocol"] = {
        "name": "SymbolicDet-faithful disjoint labeled-event control",
        "normal_count": int(len(normal)),
        "labeled_anomaly_development_count": len(development),
        "anomaly_shot_per_defect": args.anomaly_shot,
        "development_paths_excluded_from_evaluation": True,
        "evaluation_or_query_used_for_search": False,
        "status": "semi-supervised control; not normal-only FSAD",
    }
    _dump(output / "evolution_search_audit.json", search)
    _dump(output / "development_events.json", dev_audit)
    library = _build_evolved_library(base, search, rules)
    library["schema_version"] = "9.2-semisupervised-symbolicdet-control"
    library["rule_discovery"].update(search["protocol"])
    library_path = output / "normality_rules_semisupervised.json"
    _dump(library_path, library)

    cache_root = Path(args.prediction_cache_dir).expanduser().resolve()
    if (cache_root / "cache").is_dir():
        cache_root /= "cache"
    predictions = _cache(cache_root / "test_inp_v4.npz")
    cache_by_path = {
        str(Path(path).resolve()): index for index, path in enumerate(predictions["paths"])
    }
    indices = [cache_by_path[str(Path(sample["path"]).resolve())] for sample in evaluation]
    baseline_scores = np.clip(predictions["scores"][indices], 0.0, 1.0)
    baseline_maps = np.clip(predictions["maps"][indices], 0.0, 1.0)
    engine = TransistorLogicEngine(library)
    eval_labels = np.asarray([int(row["label"]) for row in evaluation], dtype=np.uint8)
    masks, symbolic_scores, fused_scores = [], [], []
    symbolic_maps, fused_maps, csv_rows, traces, visuals = [], [], [], [], []
    for index, (sample, baseline_score, baseline_map) in enumerate(
        zip(evaluation, baseline_scores, baseline_maps)
    ):
        result = engine.evaluate(sample["path"])
        score, final_trace = execute_grounded_neural_symbolic(
            float(baseline_score), result.symbolic_anomaly
        )
        fused_map, _ = execute_grounded_neural_symbolic(baseline_map, result.symbolic_map)
        mask = _mask_for_sample(sample, class_root)
        masks.append(mask); symbolic_scores.append(result.symbolic_anomaly)
        fused_scores.append(float(score)); symbolic_maps.append(result.symbolic_map)
        fused_maps.append(np.asarray(fused_map, dtype=np.float32))
        violated = [name for name, atom in result.atoms.items() if atom["violation"] > 1e-8]
        trace = {
            "path": str(Path(sample["path"]).resolve()),
            "defect_type": str(sample["defect_type"]), "facts": result.facts,
            "atoms": result.atoms, "symbolic_logic_nodes": result.scalar_trace,
            "final_fusion_logic_nodes": final_trace, "baseline": float(baseline_score),
            "symbolic": float(result.symbolic_anomaly), "fused": float(score),
            "violated_rules": violated, "explanation": result.explanation,
        }
        traces.append(trace)
        csv_rows.append({
            "path": trace["path"], "defect_type": trace["defect_type"],
            "label": int(sample["label"]), "baseline_score": float(baseline_score),
            "symbolic_score": float(result.symbolic_anomaly), "fused_score": float(score),
            "violated_rules": "|".join(violated) if violated else "none",
        })
        if args.max_visualizations <= 0 or index < args.max_visualizations:
            visuals.append(_save_visualization(
                output / "visualizations", sample, mask, baseline_map,
                result.symbolic_map, np.asarray(fused_map), index,
            ))
    masks = np.asarray(masks, dtype=bool)
    symbolic_scores = np.asarray(symbolic_scores); fused_scores = np.asarray(fused_scores)
    symbolic_maps = np.asarray(symbolic_maps); fused_maps = np.asarray(fused_maps)
    baseline_metrics = _metrics(eval_labels, baseline_scores, masks, baseline_maps)
    fused_metrics = _metrics(eval_labels, fused_scores, masks, fused_maps)
    defect_types = np.asarray([str(row["defect_type"]) for row in evaluation])
    result = {
        "schema_version": "9.2-semisupervised-symbolicdet-control",
        "status": "semi-supervised control; not normal-only FSAD",
        "anomaly_shot_per_defect": args.anomaly_shot,
        "development_count": len(development), "evaluation_count": len(evaluation),
        "development_paths_excluded_from_evaluation": True,
        "evolved_normality_ast": search["best_normality_ast"],
        "evolved_active_atoms": search["best_active_atoms"],
        "best_search_metrics": search["best_metrics"],
        "llm_feedback_enabled": not args.no_llm_feedback,
        "llm_feedback_rounds": len(search["llm_feedback"]),
        "baseline": baseline_metrics,
        "symbolic_only": _metrics(eval_labels, symbolic_scores, masks, symbolic_maps),
        "evolved_neuro_symbolic": fused_metrics,
        "delta_evolved_minus_baseline": _delta(fused_metrics, baseline_metrics),
        "per_defect_type": _per_defect(
            eval_labels, defect_types, fused_scores, baseline_scores,
            masks, fused_maps, baseline_maps,
        ),
        "rule_library": str(library_path),
    }
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    _dump(output / "rule_evidence.json", traces)
    _dump(output / "visualization_manifest.json", visuals)
    _dump(output / "metrics.json", result)
    return result


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-rule-library", required=True)
    parser.add_argument("--prediction-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anomaly-shot", type=int, default=4)
    parser.add_argument("--atom-cost", type=float, default=0.01)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--search-seed", type=int, default=668)
    parser.add_argument("--llm-model", default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking")
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument("--llm-max-new-tokens", type=int, default=512)
    parser.add_argument("--no-llm-feedback", action="store_true")
    parser.add_argument("--max-visualizations", type=int, default=100)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))
