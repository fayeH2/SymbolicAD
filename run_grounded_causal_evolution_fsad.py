#!/usr/bin/env python3
"""Grounded-counterfactual SymbolicDet-style experiment on MVTec transistor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import cv2
import numpy as np

from neurosymbolic_fsad.causal_evolution import GroupRobustEvolutionSearch
from neurosymbolic_fsad.data import collect_test_samples
from neurosymbolic_fsad.experiment_logic import _mask_for_sample, _save_visualization
from neurosymbolic_fsad.fuzzy_logic import FuzzyLogicProgram
from neurosymbolic_fsad.grounded_counterfactuals import (
    generate_transistor_counterfactuals,
    ground_counterfactual_truths,
)
from neurosymbolic_fsad.learned_predicates import (
    PREDICATE_GROUPS,
    crossfit_predicate_heads,
    extract_feature_matrix,
    transistor_predicate_features,
)
from neurosymbolic_fsad.qwen_evolution_advisor import QwenEvolutionAdvisor
from neurosymbolic_fsad.symbolicdet_evolution import EvolutionConfig, qwen_seed_asts
from neurosymbolic_fsad.transistor_logic import (
    ATOM_SPECS,
    SpatialTransistorGrounder,
    TransistorLogicEngine,
    execute_grounded_neural_symbolic,
)
from run_symbolicdet_evolution_fsad import (
    _build_evolved_library,
    _cache,
    _delta,
    _dump,
    _metrics,
    _per_defect,
)


def _save_counterfactual_examples(output: Path, records, limit: int = 20) -> list[str]:
    target = output / "grounded_counterfactual_examples"
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    counts: dict[str, int] = {}
    for record in records:
        if record.label == 0 or counts.get(record.kind, 0) >= 5:
            continue
        index = counts.get(record.kind, 0)
        counts[record.kind] = index + 1
        overlay = record.image.copy()
        overlay[record.intervention_mask.astype(bool)] = (
            0.35 * overlay[record.intervention_mask.astype(bool)]
            + 0.65 * np.asarray([255, 0, 0])
        ).astype(np.uint8)
        canvas = np.concatenate([record.image, overlay], axis=1)
        path = target / f"{record.kind}_{index:02d}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        paths.append(str(path))
        if len(paths) >= limit:
            break
    return paths


def _evaluate_learned_predicates(image, grounder, heads, program, active):
    features, details, localizers = transistor_predicate_features(image, grounder)
    scalar_values, map_values, atoms = {}, {}, {}
    for name in active:
        violation = float(heads[name].violation(features[None])[0])
        truth = 1.0 - violation
        localization = np.asarray(localizers[name], dtype=np.float32)
        scalar_values[name] = truth
        map_values[name] = 1.0 - np.clip(violation * localization, 0.0, 1.0)
        atoms[name] = {
            "rule_id": name,
            "predicate": name,
            "truth": truth,
            "violation": violation,
            "raw_probability": float(heads[name].raw_probability(features[None])[0]),
            "normal_upper": float(heads[name].normal_upper),
            "defined": True,
        }
    symbolic, trace = program.evaluate(scalar_values)
    symbolic_map, _ = program.evaluate(map_values)
    violated = sorted(
        (row for row in atoms.values() if row["violation"] > 1e-8),
        key=lambda row: row["violation"], reverse=True,
    )
    explanation = (
        "; ".join(
            f"{row['rule_id']} violated with calibrated probability "
            f"{row['raw_probability']:.4f}" for row in violated
        ) if violated else "All evolved learned normality predicates are satisfied."
    )
    return SimpleNamespace(
        symbolic_anomaly=float(symbolic),
        symbolic_map=np.asarray(symbolic_map, dtype=np.float32),
        facts=details["facts"], atoms=atoms, scalar_trace=trace,
        diagnostics=details["diagnostics"], explanation=explanation,
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base_rule_library).expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("dataset") != "mvtec" or base.get("class_name") != "transistor":
        raise ValueError("controlled grounded experiment requires mvtec/transistor")

    required_vote = float(base["selection"]["required_llm_vote_rate"])
    grounder = SpatialTransistorGrounder(state=base["grounder"])
    records = generate_transistor_counterfactuals(
        base["support"], grounder, seed=args.search_seed
    )
    predicate_heads = None
    predicate_training_audit = None
    if args.learned_predicates:
        eligible = list(PREDICATE_GROUPS)
        eligible_rows = [
            {
                "id": name, "fact": f"learned_probability/{name}",
                "predicate": name,
                "description": (
                    f"Cross-fitted visual predicate for grounded intervention "
                    f"family {PREDICATE_GROUPS[name]}"
                ),
                "localizer": (
                    "lead_template_union" if name in {"detach_relation", "extra_part_count"}
                    else "lead_template_residual"
                ),
            }
            for name in eligible
        ]
        feature_matrix, grounded_audit = extract_feature_matrix(records, grounder)
        X, predicate_heads, predicate_training_audit = crossfit_predicate_heads(
            feature_matrix, records, folds=5
        )
        y = np.asarray([record.label for record in records], dtype=np.uint8)
        groups = np.asarray([record.kind for record in records], dtype=str)
    else:
        eligible_rows = [dict(row) for row in base["candidate_audit"]]
        if args.stable_only:
            eligible_rows = [
                row for row in eligible_rows
                if float(row.get("llm_vote_rate", 0.0)) >= required_vote
                and str(row.get("status")) == "accepted"
            ]
        eligible = [str(row["id"]) for row in eligible_rows]
        X, y, groups, grounded_audit = ground_counterfactual_truths(
            records, grounder, eligible_rows
        )
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("grounded search data must contain both labels")

    advisor = None
    if not args.no_llm_feedback:
        advisor = QwenEvolutionAdvisor(
            eligible,
            {
                name: (
                    next(row["description"] for row in eligible_rows if row["id"] == name)
                    if args.learned_predicates else ATOM_SPECS[name]["description"]
                )
                for name in eligible
            },
            model_path=args.llm_model,
            device=args.llm_device,
            max_new_tokens=args.llm_max_new_tokens,
            task_context=(
                "MVTec transistor normal-only FSAD with image-grounded lead "
                "detach, cut, bend and extra-part counterfactual supervision"
            ),
        )
    searcher = GroupRobustEvolutionSearch(
        eligible,
        EvolutionConfig(
            population_size=args.population_size,
            generations=args.generations,
            tournament_size=args.tournament_size,
            elite_count=args.elite_count,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            complexity_penalty=args.complexity_penalty,
            normal_false_positive_penalty=0.0,
            early_stop_patience=args.early_stop_patience,
            random_seed=args.search_seed,
            llm_feedback_interval=args.llm_feedback_interval,
        ),
        groups,
        atom_cost=args.atom_cost,
        worst_group_weight=args.worst_group_weight,
        normal_invariance_weight=args.normal_invariance_weight,
        llm_advisor=advisor,
        llm_seed_asts=qwen_seed_asts(eligible),
    )
    search = searcher.run(X, y)
    if advisor is not None:
        advisor.close()
    search["protocol"] = {
        "name": "image-grounded causal counterfactual evolution",
        "normal_source": "identity and nuisance transforms of normal supports",
        "positive_source": "image-level lead detach/cut/bend/extra interventions",
        "facts_reextracted_after_intervention": True,
        "atom_truths_directly_modified": False,
        "query_or_test_used": False,
        "groups": {name: int(np.sum(groups == name)) for name in sorted(set(groups))},
        "candidate_atoms": eligible,
        "predicate_source": (
            "source-disjoint cross-fitted weakly supervised visual heads"
            if args.learned_predicates else "fixed hand-engineered visual facts"
        ),
        "objective": (
            "global AUROC/AUPR + mean/worst intervention-family AUROC - "
            "normal invariance - MDL atom/node costs"
        ),
    }
    _dump(output / "evolution_search_audit_v9.json", search)
    _dump(output / "grounded_counterfactual_audit_v9.json", grounded_audit)
    if predicate_training_audit is not None:
        _dump(output / "predicate_training_audit_v9.json", predicate_training_audit)
    np.savez_compressed(
        output / "grounded_counterfactual_search_data_v9.npz",
        atom_names=np.asarray(eligible), X=X.astype(np.float32), y=y, groups=groups,
    )
    cf_visuals = _save_counterfactual_examples(output, records)

    if args.learned_predicates:
        active = list(search["best_active_atoms"])
        program_check = FuzzyLogicProgram(search["best_anomaly_ast"], active)
        library = json.loads(json.dumps(base))
        library.update({
            "schema_version": "9.1-grounded-learned-predicate-symbolicdet-library",
            "accepted_rule_ids": active,
            "accepted_rule_count": len(active),
            "rules": [row for row in eligible_rows if row["id"] in active],
            "predicate_heads": {
                name: predicate_heads[name].state_dict() for name in active
            },
            "logic": {
                "semantics": "evolved_product_fuzzy_logic_over_learned_predicates",
                "symbolic_anomaly_ast": search["best_anomaly_ast"],
                "runtime_validation": {
                    "active_atom_counts": program_check.validation.active_atom_counts,
                    "execution_atoms": list(program_check.validation.execution_atoms),
                    "node_count": program_check.validation.node_count,
                },
            },
            "rule_discovery": {
                "method": "LLM-guided group-robust evolution over cross-fitted learned predicates",
                "fitness_data": "normal support images and re-grounded causal interventions",
                "atom_truths_directly_modified": False,
                "counterfactual_groups": sorted(set(groups.tolist())),
                "query_or_test_data_used": False,
            },
            "runtime_requires_llm": False,
        })
    else:
        library = _build_evolved_library(base, search, eligible_rows)
        library["schema_version"] = "9.0-grounded-causal-symbolicdet-library"
        library["rule_discovery"].update({
            "method": "LLM-guided group-robust evolution on grounded image counterfactuals",
            "fitness_data": "normal support images and re-grounded causal interventions",
            "atom_truths_directly_modified": False,
            "counterfactual_groups": sorted(set(groups.tolist())),
        })
    library_path = output / "normality_rules_v9_grounded_causal.json"
    _dump(library_path, library)

    if args.search_only:
        summary = {
            "schema_version": "9.1-grounded-causal-search-only",
            "query_or_test_opened": False,
            "learned_predicates": bool(args.learned_predicates),
            "evolved_normality_ast": search["best_normality_ast"],
            "evolved_active_atoms": search["best_active_atoms"],
            "best_search_metrics": search["best_metrics"],
            "llm_feedback_rounds": len(search["llm_feedback"]),
            "rule_library": str(library_path),
        }
        _dump(output / "search_only_summary_v9.json", summary)
        return summary

    # Query barrier: the rule and AST above are frozen before any test asset is read.
    cache_root = Path(args.prediction_cache_dir).expanduser().resolve()
    if (cache_root / "cache").is_dir():
        cache_root /= "cache"
    predictions = _cache(cache_root / "test_inp_v4.npz")
    data_root = Path(args.data_root).expanduser().resolve()
    class_root = data_root / "transistor"
    samples = collect_test_samples(class_root)
    expected_paths = np.asarray([str(Path(row["path"]).resolve()) for row in samples])
    if not np.array_equal(predictions["paths"], expected_paths):
        raise ValueError("frozen baseline cache and query order differ")
    if set(library["support"]).intersection(set(expected_paths.tolist())):
        raise ValueError("normal rule support overlaps query/test")

    engine = None if args.learned_predicates else TransistorLogicEngine(library)
    learned_program = (
        FuzzyLogicProgram(search["best_anomaly_ast"], search["best_active_atoms"])
        if args.learned_predicates else None
    )
    labels = np.asarray([int(row["label"]) for row in samples], dtype=np.uint8)
    baseline_scores = np.clip(predictions["scores"], 0.0, 1.0)
    baseline_maps = np.clip(predictions["maps"], 0.0, 1.0)
    masks, symbolic_scores, fused_scores = [], [], []
    symbolic_maps, fused_maps = [], []
    rows, traces, visuals = [], [], []
    for index, (sample, baseline_score, baseline_map) in enumerate(
        zip(samples, baseline_scores, baseline_maps)
    ):
        result = (
            _evaluate_learned_predicates(
                sample["path"], grounder, predicate_heads,
                learned_program, search["best_active_atoms"],
            )
            if args.learned_predicates else engine.evaluate(sample["path"])
        )
        score, final_trace = execute_grounded_neural_symbolic(
            float(baseline_score), result.symbolic_anomaly
        )
        fused_map, _ = execute_grounded_neural_symbolic(
            baseline_map, result.symbolic_map
        )
        mask = _mask_for_sample(sample, class_root)
        masks.append(mask)
        symbolic_scores.append(result.symbolic_anomaly)
        fused_scores.append(float(score))
        symbolic_maps.append(result.symbolic_map)
        fused_maps.append(np.asarray(fused_map, dtype=np.float32))
        violated = [
            name for name, atom in result.atoms.items()
            if float(atom["violation"]) > 1e-8
        ]
        trace = {
            "path": str(Path(sample["path"]).resolve()),
            "defect_type": str(sample["defect_type"]),
            "facts": result.facts,
            "atoms": result.atoms,
            "evolved_symbolic_ast": search["best_anomaly_ast"],
            "symbolic_logic_nodes": result.scalar_trace,
            "final_fusion_logic_nodes": final_trace,
            "baseline_anomaly": float(baseline_score),
            "symbolic_anomaly": float(result.symbolic_anomaly),
            "final_anomaly": float(score),
            "violated_rules": violated,
            "explanation": result.explanation,
        }
        trace_path = output / "logic_traces" / f"{index:03d}_{Path(sample['path']).parent.name}_{Path(sample['path']).stem}.json"
        _dump(trace_path, trace)
        traces.append(trace)
        rows.append({
            "path": trace["path"], "defect_type": trace["defect_type"],
            "label": int(sample["label"]), "baseline_score": float(baseline_score),
            "symbolic_score": float(result.symbolic_anomaly),
            "fused_score": float(score),
            "violated_rules": "|".join(violated) if violated else "none",
            "trace": str(trace_path),
        })
        if args.max_visualizations <= 0 or index < args.max_visualizations:
            visuals.append(_save_visualization(
                output / "visualizations", sample, mask, baseline_map,
                result.symbolic_map, np.asarray(fused_map), index,
            ))

    masks = np.asarray(masks, dtype=bool)
    symbolic_scores = np.asarray(symbolic_scores, dtype=float)
    fused_scores = np.asarray(fused_scores, dtype=float)
    symbolic_maps = np.asarray(symbolic_maps, dtype=np.float32)
    fused_maps = np.asarray(fused_maps, dtype=np.float32)
    baseline_metrics = _metrics(labels, baseline_scores, masks, baseline_maps)
    symbolic_metrics = _metrics(labels, symbolic_scores, masks, symbolic_maps)
    fused_metrics = _metrics(labels, fused_scores, masks, fused_maps)
    defect_types = np.asarray([str(row["defect_type"]) for row in samples])
    metrics = {
        "schema_version": "9.0-grounded-causal-evolution-experiment",
        "dataset": "mvtec", "class_name": "transistor",
        "baseline_shot": int(predictions["metadata"].get("shot", 4)),
        "rule_support_count": len(base["support"]),
        "search_protocol": search["protocol"],
        "learned_predicates": bool(args.learned_predicates),
        "llm_model": args.llm_model,
        "llm_feedback_enabled": not args.no_llm_feedback,
        "llm_feedback_rounds": len(search["llm_feedback"]),
        "evolved_normality_ast": search["best_normality_ast"],
        "evolved_anomaly_ast": search["best_anomaly_ast"],
        "evolved_active_atoms": search["best_active_atoms"],
        "best_search_metrics": search["best_metrics"],
        "baseline": baseline_metrics,
        "symbolic_only": symbolic_metrics,
        "evolved_neuro_symbolic": fused_metrics,
        "delta_evolved_minus_baseline": _delta(fused_metrics, baseline_metrics),
        "per_defect_type": _per_defect(
            labels, defect_types, fused_scores, baseline_scores,
            masks, fused_maps, baseline_maps,
        ),
        "normal_symbolic_trigger_rate": float(np.mean(symbolic_scores[labels == 0] > 1e-8)),
        "anomaly_symbolic_trigger_rate": float(np.mean(symbolic_scores[labels == 1] > 1e-8)),
        "query_or_test_used_for_search": False,
        "counterfactual_visuals": cf_visuals,
        "rule_library": str(library_path),
        "search_audit": str(output / "evolution_search_audit_v9.json"),
    }
    with (output / "predictions_v9.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    _dump(output / "rule_evidence_v9.json", traces)
    _dump(output / "visualization_manifest_v9.json", visuals)
    np.savez_compressed(
        output / "pixel_predictions_v9.npz", paths=expected_paths, labels=labels,
        masks=masks, baseline_maps=baseline_maps,
        symbolic_maps=symbolic_maps, evolved_maps=fused_maps,
    )
    _dump(output / "metrics_v9.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-rule-library", required=True)
    parser.add_argument("--prediction-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-model", default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking")
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument("--llm-max-new-tokens", type=int, default=512)
    parser.add_argument("--no-llm-feedback", action="store_true")
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument(
        "--learned-predicates", action="store_true",
        help="learn source-disjoint weakly supervised visual relation predicates",
    )
    parser.add_argument(
        "--stable-only", action="store_true",
        help="restrict evolution to candidates accepted by the old normal-only stability gate",
    )
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--tournament-size", type=int, default=5)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--crossover-rate", type=float, default=0.5)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--complexity-penalty", type=float, default=0.004)
    parser.add_argument("--atom-cost", type=float, default=0.025)
    parser.add_argument("--worst-group-weight", type=float, default=0.30)
    parser.add_argument("--normal-invariance-weight", type=float, default=0.25)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--llm-feedback-interval", type=int, default=5)
    parser.add_argument("--search-seed", type=int, default=668)
    parser.add_argument("--max-visualizations", type=int, default=100)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))
