#!/usr/bin/env python3
"""Leakage-free SymbolicDet-style evolutionary FSAD experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from neurosymbolic_fsad.data import collect_test_samples
from neurosymbolic_fsad.experiment_logic import _mask_for_sample, _save_visualization
from neurosymbolic_fsad.fuzzy_logic import FuzzyLogicProgram
from neurosymbolic_fsad.qwen_evolution_advisor import QwenEvolutionAdvisor
from neurosymbolic_fsad.symbolicdet_evolution import (
    EvolutionConfig,
    SymbolicEvolutionSearch,
    counterfactual_search_data,
    qwen_seed_asts,
    support_truth_matrix,
)
from neurosymbolic_fsad.transistor_logic import (
    ATOM_SPECS,
    TransistorLogicEngine,
    execute_grounded_neural_symbolic,
)


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _cache(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "paths": payload["paths"].astype(str),
            "scores": payload["scores"].astype(np.float64),
            "maps": payload["maps"].astype(np.float32),
            "metadata": json.loads(str(payload["metadata_json"].item())),
        }


def _metrics(labels, scores, masks, maps) -> Dict[str, float]:
    return {
        "i_auroc": float(roc_auc_score(labels.reshape(-1), scores.reshape(-1))),
        "i_aupr": float(average_precision_score(labels.reshape(-1), scores.reshape(-1))),
        "p_auroc": float(roc_auc_score(masks.reshape(-1), maps.reshape(-1))),
        "p_aupr": float(average_precision_score(masks.reshape(-1), maps.reshape(-1))),
    }


def _delta(candidate: Mapping[str, float], baseline: Mapping[str, float]):
    return {name: float(candidate[name] - baseline[name]) for name in baseline}


def _per_defect(labels, defect_types, scores, baseline_scores, masks, maps, baseline_maps):
    result = {}
    for defect in sorted(set(defect_types) - {"good", "ok"}):
        selected = np.isin(defect_types, ["good", "ok", defect])
        anomaly = defect_types == defect
        result[defect] = {
            "anomaly_count": int(np.sum(anomaly)),
            "baseline_i_auroc": float(
                roc_auc_score(labels[selected], baseline_scores[selected])
            ),
            "evolved_i_auroc": float(roc_auc_score(labels[selected], scores[selected])),
            "baseline_p_auroc": float(
                roc_auc_score(masks[anomaly].reshape(-1), baseline_maps[anomaly].reshape(-1))
            ),
            "evolved_p_auroc": float(
                roc_auc_score(masks[anomaly].reshape(-1), maps[anomaly].reshape(-1))
            ),
            "baseline_p_aupr": float(
                average_precision_score(
                    masks[anomaly].reshape(-1), baseline_maps[anomaly].reshape(-1)
                )
            ),
            "evolved_p_aupr": float(
                average_precision_score(masks[anomaly].reshape(-1), maps[anomaly].reshape(-1))
            ),
        }
    return result


def _build_evolved_library(
    base: Mapping[str, Any], search: Mapping[str, Any],
    eligible_rule_rows: list[Mapping[str, Any]],
):
    library = json.loads(json.dumps(base))
    active = list(search["best_active_atoms"])
    rules = [dict(row) for row in eligible_rule_rows if str(row["id"]) in active]
    if {str(row["id"]) for row in rules} != set(active):
        raise ValueError("Evolved AST selected an atom absent from the stable rule library")
    program = FuzzyLogicProgram(search["best_anomaly_ast"], active)
    library.update({
        "schema_version": "8.0-symbolicdet-evolved-static-rule-library",
        "accepted_rule_ids": active,
        "accepted_rule_count": len(active),
        "rules": rules,
        "rule_discovery": {
            "method": "LLM-guided evolutionary symbolic search",
            "fitness_data": "normal-support LOO truths plus synthetic atom counterfactuals",
            "query_or_test_data_used": False,
            "search_schema": search["schema_version"],
            "best_fitness": search["best_fitness"],
            "best_metrics": search["best_metrics"],
            "llm_feedback_rounds": len(search["llm_feedback"]),
            "pre_evolution_candidate_atoms": [
                str(row["id"]) for row in eligible_rule_rows
            ],
        },
        "runtime_requires_llm": False,
    })
    library["logic"] = {
        **library["logic"],
        "semantics": "evolved_product_fuzzy_logic",
        "symbolic_anomaly_ast": search["best_anomaly_ast"],
        "formula": (
            "evolved NormalityAST; SymbolicAnomaly=NOT(NormalityAST); "
            "final=OR(INPFormerAnomaly,AND(NeuralGrounding,SymbolicAnomaly))"
        ),
        "runtime_validation": {
            "active_atom_counts": program.validation.active_atom_counts,
            "execution_atoms": list(program.validation.execution_atoms),
            "node_count": program.validation.node_count,
        },
    }
    return library


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base_rule_library).expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("dataset") != "mvtec" or base.get("class_name") != "transistor":
        raise ValueError("v8 controlled runner currently requires mvtec/transistor")

    # Search is deliberately completed and persisted before any query cache,
    # query directory, query mask, or query label is opened.
    if args.evolve_all_groundable:
        required_vote = float(base["selection"]["required_llm_vote_rate"])
        eligible_rows = [
            row for row in base["candidate_audit"]
            if float(row.get("llm_vote_rate", 0.0)) >= required_vote
        ]
    else:
        eligible_rows = list(base["rules"])
    eligible = [str(row["id"]) for row in eligible_rows]
    if not eligible:
        raise ValueError("No stable support-calibrated atoms are eligible for evolution")
    normal_truths = support_truth_matrix(base["candidate_audit"], eligible)
    X_search, y_search, provenance = counterfactual_search_data(
        normal_truths,
        eligible,
        seed=args.search_seed,
        variants_per_atom=args.variants_per_atom,
        pair_variants=args.pair_variants,
    )
    advisor = None
    if not args.no_llm_feedback:
        advisor = QwenEvolutionAdvisor(
            eligible,
            {name: ATOM_SPECS[name]["description"] for name in eligible},
            model_path=args.llm_model,
            device=args.llm_device,
            max_new_tokens=args.llm_max_new_tokens,
        )
    config = EvolutionConfig(
        population_size=args.population_size,
        generations=args.generations,
        tournament_size=args.tournament_size,
        elite_count=args.elite_count,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        complexity_penalty=args.complexity_penalty,
        normal_false_positive_penalty=args.normal_fpr_penalty,
        early_stop_patience=args.early_stop_patience,
        random_seed=args.search_seed,
        llm_feedback_interval=args.llm_feedback_interval,
    )
    search_engine = SymbolicEvolutionSearch(
        eligible,
        config,
        llm_advisor=advisor,
        llm_seed_asts=qwen_seed_asts(eligible),
    )
    search = search_engine.run(X_search, y_search)
    if advisor is not None:
        advisor.close()
    search.update({
        "protocol": {
            "normal_source": "leave-one-normal-support-out grounded facts",
            "positive_source": "deterministic atom-space counterfactual violations",
            "test_images_seen_during_search": False,
            "test_labels_seen_during_search": False,
            "candidate_atoms": eligible,
            "candidate_atom_source": (
                "all Qwen-supported groundable families"
                if args.evolve_all_groundable else
                "Qwen proposals plus normal-only stability compiler"
            ),
            "evolve_all_qwen_groundable_candidates": bool(args.evolve_all_groundable),
            "search_example_count": int(len(y_search)),
            "real_normal_count": int(np.sum(y_search == 0)),
            "counterfactual_count": int(np.sum(y_search == 1)),
        },
        "base_rule_library": str(base_path),
    })
    _dump(output / "evolution_search_audit_v8.json", search)
    np.savez_compressed(
        output / "support_counterfactual_search_data_v8.npz",
        atom_names=np.asarray(eligible),
        X=X_search.astype(np.float32),
        y=y_search,
        provenance_json=np.asarray(json.dumps(provenance, ensure_ascii=False)),
    )
    library = _build_evolved_library(base, search, eligible_rows)
    library_path = output / "normality_rules_v8_symbolicdet_evolved.json"
    _dump(library_path, library)

    # Query/test data are first opened below, after the evolved AST is frozen.
    cache_root = Path(args.prediction_cache_dir).expanduser().resolve()
    if (cache_root / "cache").is_dir():
        cache_root = cache_root / "cache"
    predictions = _cache(cache_root / "test_inp_v4.npz")
    data_root = Path(args.data_root).expanduser().resolve()
    class_root = data_root / "transistor"
    samples = collect_test_samples(class_root)
    expected_paths = np.asarray([str(Path(row["path"]).resolve()) for row in samples])
    if not np.array_equal(predictions["paths"], expected_paths):
        raise ValueError("Frozen baseline cache and query order differ")
    if set(library["support"]).intersection(set(expected_paths.tolist())):
        raise ValueError("Normal rule support overlaps the test/query split")

    engine = TransistorLogicEngine(library)
    labels = np.asarray([int(row["label"]) for row in samples], dtype=np.uint8)
    baseline_scores = np.clip(predictions["scores"], 0.0, 1.0)
    baseline_maps = np.clip(predictions["maps"], 0.0, 1.0)
    masks, symbolic_scores, evolved_scores = [], [], []
    symbolic_maps, evolved_maps = [], []
    rows, trace_rows, visuals = [], [], []
    for index, (sample, baseline_score, baseline_map) in enumerate(
        zip(samples, baseline_scores, baseline_maps)
    ):
        result = engine.evaluate(sample["path"])
        score, final_trace = execute_grounded_neural_symbolic(
            float(baseline_score), result.symbolic_anomaly
        )
        fused_map, _ = execute_grounded_neural_symbolic(
            baseline_map, result.symbolic_map
        )
        mask = _mask_for_sample(sample, class_root)
        masks.append(mask)
        symbolic_scores.append(result.symbolic_anomaly)
        evolved_scores.append(float(score))
        symbolic_maps.append(result.symbolic_map)
        evolved_maps.append(np.asarray(fused_map, dtype=np.float32))
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
        trace_rows.append(trace)
        rows.append({
            "path": trace["path"],
            "defect_type": trace["defect_type"],
            "label": int(sample["label"]),
            "baseline_score": float(baseline_score),
            "symbolic_score": float(result.symbolic_anomaly),
            "evolved_score": float(score),
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
    evolved_scores = np.asarray(evolved_scores, dtype=float)
    symbolic_maps = np.asarray(symbolic_maps, dtype=np.float32)
    evolved_maps = np.asarray(evolved_maps, dtype=np.float32)
    baseline_metrics = _metrics(labels, baseline_scores, masks, baseline_maps)
    symbolic_metrics = _metrics(labels, symbolic_scores, masks, symbolic_maps)
    evolved_metrics = _metrics(labels, evolved_scores, masks, evolved_maps)
    defect_types = np.asarray([str(row["defect_type"]) for row in samples])
    fixed_v7 = None
    if args.fixed_v7_metrics:
        previous = json.loads(
            Path(args.fixed_v7_metrics).expanduser().resolve().read_text(encoding="utf-8")
        )
        fixed_v7 = previous.get("neuro_symbolic")
    metrics = {
        "schema_version": "8.0-symbolicdet-evolution-controlled-experiment",
        "dataset": "mvtec",
        "class_name": "transistor",
        "baseline_shot": int(predictions["metadata"].get("shot", 4)),
        "rule_support_count": int(base["rule_support_count"]),
        "rule_search": "support-only counterfactual evolutionary search",
        "llm_model": args.llm_model,
        "llm_feedback_enabled": not args.no_llm_feedback,
        "query_or_test_used_for_search": False,
        "evolved_normality_ast": search["best_normality_ast"],
        "evolved_anomaly_ast": search["best_anomaly_ast"],
        "evolved_active_atoms": search["best_active_atoms"],
        "pre_evolution_candidate_atoms": eligible,
        "evolve_all_qwen_groundable_candidates": bool(args.evolve_all_groundable),
        "baseline": baseline_metrics,
        "symbolic_only": symbolic_metrics,
        "evolved_neuro_symbolic": evolved_metrics,
        "delta_evolved_minus_baseline": _delta(evolved_metrics, baseline_metrics),
        "fixed_v7_neuro_symbolic_reference": fixed_v7,
        "per_defect_type": _per_defect(
            labels, defect_types, evolved_scores, baseline_scores,
            masks, evolved_maps, baseline_maps,
        ),
        "normal_symbolic_trigger_rate": float(
            np.mean(symbolic_scores[labels == 0] > 1e-8)
        ),
        "anomaly_symbolic_trigger_rate": float(
            np.mean(symbolic_scores[labels == 1] > 1e-8)
        ),
        "rule_library": str(library_path),
        "search_audit": str(output / "evolution_search_audit_v8.json"),
        "prediction_cache": str(cache_root),
        "runtime_requires_llm": False,
        "single_category_mechanism_development": True,
    }
    with (output / "predictions_v8_symbolicdet.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _dump(output / "rule_evidence_v8.json", trace_rows)
    _dump(output / "visualization_manifest_v8.json", visuals)
    np.savez_compressed(
        output / "pixel_predictions_v8.npz",
        paths=expected_paths,
        labels=labels,
        masks=masks,
        baseline_maps=baseline_maps,
        symbolic_maps=symbolic_maps,
        evolved_maps=evolved_maps,
    )
    _dump(output / "metrics_v8_symbolicdet.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-rule-library", required=True)
    parser.add_argument("--prediction-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fixed-v7-metrics")
    parser.add_argument(
        "--llm-model", default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking"
    )
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument("--llm-max-new-tokens", type=int, default=1024)
    parser.add_argument("--no-llm-feedback", action="store_true")
    parser.add_argument(
        "--evolve-all-groundable",
        action="store_true",
        help=(
            "evolve all Qwen-supported groundable atom families, including "
            "ones rejected by the conservative bootstrap stability gate"
        ),
    )
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--tournament-size", type=int, default=5)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--crossover-rate", type=float, default=0.5)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--complexity-penalty", type=float, default=0.004)
    parser.add_argument("--normal-fpr-penalty", type=float, default=0.20)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--llm-feedback-interval", type=int, default=5)
    parser.add_argument("--search-seed", type=int, default=668)
    parser.add_argument("--variants-per-atom", type=int, default=6)
    parser.add_argument("--pair-variants", type=int, default=3)
    parser.add_argument("--max-visualizations", type=int, default=100)
    return parser


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
