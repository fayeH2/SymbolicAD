#!/usr/bin/env python3
"""Frozen-policy multi-class audit of SymbolicDet-style rule evolution.

This runner deliberately reports image metrics only.  The category-specific
grounders currently expose scalar facts but not spatial evidence maps; claiming
pixel metrics by broadcasting a scalar rule score would not be a valid
localisation experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from neurosymbolic_fsad.class_specific_rules import ClassSpecificRuleEngine
from neurosymbolic_fsad.data import collect_test_samples
from neurosymbolic_fsad.fuzzy_logic import FuzzyLogicProgram, fuzzy_or
from neurosymbolic_fsad.qwen_evolution_advisor import QwenEvolutionAdvisor
from neurosymbolic_fsad.symbolicdet_evolution import (
    EvolutionConfig,
    SymbolicEvolutionSearch,
    counterfactual_search_data,
    qwen_seed_asts,
    support_truth_matrix,
)


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _cache(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "paths": payload["paths"].astype(str),
            "scores": payload["scores"].astype(np.float64),
            "metadata": json.loads(str(payload["metadata_json"].item())),
        }


def _metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "i_auroc": float(roc_auc_score(labels, scores)),
        "i_aupr": float(average_precision_score(labels, scores)),
    }


def _candidate_audit(base: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    for rule in base["rules"]:
        rows.append({
            **dict(rule),
            "scale": float(rule["soft_scale"]),
            "leave_one_out_values": list(rule["leave_one_support_out_values"]),
        })
    return rows


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_path = Path(args.base_rule_library).expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    dataset, class_name = str(base["dataset"]), str(base["class_name"])
    if dataset != args.dataset or class_name != args.class_name:
        raise ValueError("dataset/class does not match the frozen base rule library")

    # Everything above the query barrier is support-only.  Persist the search
    # before opening either the query cache or the test directory.
    audit = _candidate_audit(base)
    eligible = [str(row["id"]) for row in audit]
    normal_truths = support_truth_matrix(audit, eligible)
    X, y, provenance = counterfactual_search_data(
        normal_truths, eligible, seed=args.search_seed,
        variants_per_atom=args.variants_per_atom, pair_variants=args.pair_variants,
    )
    descriptions = {
        str(row["id"]): f"{row['predicate']}: normal when {row['fact']} stays below its support-calibrated upper bound"
        for row in audit
    }
    advisor = None
    if not args.no_llm_feedback:
        advisor = QwenEvolutionAdvisor(
            eligible, descriptions, model_path=args.llm_model,
            device=args.llm_device, max_new_tokens=args.llm_max_new_tokens,
            task_context=f"{dataset}/{class_name} few-shot anomaly detection",
        )
    searcher = SymbolicEvolutionSearch(
        eligible,
        EvolutionConfig(
            population_size=args.population_size, generations=args.generations,
            tournament_size=args.tournament_size, elite_count=args.elite_count,
            crossover_rate=args.crossover_rate, mutation_rate=args.mutation_rate,
            complexity_penalty=args.complexity_penalty,
            normal_false_positive_penalty=args.normal_fpr_penalty,
            early_stop_patience=args.early_stop_patience,
            random_seed=args.search_seed,
            llm_feedback_interval=args.llm_feedback_interval,
        ),
        llm_advisor=advisor,
        llm_seed_asts=qwen_seed_asts(eligible),
    )
    search = searcher.run(X, y)
    if advisor is not None:
        advisor.close()
    search["protocol"] = {
        "candidate_source": "frozen category-specific support-only rule library",
        "real_normal_source": "leave-one-support-out truths",
        "positive_source": "deterministic atom-space counterfactuals",
        "query_or_test_used": False,
        "candidate_atoms": eligible,
    }
    _dump(output / "evolution_search_audit.json", search)
    np.savez_compressed(
        output / "support_counterfactual_search_data.npz",
        atom_names=np.asarray(eligible), X=X.astype(np.float32), y=y,
        provenance_json=np.asarray(json.dumps(provenance, ensure_ascii=False)),
    )

    active = list(search["best_active_atoms"])
    active_rows = [dict(row) for row in base["rules"] if str(row["id"]) in active]
    evolved_library = dict(base)
    evolved_library.update({
        "schema_version": "8.1-symbolicdet-evolved-category-library",
        "rules": active_rows,
        "accepted_rule_ids": active,
        "symbolic_normality_ast": search["best_normality_ast"],
        "symbolic_anomaly_ast": search["best_anomaly_ast"],
        "runtime_requires_llm": False,
        "query_or_test_used_for_search": False,
    })
    library_path = output / "normality_rules_v8_evolved.json"
    _dump(library_path, evolved_library)

    # Query barrier: no code above this point reads query/test images or scores.
    cache_root = Path(args.prediction_cache_dir).expanduser().resolve()
    if (cache_root / "cache").is_dir():
        cache_root /= "cache"
    predictions = _cache(cache_root / "test_inp_v4.npz")
    class_root = Path(args.data_root).expanduser().resolve() / class_name
    samples = collect_test_samples(class_root)
    expected = np.asarray([str(Path(row["path"]).resolve()) for row in samples])
    if not np.array_equal(predictions["paths"], expected):
        raise ValueError("baseline cache path/order mismatch")
    if set(map(str, base["support"])).intersection(set(expected.tolist())):
        raise ValueError("normal support overlaps test/query split")

    engine = ClassSpecificRuleEngine(base)
    program = FuzzyLogicProgram(search["best_anomaly_ast"], active)
    labels = np.asarray([int(row["label"]) for row in samples], dtype=np.uint8)
    baseline = np.clip(predictions["scores"], 0.0, 1.0)
    symbolic, fused, output_rows, traces = [], [], [], []
    for sample, neural in zip(samples, baseline):
        _, evidence, facts, diagnostics = engine.evaluate(sample["path"])
        by_id = {str(row["rule_id"]): row for row in evidence}
        normality = {name: 1.0 - float(by_id[name]["violation"]) for name in active}
        symbolic_score, nodes = program.evaluate(normality)
        final = float(fuzzy_or(float(neural), float(symbolic_score)))
        symbolic.append(float(symbolic_score))
        fused.append(final)
        violated = [name for name in active if by_id[name]["violation"] > 1e-8]
        trace = {
            "path": str(Path(sample["path"]).resolve()), "facts": facts,
            "diagnostics": diagnostics, "active_atom_truths": normality,
            "logic_nodes": nodes, "baseline": float(neural),
            "symbolic": float(symbolic_score), "fused": final,
            "violated_rules": violated,
        }
        traces.append(trace)
        output_rows.append({
            "path": trace["path"], "defect_type": sample["defect_type"],
            "label": int(sample["label"]), "baseline_score": float(neural),
            "symbolic_score": float(symbolic_score), "fused_score": final,
            "violated_rules": "|".join(violated) if violated else "none",
        })
    symbolic = np.asarray(symbolic); fused = np.asarray(fused)
    baseline_metrics, fused_metrics = _metrics(labels, baseline), _metrics(labels, fused)
    defect_types = np.asarray([str(row["defect_type"]) for row in samples])
    per_defect = {}
    for defect in sorted(set(defect_types) - {"good", "ok"}):
        selected = np.isin(defect_types, ["good", "ok", defect])
        per_defect[defect] = {
            "baseline_i_auroc": float(roc_auc_score(labels[selected], baseline[selected])),
            "evolved_i_auroc": float(roc_auc_score(labels[selected], fused[selected])),
        }
    result = {
        "schema_version": "8.1-symbolicdet-evolution-multiclass-audit",
        "dataset": dataset, "class_name": class_name,
        "baseline_shot": int(predictions["metadata"].get("shot", 4)),
        "rule_support_count": len(base["support"]),
        "candidate_atoms": eligible, "evolved_active_atoms": active,
        "evolved_normality_ast": search["best_normality_ast"],
        "llm_feedback_enabled": not args.no_llm_feedback,
        "llm_feedback_rounds": len(search["llm_feedback"]),
        "query_or_test_used_for_search": False,
        "fusion": "product-fuzzy OR(INP-Former anomaly, symbolic anomaly)",
        "baseline": baseline_metrics, "symbolic_only": _metrics(labels, symbolic),
        "evolved_neuro_symbolic": fused_metrics,
        "delta_evolved_minus_baseline": {
            key: fused_metrics[key] - baseline_metrics[key] for key in baseline_metrics
        },
        "per_defect_type": per_defect,
        "pixel_metrics_status": "not reported: scalar class grounder has no spatial evidence map",
        "rule_library": str(library_path),
        "search_audit": str(output / "evolution_search_audit.json"),
    }
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)
    _dump(output / "rule_evidence.json", traces)
    _dump(output / "metrics.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-rule-library", required=True)
    parser.add_argument("--prediction-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llm-model", default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking")
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument("--llm-max-new-tokens", type=int, default=1024)
    parser.add_argument("--no-llm-feedback", action="store_true")
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
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))
