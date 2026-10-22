#!/usr/bin/env python3
"""End-to-end verification of the breakfast_box rule-evolution search.

This driver answers one question with executable evidence:

    Given the four Qwen support images, can the proposed rule set be executed on
    independent normal validation images (``L_normal``), does it react to
    independently confirmed semantic interventions (``L_cf``), and does the
    evolutionary search find the same Pareto front as an exhaustive oracle?

Every stage fails closed and every number is written to
``verification.json``; ``verification.md`` is the human-readable report.

Example
-------
    python -m symbolicAD.rules.verify_breakfast_box \
        --support /home/hjh/Desktop/test02.png /home/hjh/Desktop/test03.png \
                  /home/hjh/Desktop/test04.png /home/hjh/Desktop/test05.png \
        --audit-json results/qwen_rule_evolution/breakfast_box_k4/audit.json \
        --output-dir results/qwen_rule_evolution/breakfast_box_k4 \
        --devices cuda:0 cuda:1 --seeds 1 7 668
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if __package__ in {None, ""}:  # allow ``python symbolicAD/rules/verify_breakfast_box.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from symbolicAD.reasoning.predicate_executor import ExecutionContext
from symbolicAD.reasoning.violation_result import RuleStatus
from symbolicAD.rules.facts.calibration import CountCalibration
from symbolicAD.rules.facts.open_vocab import GroundingDinoDetector
from symbolicAD.rules.facts.pipeline import (
    DatasetSpec, assemble_facts, calibrate_split, detect_split, discover_breakfast_box,
    falsifiability_audit, finalize_counterfactual_pairs, materialize_interventions,
    sha256_file,
)
from symbolicAD.rules.facts.scene_builder import SceneFactBuilder, SceneFactConfig
from symbolicAD.rules.qwen_adapter import adapt_qwen_candidates, payloads_from_audit
from symbolicAD.rules.rule_registry import default_rule_registry
from symbolicAD.rules.rule_schema import RuleSpec
from symbolicAD.rules.search.symbolicdet_evolution import (
    CounterfactualPair, EvolutionaryRuleSearch, RuleSetEvaluation, SearchConfig,
    SearchOutcome, exhaustive_rule_search, hard_constraint_reasons,
)


DEFAULT_CLASS_ROOT = Path(
    "/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO/breakfast_box"
)


# ---------------------------------------------------------------------------
# Candidate ledger
# ---------------------------------------------------------------------------


def load_candidates(audit_path: Path) -> Tuple[Tuple[RuleSpec, ...], List[Dict[str, Any]]]:
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    payloads = payloads_from_audit(report)
    adaptations = adapt_qwen_candidates(payloads)
    registry = default_rule_registry()
    rules: List[RuleSpec] = []
    ledger: List[Dict[str, Any]] = []
    for payload, adaptation in zip(payloads, adaptations):
        row = {
            "rule_id": adaptation.source_rule_id,
            "description": payload.get("description", ""),
            "qwen_predicate": adaptation.source_predicate,
            "adaptation_status": adaptation.status.value,
            "adaptation_reason": adaptation.reason_code,
            "typed_rule": None if adaptation.rule is None else adaptation.rule.to_dict(),
            "executable": bool(
                adaptation.rule is not None
                and registry.resolve(adaptation.rule.canonical_predicate) is not None
                and not hard_constraint_reasons((adaptation.rule,), registry)
            ),
            "hard_constraint_reasons": (
                list(hard_constraint_reasons((adaptation.rule,), registry))
                if adaptation.rule is not None else []
            ),
        }
        ledger.append(row)
        if adaptation.rule is not None:
            rules.append(adaptation.rule)
    return tuple(rules), ledger


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _rule_set_score(
    rules: Sequence[RuleSpec],
    context: ExecutionContext,
    registry,
) -> Tuple[float, int, int]:
    """Return ``(max_violation, known_rules, total_rules)`` for one context."""
    scores = []
    known = 0
    for rule in rules:
        capability = registry.resolve(rule.canonical_predicate)
        if capability is None:
            continue
        result = capability.executor.execute(rule, context)
        if result.status == RuleStatus.UNKNOWN:
            continue
        known += 1
        scores.append(float(result.violation_score))
    return (max(scores) if scores else 0.0), known, len(rules)


def evaluate_selected_on_split(
    rules: Sequence[RuleSpec],
    facts: Mapping[str, ExecutionContext],
    *,
    registry,
) -> Dict[str, Any]:
    rows = []
    for path, context in facts.items():
        score, known, total = _rule_set_score(rules, context, registry)
        rows.append({
            "image": Path(path).name,
            "score": score,
            "known_rules": known,
            "total_rules": total,
            "known_fraction": (known / total) if total else 0.0,
        })
    scores = np.asarray([row["score"] for row in rows], dtype=float)
    known_fractions = np.asarray([row["known_fraction"] for row in rows], dtype=float)
    return {
        "image_count": len(rows),
        "mean_score": float(scores.mean()) if scores.size else 0.0,
        "max_score": float(scores.max()) if scores.size else 0.0,
        "fraction_scored_positive": float(np.mean(scores > 0)) if scores.size else 0.0,
        "mean_known_fraction": (
            float(known_fractions.mean()) if known_fractions.size else 0.0
        ),
        "rows": rows,
    }


def per_rule_normal_statistics(
    rules: Sequence[RuleSpec],
    facts: Mapping[str, ExecutionContext],
    *,
    registry,
    minimum_known_fraction: float,
) -> List[Dict[str, Any]]:
    rows = []
    contexts = list(facts.values())
    for rule in rules:
        capability = registry.resolve(rule.canonical_predicate)
        if capability is None:
            rows.append({
                "rule_id": rule.rule_id, "predicate": rule.canonical_predicate,
                "executable": False,
            })
            continue
        scores, unknown = [], 0
        for context in contexts:
            result = capability.executor.execute(rule, context)
            if result.status == RuleStatus.UNKNOWN:
                unknown += 1
                continue
            scores.append(float(result.violation_score))
        known_fraction = len(scores) / len(contexts) if contexts else 0.0
        rows.append({
            "rule_id": rule.rule_id,
            "predicate": rule.canonical_predicate,
            "entity_types": list(rule.entity_types),
            "parameters": dict(rule.parameters),
            "executable": True,
            "known_fraction": known_fraction,
            "unknown_count": unknown,
            "mean_violation_on_known": float(np.mean(scores)) if scores else None,
            "violated_fraction": float(np.mean(np.asarray(scores) > 0)) if scores else None,
            "passes_normal_evidence_gate": bool(
                known_fraction >= float(minimum_known_fraction)
            ),
        })
    return rows


def _outcome_summary(outcome: SearchOutcome) -> Dict[str, Any]:
    return {
        "method": outcome.method,
        "evaluated_genomes": outcome.evaluated_genomes,
        "selected": {
            "rule_ids": [rule.rule_id for rule in outcome.selected.rules],
            "L_normal": outcome.selected.normal_loss,
            "L_cf": outcome.selected.counterfactual_loss,
            "normal_known_fraction": outcome.selected.normal_known_fraction,
            "counterfactual_known_fraction": (
                outcome.selected.counterfactual_known_fraction
            ),
            "counterfactual_margins": list(outcome.selected.counterfactual_margins),
        },
        "pareto_front": [
            {
                "rule_ids": [rule.rule_id for rule in row.rules],
                "L_normal": row.normal_loss,
                "L_cf": row.counterfactual_loss,
            }
            for row in outcome.pareto_front
        ],
    }


def compare_search(evolved: SearchOutcome, oracle: SearchOutcome) -> Dict[str, Any]:
    def key(row: RuleSetEvaluation) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            (rule.rule_id, json.dumps(dict(rule.parameters), sort_keys=True))
            for rule in row.rules
        )

    oracle_front = {key(row) for row in oracle.pareto_front}
    evolved_front = {key(row) for row in evolved.pareto_front}
    intersection = oracle_front & evolved_front
    return {
        "oracle_front_size": len(oracle_front),
        "evolved_front_size": len(evolved_front),
        "front_intersection": len(intersection),
        "front_recall": (
            len(intersection) / len(oracle_front) if oracle_front else 0.0
        ),
        "selected_objectives_match": bool(
            evolved.selected.objectives == oracle.selected.objectives
        ),
        "selected_rule_ids_match": bool(
            [rule.rule_id for rule in evolved.selected.rules]
            == [rule.rule_id for rule in oracle.selected.rules]
        ),
        "selected_objectives": {
            "evolved": list(evolved.selected.objectives),
            "oracle": list(oracle.selected.objectives),
        },
        "oracle_front": _outcome_summary(oracle)["pareto_front"],
        "evolved_front": _outcome_summary(evolved)["pareto_front"],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    policy = report["decision"]
    lines = [
        f"# breakfast_box 规则进化搜索验证：{report['generated_at_utc']}",
        "",
        "## 结论",
        "",
        f"- Qwen 候选：{summary['candidate_count']} 条；可执行："
        f"{summary['executable_count']} 条；硬约束拒绝："
        f"{summary['rejected_candidate_count']} 条。",
        f"- 独立验证图（``L_normal``）：{summary['validation_image_count']} 张。",
        f"- 已确认反事实对（``L_cf``）：{summary['confirmed_pair_count']} 对"
        f"（候选 {summary['counterfactual_candidate_count']} 对，"
        f"拒绝 {summary['unconfirmed_pair_count']} 对）。",
        f"- 搜索是否启动：`{str(policy['search_started']).lower()}`。",
        f"- 选中规则集：{policy['selected_rule_ids']}",
        f"- ``L_normal`` = {policy['selected_L_normal']:.4f}，"
        f"``L_cf`` = {policy['selected_L_cf']:.4f}。",
        f"- GA 与 2^12-1 穷举 oracle："
        f"目标值一致 `{str(policy['search_matches_oracle_objectives']).lower()}`，"
        f"Pareto 前沿召回 = {policy['oracle_front_recall']:.3f}。",
        "",
        "## 逐规则证据",
        "",
        "| rule_id | predicate | 可执行 | 正常集已知率 | 正常集平均违反 | 可被反事实证伪 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rules"]:
        lines.append(
            "| {rule_id} | {predicate} | {executable} | {known} | {violation} | "
            "{falsifiable} |".format(
                rule_id=row["rule_id"],
                predicate=row["predicate"],
                executable=row["executable"],
                known=(
                    "n/a" if row.get("known_fraction") is None
                    else f"{row['known_fraction']:.3f}"
                ),
                violation=(
                    "n/a" if row.get("mean_violation_on_known") is None
                    else f"{row['mean_violation_on_known']:.3f}"
                ),
                falsifiable=row.get("falsifiable"),
            )
        )
    lines += [
        "",
        "## 反事实干预",
        "",
        "| 干预 | 基础图 | 确认 | 说明 |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["counterfactuals"]:
        lines.append(
            f"| {row['intervention']} | {Path(row['base_image']).name} | "
            f"{row['confirmed']} | `{row['edited_sha256'][:16]}` "
            f"({row['changed_pixels']} px) |"
        )
    lines += [
        "",
        "## 观测校准（train 正常图，与 support/val 均不重叠）",
        "",
        "| entity | detection_rate | count_stability | 精确计数可用 | 存在性可用 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in report["calibration"]["entities"].items():
        lines.append(
            f"| {name} | {row['detection_rate']:.3f} | {row['count_stability']:.3f} | "
            f"{report['calibration']['count_complete'][name]} | "
            f"{report['calibration']['presence_complete'][name]} |"
        )
    if report.get("posthoc_test"):
        post = report["posthoc_test"]
        lines += [
            "",
            "## 事后测试集评估（不参与搜索）",
            "",
            "| split | images | mean score | max score | score>0 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name in ("good", "logical_anomalies", "structural_anomalies"):
            row = post.get(name)
            if not row:
                continue
            lines.append(
                f"| {name} | {row['image_count']} | {row['mean_score']:.4f} | "
                f"{row['max_score']:.4f} | {row['fraction_scored_positive']:.3f} |"
            )
    lines += ["", "## 解释边界", ""] + [f"- {item}" for item in report["caveats"]]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    *,
    class_root: Path,
    support_paths: Sequence[Path],
    audit_json: Path,
    output_dir: Path,
    devices: Sequence[str],
    seeds: Sequence[int],
    population_size: int,
    generations: int,
    minimum_known_fraction: float,
    cf_margin: float,
    lambda_cf: float,
    base_image_count: int,
    max_validation: Optional[int],
    include_test_split: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "detection_cache"
    edited_dir = output_dir / "interventions"

    candidates, ledger = load_candidates(audit_json)
    spec = discover_breakfast_box(class_root, support_paths)
    registry = default_rule_registry()

    validation_paths = list(spec.validation_paths)
    if max_validation is not None:
        validation_paths = validation_paths[: int(max_validation)]

    builder = SceneFactBuilder(GroundingDinoDetector(device=devices[0]), SceneFactConfig())
    vocabulary_payload = {
        key: list(value) for key, value in builder.config.vocabulary.entity_types.items()
    }

    # --- stage 1: detections for the normal splits (+ held-out test splits) ---
    base_paths = list(spec.support_paths) + list(spec.calibration_paths) + validation_paths
    if include_test_split:
        base_paths += list(spec.logical_anomaly_paths)
        base_paths += list(spec.structural_anomaly_paths)
        base_paths += list(spec.good_test_paths)
    base_paths = list(dict.fromkeys(str(Path(p).resolve()) for p in base_paths))

    def progress(done: int, total: int) -> None:
        if done == total or done % 25 == 0:
            print(f"[detect] {done}/{total}", flush=True)

    records = detect_split(
        base_paths, cache_dir, devices=devices,
        vocabulary_payload=vocabulary_payload, progress=progress,
    )

    # --- severity calibration, independent of support and validation ---
    calibration = calibrate_split(
        builder,
        {str(path): records[str(Path(path).resolve())] for path in spec.calibration_paths},
        split="train_good_minus_supports",
        explicit_entities=tuple(builder.config.supertypes),
    )

    # --- counterfactual interventions on independent validation images ---
    intervention_bases = validation_paths[: max(1, int(base_image_count))]
    manifest = materialize_interventions(
        builder, intervention_bases,
        {
            str(Path(path).resolve()): records[str(Path(path).resolve())]
            for path in intervention_bases
        },
        edited_dir,
    )
    edited_records = detect_split(
        list(manifest), cache_dir, devices=devices,
        vocabulary_payload=vocabulary_payload, progress=progress,
    )
    pairs, counterfactual_audit = finalize_counterfactual_pairs(
        builder,
        {str(Path(path).resolve()): records[str(Path(path).resolve())]
         for path in intervention_bases},
        edited_records, manifest, calibration,
    )

    # --- facts ---
    validation_facts = assemble_facts(
        builder,
        {str(path): records[str(Path(path).resolve())] for path in validation_paths},
        calibration,
    )

    # --- per-rule evidence + falsifiability ---
    try:
        falsification = {
            row["rule_id"]: row for row in falsifiability_audit(candidates, pairs)
        }
    except Exception:  # pragma: no cover - defensive
        falsification = {}
    rule_rows = per_rule_normal_statistics(
        candidates, validation_facts, registry=registry,
        minimum_known_fraction=minimum_known_fraction,
    )
    for row in rule_rows:
        row.update({
            "falsifiable": falsification.get(row["rule_id"], {}).get("falsifiable"),
            "falsified_by": falsification.get(row["rule_id"], {}).get("falsified_by", []),
            "max_counterfactual_margin": falsification.get(
                row["rule_id"], {}
            ).get("max_margin"),
        })

    executable_rules = [
        rule for rule in candidates
        if registry.resolve(rule.canonical_predicate) is not None
        and not hard_constraint_reasons((rule,), registry)
    ]
    rejected = [
        {"rule_id": row["rule_id"], "reasons": row["hard_constraint_reasons"]}
        for row in ledger if not row["executable"]
    ]

    search_started = bool(executable_rules and validation_facts and pairs)
    # The search itself must decide feasibility; it raises if nothing is feasible.
    search_error = None
    evolved = None
    oracle = None
    comparison = None
    multi_seed = []
    posthoc = None
    config = SearchConfig(
        population_size=int(population_size),
        generations=int(generations),
        tolerance_grid=(0,),
        minimum_known_fraction=float(minimum_known_fraction),
        counterfactual_margin=float(cf_margin),
        lambda_cf=float(lambda_cf),
    )
    try:
        oracle = exhaustive_rule_search(
            executable_rules, list(validation_facts.values()), list(pairs),
            config=config, registry=registry,
        )
        evolved = EvolutionaryRuleSearch(config, registry).search(
            executable_rules, list(validation_facts.values()), list(pairs),
        )
        comparison = compare_search(evolved, oracle)
        for seed in seeds:
            variant = SearchConfig(
                population_size=config.population_size,
                generations=config.generations,
                tolerance_grid=config.tolerance_grid,
                minimum_known_fraction=config.minimum_known_fraction,
                counterfactual_margin=config.counterfactual_margin,
                lambda_cf=config.lambda_cf,
                seed=int(seed),
            )
            outcome = EvolutionaryRuleSearch(variant, registry).search(
                executable_rules, list(validation_facts.values()), list(pairs),
            )
            row = _outcome_summary(outcome)
            row["seed"] = int(seed)
            row["matches_oracle_objectives"] = bool(
                outcome.selected.objectives == oracle.selected.objectives
            )
            row["matches_oracle_rule_ids"] = bool(
                [rule.rule_id for rule in outcome.selected.rules]
                == [rule.rule_id for rule in oracle.selected.rules]
            )
            multi_seed.append(row)
    except Exception as error:  # fail closed, but keep the evidence
        search_error = f"{type(error).__name__}: {error}"
        search_started = False

    # --- post-hoc evaluation on held-out test splits (never used for search) ---
    if include_test_split and evolved is not None:
        selected = evolved.selected.rules
        try:
            posthoc = {}
            for name, paths in (
                ("good", spec.good_test_paths),
                ("logical_anomalies", spec.logical_anomaly_paths),
                ("structural_anomalies", spec.structural_anomaly_paths),
            ):
                facts = assemble_facts(
                    builder,
                    {str(path): records[str(Path(path).resolve())] for path in paths},
                    calibration,
                )
                posthoc[name] = evaluate_selected_on_split(
                    selected, facts, registry=registry,
                )
        except Exception as error:
            posthoc = {"error": f"{type(error).__name__}: {error}"}

    decision = {
        "search_started": bool(search_started and search_error is None),
        "search_error": search_error,
        "selected_rule_ids": (
            [] if evolved is None else [rule.rule_id for rule in evolved.selected.rules]
        ),
        "selected_L_normal": (
            None if evolved is None else evolved.selected.normal_loss
        ),
        "selected_L_cf": (
            None if evolved is None else evolved.selected.counterfactual_loss
        ),
        "search_matches_oracle_objectives": (
            None if comparison is None else comparison["selected_objectives_match"]
        ),
        "search_matches_oracle_rule_ids": (
            None if comparison is None else comparison["selected_rule_ids_match"]
        ),
        "oracle_front_recall": (
            None if comparison is None else comparison["front_recall"]
        ),
        "multi_seed_oracle_objective_match": [
            row["matches_oracle_objectives"] for row in multi_seed
        ],
    }

    report = {
        "schema_version": "breakfast-box-rule-evolution-verification-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective_contract": {
            "optimized": ["L_normal", "L_cf"],
            "hard_constraints": [
                "structural_validity", "exact_executor_available",
                "parameter_legality", "dependency_closure",
                "proven_exact_nonredundancy", "unconditional_noncontradiction",
            ],
            "unknown_policy": "UNKNOWN never counts as satisfied",
            "counterfactual_policy": (
                "pixel-level semantic intervention on a real normal image, "
                "re-detected by the same perception stack and independently "
                "confirmed; rule truth is never flipped by hand"
            ),
        },
        "dataset": spec.to_dict(),
        "support_sha256": {
            Path(path).name: sha256_file(path) for path in spec.support_paths
        },
        "summary": {
            "candidate_count": len(candidates),
            "executable_count": len(executable_rules),
            "rejected_candidate_count": len(rejected),
            "validation_image_count": len(validation_facts),
            "counterfactual_candidate_count": len(counterfactual_audit),
            "confirmed_pair_count": len(pairs),
            "unconfirmed_pair_count": len(counterfactual_audit) - len(pairs),
        },
        "decision": decision,
        "candidates": ledger,
        "rejected_candidates": rejected,
        "calibration": calibration.to_dict(),
        "rules": rule_rows,
        "counterfactuals": [row.to_dict() for row in counterfactual_audit],
        "search": {
            "config": {
                "population_size": config.population_size,
                "generations": config.generations,
                "tolerance_grid": list(config.tolerance_grid),
                "minimum_known_fraction": config.minimum_known_fraction,
                "counterfactual_margin": config.counterfactual_margin,
                "lambda_cf": config.lambda_cf,
            },
            "oracle": None if oracle is None else _outcome_summary(oracle),
            "evolved": None if evolved is None else _outcome_summary(evolved),
            "comparison": comparison,
            "multi_seed": multi_seed,
        },
        "posthoc_test": posthoc,
        "caveats": [
            "规则由 Qwen 4-shot 提出，但原始 raw response 缺失；候选集由 "
            "audit.json 中的 qwen_arguments/qwen_parameters 逐字重建。",
            "实例 mask 是检测框实心区域，因此只声称框几何/计数/区域包含/质心关系可用。",
            "L_cf 的反事实是像素级编辑后重检测的结果，不是人工真值翻转；"
            "只有独立确认器通过的干预才进入优化。",
            "left_of(left_compartment, right_compartment) 与 "
            "disjoint(left_compartment, right_compartment) 由几何定义恒真，"
            "任何视觉干预都无法证伪它们；见 falsifiable 字段。",
            "事后测试集指标只用于报告，不参与搜索，也不用于调阈值。",
        ],
    }

    (output_dir / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "verification.md").write_text(_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-root", type=Path, default=DEFAULT_CLASS_ROOT)
    parser.add_argument("--support", type=Path, nargs="+", required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 668])
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--minimum-known-fraction", type=float, default=0.95)
    parser.add_argument("--cf-margin", type=float, default=0.5)
    parser.add_argument("--lambda-cf", type=float, default=1.0)
    parser.add_argument("--base-image-count", type=int, default=2)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--include-test-split", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(
        class_root=args.class_root,
        support_paths=args.support,
        audit_json=args.audit_json,
        output_dir=args.output_dir,
        devices=args.devices,
        seeds=args.seeds,
        population_size=args.population_size,
        generations=args.generations,
        minimum_known_fraction=args.minimum_known_fraction,
        cf_margin=args.cf_margin,
        lambda_cf=args.lambda_cf,
        base_image_count=args.base_image_count,
        max_validation=args.max_validation,
        include_test_split=args.include_test_split,
    )
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
