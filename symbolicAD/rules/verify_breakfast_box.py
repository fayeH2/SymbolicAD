#!/usr/bin/env python3
"""End-to-end verification of the breakfast_box rule-evolution search.

This driver answers one question with executable evidence:

    Given the four Qwen support images, can the proposed rule set be executed on
    the predeclared normal facts (``L_normal``), does it react to
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
from symbolicAD.reasoning.constraint_executor import (
    ConstraintOperator, combine_statuses,
)
from symbolicAD.reasoning.violation_result import RuleStatus
from symbolicAD.evaluation.loco_rule_evaluation import evaluate_rule_sets
from symbolicAD.evaluation.protocol_admission import (
    assess_strict_kshot_protocol,
    validate_qwen_audit_provenance,
)
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
from symbolicAD.rules.rule_validator import audit_rule_entity_identity
from symbolicAD.rules.search.symbolicdet_evolution import (
    CounterfactualPair, EvolutionaryRuleSearch, RuleSetEvaluation, SearchConfig,
    SearchOutcome, candidate_structurally_valid, exhaustive_rule_search,
    greedy_rule_search, rule_set_phenotype,
    hard_constraint_reasons,
)


DEFAULT_CLASS_ROOT = Path(
    "/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO/breakfast_box"
)


# ---------------------------------------------------------------------------
# Candidate ledger
# ---------------------------------------------------------------------------


def load_candidates(
    audit_path: Path,
    *,
    replay_payloads: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[Tuple[RuleSpec, ...], List[Dict[str, Any]]]:
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    payloads = (
        tuple(dict(row) for row in replay_payloads)
        if replay_payloads is not None else payloads_from_audit(report)
    )
    adaptations = adapt_qwen_candidates(payloads)
    registry = default_rule_registry()
    rules: List[RuleSpec] = []
    ledger: List[Dict[str, Any]] = []
    adapted_rules = [
        adaptation.rule for adaptation in adaptations
        if adaptation.rule is not None
    ]
    candidate_ids = {rule.rule_id for rule in adapted_rules}
    for payload, adaptation in zip(payloads, adaptations):
        reasons = []
        if adaptation.rule is None:
            reasons.append(adaptation.reason_code)
        else:
            if not candidate_structurally_valid(adaptation.rule):
                reasons.append("STRUCTURALLY_INVALID")
            if registry.resolve(adaptation.rule.canonical_predicate) is None:
                reasons.append(
                    f"UNSUPPORTED_PREDICATE:{adaptation.rule.canonical_predicate}"
                )
            missing = sorted(set(adaptation.rule.dependencies) - candidate_ids)
            reasons.extend(f"MISSING_DEPENDENCY:{value}" for value in missing)
        row = {
            "rule_id": adaptation.source_rule_id,
            "description": payload.get("description", ""),
            "qwen_predicate": adaptation.source_predicate,
            "adaptation_status": adaptation.status.value,
            "adaptation_reason": adaptation.reason_code,
            "typed_rule": None if adaptation.rule is None else adaptation.rule.to_dict(),
            "executable": bool(adaptation.rule is not None and not reasons),
            "hard_constraint_reasons": reasons,
        }
        ledger.append(row)
        if adaptation.rule is not None:
            rules.append(adaptation.rule)
    return tuple(rules), ledger


def _baseline_manifest_provenance(
    manifest_path: Optional[Path],
) -> Tuple[Optional[Tuple[str, ...]], bool, Dict[str, Any]]:
    """Verify support bytes and checkpoint bytes recorded by the baseline."""
    if manifest_path is None:
        return None, False, {"valid": False, "reason_codes": ["NOT_SUPPLIED"]}
    path = manifest_path.expanduser().resolve()
    reasons = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, False, {"valid": False, "reason_codes": ["UNREADABLE_MANIFEST"]}
    supports = [Path(str(value)).expanduser().resolve() for value in payload.get("support", [])]
    recorded_hashes = payload.get("support_sha256")
    if not supports or not isinstance(recorded_hashes, Mapping):
        reasons.append("SUPPORT_MANIFEST_MISSING")
    current_hashes = []
    for support in supports:
        if not support.is_file():
            reasons.append("BASELINE_SUPPORT_FILE_MISSING")
            continue
        digest = sha256_file(support)
        current_hashes.append(digest)
        if recorded_hashes.get(str(support)) != digest:
            reasons.append("BASELINE_SUPPORT_HASH_MISMATCH")
    checkpoint = Path(str(payload.get("checkpoint", ""))).expanduser()
    recorded_checkpoint_hash = payload.get("checkpoint_sha256")
    current_checkpoint_hash = None
    if not checkpoint.is_file():
        reasons.append("BASELINE_CHECKPOINT_MISSING")
    else:
        current_checkpoint_hash = sha256_file(checkpoint)
        if not recorded_checkpoint_hash:
            reasons.append("BASELINE_CHECKPOINT_SHA256_MISSING")
        elif recorded_checkpoint_hash != current_checkpoint_hash:
            reasons.append("BASELINE_CHECKPOINT_HASH_MISMATCH")
    reasons = list(dict.fromkeys(reasons))
    details = {
        "valid": not reasons,
        "reason_codes": reasons,
        "manifest": str(path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": current_checkpoint_hash,
    }
    return tuple(current_hashes) if current_hashes else None, not reasons, details


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _rule_set_score(
    rules: Sequence[RuleSpec],
    context: ExecutionContext,
    registry,
) -> Tuple[Optional[float], int, int, RuleStatus]:
    """Strong-Kleene AND score; UNKNOWN is an explicit abstention."""
    results = []
    known = 0
    for rule in rules:
        capability = registry.resolve(rule.canonical_predicate)
        if capability is None:
            continue
        result = capability.executor.execute(rule, context)
        results.append(result)
        known += int(result.status != RuleStatus.UNKNOWN)
    status_values = [row.status for row in results]
    status_values.extend(
        RuleStatus.UNKNOWN for _ in range(len(rules) - len(results))
    )
    status = combine_statuses(status_values, ConstraintOperator.AND)
    if status == RuleStatus.VIOLATED:
        score: Optional[float] = max(
            float(row.violation_score) for row in results
            if row.status == RuleStatus.VIOLATED
        )
    elif status == RuleStatus.SATISFIED:
        score = 0.0
    else:
        score = None
    return score, known, len(rules), status


def evaluate_selected_on_split(
    rules: Sequence[RuleSpec],
    facts: Mapping[str, ExecutionContext],
    *,
    registry,
) -> Dict[str, Any]:
    rows = []
    for path, context in facts.items():
        score, known, total, status = _rule_set_score(rules, context, registry)
        rows.append({
            "image": Path(path).name,
            "score": score,
            "status": status.value,
            "known_rules": known,
            "total_rules": total,
            "known_fraction": (known / total) if total else 0.0,
        })
    decided_scores = np.asarray([
        row["score"] for row in rows if row["score"] is not None
    ], dtype=float)
    known_fractions = np.asarray([row["known_fraction"] for row in rows], dtype=float)
    status_counts = {
        status.value: int(sum(row["status"] == status.value for row in rows))
        for status in RuleStatus
    }
    return {
        "image_count": len(rows),
        "decided_count": int(decided_scores.size),
        "decision_coverage": (
            float(decided_scores.size / len(rows)) if rows else 0.0
        ),
        "status_counts": status_counts,
        "mean_score_on_decided": (
            float(decided_scores.mean()) if decided_scores.size else None
        ),
        "max_score_on_decided": (
            float(decided_scores.max()) if decided_scores.size else None
        ),
        "fraction_decided_positive": (
            float(np.mean(decided_scores > 0)) if decided_scores.size else None
        ),
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
            "rules": [rule.to_dict() for rule in outcome.selected.rules],
            "phenotype": [list(row) for row in rule_set_phenotype(
                outcome.selected.rules
            )],
            "L_normal": outcome.selected.normal_loss,
            "L_cf": outcome.selected.counterfactual_loss,
            "normal_known_fraction": outcome.selected.normal_known_fraction,
            "counterfactual_known_fraction": (
                outcome.selected.counterfactual_known_fraction
            ),
            "counterfactual_margins": list(outcome.selected.counterfactual_margins),
            "rule_known_fractions": dict(outcome.selected.rule_known_fractions),
            "rule_counterfactual_paired_fractions": dict(
                outcome.selected.rule_counterfactual_paired_fractions
            ),
        },
        "pareto_front": [
            {
                "rule_ids": [rule.rule_id for rule in row.rules],
                "rules": [rule.to_dict() for rule in row.rules],
                "phenotype": [list(value) for value in rule_set_phenotype(row.rules)],
                "L_normal": row.normal_loss,
                "L_cf": row.counterfactual_loss,
            }
            for row in outcome.pareto_front
        ],
    }


def compare_search(evolved: SearchOutcome, oracle: SearchOutcome) -> Dict[str, Any]:
    def key(row: RuleSetEvaluation):
        return rule_set_phenotype(row.rules)

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
        "selected_phenotype_match": bool(
            rule_set_phenotype(evolved.selected.rules)
            == rule_set_phenotype(oracle.selected.rules)
        ),
        "selected_objectives": {
            "evolved": list(evolved.selected.objectives),
            "oracle": list(oracle.selected.objectives),
        },
        "oracle_front": _outcome_summary(oracle)["pareto_front"],
        "evolved_front": _outcome_summary(evolved)["pareto_front"],
    }


def _assess_rule_module_admission(
    *,
    protocol_admission,
    evolved: Optional[SearchOutcome],
    greedy: Optional[SearchOutcome],
    multi_seed: Sequence[Mapping[str, Any]],
    requested_seed_count: int,
    posthoc: Optional[Mapping[str, Any]],
    lambda_cf: float,
) -> Dict[str, Any]:
    """Pre-registered gate before a rule branch may enter fusion ablations.

    Passing this gate still does not modify the main model.  It only authorizes
    a later, isolated baseline+rule fusion comparison.
    """
    reasons = []
    if not protocol_admission.strict_kshot_protocol_eligible:
        reasons.append("STRICT_KSHOT_PROTOCOL_FAILED")
    if evolved is None or greedy is None:
        reasons.append("SEARCH_OR_GREEDY_RESULT_MISSING")
    elif not (
        evolved.selected.scalar(lambda_cf)
        < greedy.selected.scalar(lambda_cf) - 1e-12
    ):
        reasons.append("EVOLUTION_NOT_BETTER_THAN_GREEDY_OBJECTIVE")
    if len(multi_seed) != int(requested_seed_count) or not all(
        bool(row.get("beats_greedy_scalar")) for row in multi_seed
    ):
        reasons.append("EVOLUTION_NOT_STABLE_ACROSS_ALL_SEEDS")
    formal = None if not isinstance(posthoc, Mapping) else posthoc.get(
        "formal_rule_evaluation"
    )
    if not isinstance(formal, Mapping):
        reasons.append("FROZEN_TEST_EVALUATION_MISSING")
    else:
        sets = formal.get("rule_sets") or {}
        evolved_row = sets.get("evolved")
        greedy_row = sets.get("greedy")
        if not isinstance(evolved_row, Mapping) or not isinstance(greedy_row, Mapping):
            reasons.append("EVOLVED_OR_GREEDY_TEST_METRICS_MISSING")
        else:
            coverage = float(evolved_row.get("decision_coverage", 0.0))
            if coverage < 0.95:
                reasons.append("DECISION_COVERAGE_BELOW_0_95")
            evolved_metrics = evolved_row.get("metrics") or {}
            greedy_metrics = greedy_row.get("metrics") or {}
            for group in ("all", "logical", "structural"):
                if float((evolved_metrics.get(group) or {}).get(
                    "evaluation_coverage", 0.0
                )) < 0.95:
                    reasons.append(f"{group.upper()}_METRIC_COVERAGE_BELOW_0_95")

            def metric(rows: Mapping[str, Any], group: str) -> Optional[float]:
                value = (rows.get(group) or {}).get("image_auroc")
                return None if value is None else float(value)

            evolved_all = metric(evolved_metrics, "all")
            evolved_logical = metric(evolved_metrics, "logical")
            evolved_structural = metric(evolved_metrics, "structural")
            greedy_all = metric(greedy_metrics, "all")
            greedy_logical = metric(greedy_metrics, "logical")
            greedy_structural = metric(greedy_metrics, "structural")
            if evolved_all is None or evolved_all < 0.80:
                reasons.append("EVOLVED_OVERALL_AUROC_BELOW_0_80")
            if None in (evolved_all, greedy_all) or evolved_all < greedy_all + 0.01:
                reasons.append("OVERALL_AUROC_GAIN_BELOW_0_01_VS_GREEDY")
            if (
                None in (evolved_logical, greedy_logical)
                or evolved_logical < greedy_logical + 0.01
            ):
                reasons.append("LOGICAL_AUROC_GAIN_BELOW_0_01_VS_GREEDY")
            if (
                None in (evolved_structural, greedy_structural)
                or evolved_structural < greedy_structural - 0.01
            ):
                reasons.append("STRUCTURAL_AUROC_DROP_EXCEEDS_0_01")
            evolved_spro = ((evolved_row.get("spro") or {}).get("logic_total") or {})
            greedy_spro = ((greedy_row.get("spro") or {}).get("logic_total") or {})
            if float(evolved_spro.get("logical_spro_5fpr", -1.0)) < float(
                greedy_spro.get("logical_spro_5fpr", 0.0)
            ) + 0.01:
                reasons.append("LOGICAL_SPRO_GAIN_BELOW_0_01_VS_GREEDY")
            if float(evolved_spro.get("mean_spro_5fpr", -1.0)) < float(
                greedy_spro.get("mean_spro_5fpr", 0.0)
            ) - 0.01:
                reasons.append("MEAN_SPRO_DROP_EXCEEDS_0_01")
    reasons = list(dict.fromkeys(reasons))
    return {
        "gate_version": "rule-branch-before-fusion-v1",
        "eligible_for_isolated_fusion_ablation": not reasons,
        "main_model_admitted": False,
        "main_model_reason": "FUSION_ABLATION_NOT_RUN",
        "reason_codes": reasons,
        "thresholds": {
            "minimum_decision_and_metric_coverage": 0.95,
            "minimum_overall_auroc": 0.80,
            "minimum_overall_and_logical_auroc_gain_vs_greedy": 0.01,
            "minimum_logical_spro_gain_vs_greedy": 0.01,
            "maximum_structural_auroc_and_mean_spro_drop": 0.01,
            "all_requested_seeds_must_beat_greedy": True,
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    policy = report["decision"]
    def metric(value: Any, digits: int = 4) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}f}"

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
        f"- 搜索是否尝试：`{str(policy['search_attempted']).lower()}`；"
        f"是否成功：`{str(policy['search_succeeded']).lower()}`。",
        f"- 严格 K=4 主模型协议资格："
        f"`{str(report['protocol_admission']['strict_kshot_protocol_eligible']).lower()}`；"
        f"原因：{report['protocol_admission']['reason_codes']}。",
        f"- 选中规则集：{policy['selected_rule_ids']}",
        f"- ``L_normal`` = {metric(policy['selected_L_normal'])}，"
        f"``L_cf`` = {metric(policy['selected_L_cf'])}。",
        f"- 简单前向贪心对照：``L_normal`` = "
        f"{metric(policy['greedy_L_normal'])}，``L_cf`` = "
        f"{metric(policy['greedy_L_cf'])}；GA 严格降低同一标量目标："
        f"`{str(policy['evolution_strictly_beats_greedy_scalar']).lower()}`。",
        f"- GA 与全部合法 add/drop/tolerance 状态的穷举 oracle："
        f"目标值一致 `{str(policy['search_matches_oracle_objectives']).lower()}`，"
        f"表型一致 `{str(policy['search_matches_oracle_phenotype']).lower()}`，"
        f"Pareto 前沿召回 = {metric(policy['oracle_front_recall'], 3)}。",
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
        f"## 观测校准（{report['split_protocol']['calibration_source']}）",
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
            "| split | images | decision coverage | mean score (decided) | status counts |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name in ("good", "logical_anomalies", "structural_anomalies"):
            row = post.get(name)
            if not row:
                continue
            lines.append(
                f"| {name} | {row['image_count']} | "
                f"{row['decision_coverage']:.3f} | "
                f"{metric(row['mean_score_on_decided'])} | "
                f"{row['status_counts']} |"
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
    calibration_image_count: int,
    normal_fit_count: int,
    counterfactual_source_count: int,
    max_validation: Optional[int],
    include_test_split: bool,
    baseline_training_manifest: Optional[Path] = None,
    protocol_mode: str = "validation_assisted",
    entity_identity_gate: bool = False,
    entity_identity_min_support: float = 0.75,
    counterfactual_policy: str = "generic",
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "detection_cache"
    edited_dir = output_dir / "interventions"

    if protocol_mode not in {"validation_assisted", "strict_k4"}:
        raise ValueError(
            "protocol_mode must be 'validation_assisted' or 'strict_k4'"
        )
    if protocol_mode == "strict_k4" and baseline_training_manifest is None:
        raise ValueError(
            "strict_k4 requires --baseline-training-manifest so support and "
            "checkpoint provenance can be verified"
        )
    spec = discover_breakfast_box(class_root, support_paths)
    audit_report = json.loads(
        audit_json.expanduser().resolve().read_text(encoding="utf-8")
    )
    qwen_provenance = validate_qwen_audit_provenance(
        audit_report, category=spec.class_name,
        support_paths=spec.support_paths,
    )
    candidates, ledger = load_candidates(
        audit_json,
        replay_payloads=(
            qwen_provenance.fresh_payloads if qwen_provenance.valid else None
        ),
    )
    registry = default_rule_registry()

    if protocol_mode == "strict_k4":
        # The same four images drive LLM proposal, perception calibration,
        # support consistency and support-derived counterfactuals.  This is a
        # stricter data budget, but L_normal is no longer an independent-val
        # estimate; that limitation is explicit in the report.
        calibration_paths = list(spec.support_paths)
        validation_paths = list(spec.support_paths)
        counterfactual_source_paths = list(spec.support_paths)
        audit_paths = []
        normal_fit_source = "declared_k4_supports"
        calibration_source = "declared_k4_supports"
    else:
        # Content-hash ordering makes the three validation roles immutable
        # under filename renaming. N_AUDIT executes only after selection.
        ordered_validation = sorted(spec.validation_paths, key=sha256_file)
        fit_end = min(int(normal_fit_count), len(ordered_validation))
        cf_end = min(
            fit_end + int(counterfactual_source_count), len(ordered_validation)
        )
        validation_paths = list(ordered_validation[:fit_end])
        counterfactual_source_paths = list(ordered_validation[fit_end:cf_end])
        audit_paths = list(ordered_validation[cf_end:])
        if max_validation is not None:
            validation_paths = validation_paths[: int(max_validation)]
        ordered_calibration = sorted(spec.calibration_paths, key=sha256_file)
        calibration_paths = ordered_calibration[
            : max(1, int(calibration_image_count))
        ]
        normal_fit_source = "independent_normal_validation"
        calibration_source = "train_good_minus_supports"

    rule_support_hashes = {sha256_file(path) for path in spec.support_paths}
    qwen_support_hash_sets = []
    for rule in candidates:
        evidence_rows = rule.support_evidence.get("qwen_support_evidence", [])
        hashes = []
        for evidence in evidence_rows:
            path = Path(str(evidence.get("support_image", ""))).expanduser()
            if path.is_file():
                hashes.append(sha256_file(path))
        qwen_support_hash_sets.append(tuple(hashes))
    (
        baseline_support_hashes,
        baseline_artifact_valid,
        baseline_provenance,
    ) = _baseline_manifest_provenance(baseline_training_manifest)
    extra_normal_hashes = {
        sha256_file(path) for path in (
            list(calibration_paths) + list(validation_paths)
            + list(counterfactual_source_paths)
        )
    }
    protocol_admission = assess_strict_kshot_protocol(
        rule_support_hashes=rule_support_hashes,
        qwen_support_hash_sets=qwen_support_hash_sets,
        qwen_audit_provenance_valid=qwen_provenance.valid,
        baseline_support_hashes=baseline_support_hashes,
        baseline_artifact_valid=baseline_artifact_valid,
        extra_normal_hashes=extra_normal_hashes,
        test_detection_after_rule_freeze=True,
        expected_shot=4,
    )
    if protocol_mode == "strict_k4" and not protocol_admission.strict_kshot_protocol_eligible:
        rejection = {
            "status": "REJECTED_BEFORE_DETECTION",
            "protocol_admission": protocol_admission.to_dict(),
            "qwen_audit_provenance": qwen_provenance.to_dict(),
            "baseline_provenance": baseline_provenance,
        }
        (output_dir / "preflight_rejection.json").write_text(
            json.dumps(rejection, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "strict_k4 preflight rejected: "
            + ", ".join(protocol_admission.reason_codes)
        )

    builder = SceneFactBuilder(GroundingDinoDetector(device=devices[0]), SceneFactConfig())
    vocabulary_payload = {
        key: list(value) for key, value in builder.config.vocabulary.entity_types.items()
    }

    # --- stage 1: normal-only detections used by calibration/search ---
    base_paths = (
        list(spec.support_paths) + calibration_paths + validation_paths
        + counterfactual_source_paths
    )
    base_paths = list(dict.fromkeys(str(Path(p).resolve()) for p in base_paths))

    def progress(done: int, total: int) -> None:
        if done == total or done % 25 == 0:
            print(f"[detect] {done}/{total}", flush=True)

    records = detect_split(
        base_paths, cache_dir, devices=devices,
        vocabulary_payload=vocabulary_payload, progress=progress,
    )

    # --- optional controlled variable: exact entity-identity gate ---
    # Qwen may name a visually similar but semantically different object.  The
    # gate can reject that rule, but it must never rename the object or broaden
    # the vocabulary.  It uses only the already declared K-shot support.
    support_records = [
        records[str(Path(path).resolve())] for path in spec.support_paths
    ]
    identity_audits = {
        rule.rule_id: audit_rule_entity_identity(
            rule,
            support_records,
            vocabulary_entities=tuple(builder.config.vocabulary.entity_types),
            supertype_members=builder.config.supertypes,
            derived_entities=(
                builder.config.left_entity, builder.config.right_entity,
            ),
            minimum_presence_fraction=float(entity_identity_min_support),
        )
        for rule in candidates
    }
    pre_identity_executable_count = sum(row["executable"] for row in ledger)
    for row in ledger:
        audit = identity_audits.get(row["rule_id"])
        row["entity_identity_audit"] = None if audit is None else audit.to_dict()
        if entity_identity_gate and audit is not None and not audit.accepted:
            row["hard_constraint_reasons"] = list(dict.fromkeys(
                list(row["hard_constraint_reasons"]) + list(audit.reason_codes)
            ))
            row["executable"] = False
    active_candidates = tuple(
        rule for rule in candidates
        if not entity_identity_gate or identity_audits[rule.rule_id].accepted
    )

    # --- severity calibration, independent of support and validation ---
    calibration = calibrate_split(
        builder,
        {str(path): records[str(Path(path).resolve())] for path in calibration_paths},
        split=calibration_source,
        explicit_entities=tuple(builder.config.supertypes),
    )

    # --- counterfactual interventions on independent validation images ---
    intervention_bases = counterfactual_source_paths[: max(1, int(base_image_count))]
    manifest = materialize_interventions(
        builder, intervention_bases,
        {
            str(Path(path).resolve()): records[str(Path(path).resolve())]
            for path in intervention_bases
        },
        edited_dir,
        policy=counterfactual_policy,
        rules=active_candidates,
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
            row["rule_id"]: row
            for row in falsifiability_audit(active_candidates, pairs)
        }
    except Exception:  # pragma: no cover - defensive
        falsification = {}
    rule_rows = per_rule_normal_statistics(
        active_candidates, validation_facts, registry=registry,
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
        rule for rule in active_candidates
        if not hard_constraint_reasons((rule,), registry)
    ]
    rejected = [
        {"rule_id": row["rule_id"], "reasons": row["hard_constraint_reasons"]}
        for row in ledger if not row["executable"]
    ]

    search_attempted = bool(executable_rules and validation_facts and pairs)
    search_succeeded = False
    # The search itself must decide feasibility; it raises if nothing is feasible.
    search_error = None
    evolved = None
    oracle = None
    greedy = None
    comparison = None
    multi_seed = []
    posthoc = None
    config = SearchConfig(
        population_size=int(population_size),
        generations=int(generations),
        # Legal count tolerances are an actual search operation.  Non-count
        # predicates keep a binary add/drop gene and do not duplicate states.
        tolerance_grid=(0, 1),
        minimum_known_fraction=float(minimum_known_fraction),
        counterfactual_margin=float(cf_margin),
        lambda_cf=float(lambda_cf),
    )
    try:
        if not search_attempted:
            raise ValueError("search prerequisites are empty")
        oracle = exhaustive_rule_search(
            executable_rules, list(validation_facts.values()), list(pairs),
            config=config, registry=registry,
        )
        greedy = greedy_rule_search(
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
            row["matches_oracle_phenotype"] = bool(
                rule_set_phenotype(outcome.selected.rules)
                == rule_set_phenotype(oracle.selected.rules)
            )
            row["beats_greedy_scalar"] = bool(
                outcome.selected.scalar(config.lambda_cf)
                < greedy.selected.scalar(config.lambda_cf) - 1e-12
            )
            multi_seed.append(row)
        search_succeeded = True
    except Exception as error:  # fail closed, but keep the evidence
        search_error = f"{type(error).__name__}: {error}"

    # --- post-hoc evaluation on held-out test splits (never used for search) ---
    if include_test_split and evolved is not None:
        selected = evolved.selected.rules
        try:
            # Strict ordering: no test pixel is detected until the selected
            # phenotype and exhaustive-oracle comparison are frozen.
            test_paths = list(spec.good_test_paths)
            test_paths += list(spec.logical_anomaly_paths)
            test_paths += list(spec.structural_anomaly_paths)
            test_records = detect_split(
                test_paths, cache_dir, devices=devices,
                vocabulary_payload=vocabulary_payload, progress=progress,
            )
            records.update(test_records)
            posthoc = {}
            posthoc_facts = {}
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
                posthoc_facts[name] = facts
            # This is the only call site that receives LOCO labels/GT masks.
            # It runs after ``evolved`` and the oracle comparison are frozen.
            posthoc["formal_rule_evaluation"] = evaluate_rule_sets(
                category=spec.class_name,
                split_facts=posthoc_facts,
                rule_sets={
                    "raw_qwen_executable": tuple(executable_rules),
                    "greedy": tuple(greedy.selected.rules),
                    "evolved": tuple(selected),
                },
                output_dir=output_dir / "test_evaluation",
                registry=registry,
            )
        except Exception as error:
            posthoc = {"error": f"{type(error).__name__}: {error}"}

    # Frozen-rule normal audit: these images never affect L_normal, L_cf,
    # tolerance selection or Pareto ranking.
    normal_audit = None
    if evolved is not None and audit_paths:
        audit_records = detect_split(
            audit_paths, cache_dir, devices=devices,
            vocabulary_payload=vocabulary_payload, progress=progress,
        )
        records.update(audit_records)
        audit_facts = assemble_facts(
            builder,
            {str(path): records[str(Path(path).resolve())] for path in audit_paths},
            calibration,
        )
        normal_audit = evaluate_selected_on_split(
            evolved.selected.rules, audit_facts, registry=registry,
        )

    decision = {
        "search_attempted": search_attempted,
        "search_succeeded": search_succeeded,
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
        "greedy_rule_ids": (
            [] if greedy is None else [rule.rule_id for rule in greedy.selected.rules]
        ),
        "greedy_L_normal": (
            None if greedy is None else greedy.selected.normal_loss
        ),
        "greedy_L_cf": (
            None if greedy is None else greedy.selected.counterfactual_loss
        ),
        "evolution_strictly_beats_greedy_scalar": (
            None if evolved is None or greedy is None else bool(
                evolved.selected.scalar(config.lambda_cf)
                < greedy.selected.scalar(config.lambda_cf) - 1e-12
            )
        ),
        "search_matches_oracle_objectives": (
            None if comparison is None else comparison["selected_objectives_match"]
        ),
        "search_matches_oracle_rule_ids": (
            None if comparison is None else comparison["selected_rule_ids_match"]
        ),
        "search_matches_oracle_phenotype": (
            None if comparison is None else comparison["selected_phenotype_match"]
        ),
        "oracle_front_recall": (
            None if comparison is None else comparison["front_recall"]
        ),
        "multi_seed_oracle_objective_match": [
            row["matches_oracle_objectives"] for row in multi_seed
        ],
    }
    module_admission = _assess_rule_module_admission(
        protocol_admission=protocol_admission,
        evolved=evolved,
        greedy=greedy,
        multi_seed=multi_seed,
        requested_seed_count=len(seeds),
        posthoc=posthoc,
        lambda_cf=config.lambda_cf,
    )

    report = {
        "schema_version": "breakfast-box-rule-evolution-verification-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective_contract": {
            "optimized": ["L_normal", "L_cf"],
            "hard_constraints": [
                "structural_validity", "exact_executor_available",
                "parameter_legality", "dependency_closure",
                "proven_logical_nonredundancy", "unconditional_noncontradiction",
                *(["exact_support_entity_identity"] if entity_identity_gate else []),
            ],
            "unknown_policy": "UNKNOWN never counts as satisfied",
            "counterfactual_policy": (
                "pixel-level semantic intervention on a real normal image, "
                "re-detected and confirmed by a rule-independent check in the "
                "same perception stack; rule truth is never flipped by hand"
            ),
            "counterfactual_intervention_family": counterfactual_policy,
        },
        "dataset": spec.to_dict(),
        "split_protocol": {
            "protocol_mode": protocol_mode,
            "ordering": "sha256_ascending",
            "calibration_source": calibration_source,
            "normal_fit_source": normal_fit_source,
            "calibration_count": len(calibration_paths),
            "normal_fit_count": len(validation_paths),
            "counterfactual_source_count": len(counterfactual_source_paths),
            "counterfactual_policy": counterfactual_policy,
            "normal_audit_count": len(audit_paths),
            "test_detection_after_rule_freeze": True,
            "normal_audit_detection_after_rule_freeze": True,
            "normal_fit_sha256": [sha256_file(path) for path in validation_paths],
            "counterfactual_source_sha256": [
                sha256_file(path) for path in counterfactual_source_paths
            ],
            "normal_audit_sha256": [sha256_file(path) for path in audit_paths],
        },
        "support_sha256": {
            Path(path).name: sha256_file(path) for path in spec.support_paths
        },
        "baseline_training_manifest": (
            None if baseline_training_manifest is None
            else str(baseline_training_manifest.expanduser().resolve())
        ),
        "protocol_admission": protocol_admission.to_dict(),
        "qwen_audit_provenance": qwen_provenance.to_dict(),
        "baseline_provenance": baseline_provenance,
        "entity_identity_gate": {
            "enabled": bool(entity_identity_gate),
            "policy": "exact_name_only_no_alias_or_automatic_correction",
            "minimum_support_presence_fraction": float(
                entity_identity_min_support
            ),
            "audits": [
                identity_audits[rule.rule_id].to_dict() for rule in candidates
            ],
        },
        "summary": {
            "candidate_count": len(ledger),
            "typed_candidate_count": len(candidates),
            "pre_identity_gate_executable_count": pre_identity_executable_count,
            "identity_gate_rejected_count": sum(
                1 for row in ledger
                if row.get("entity_identity_audit")
                and not row["entity_identity_audit"]["accepted"]
            ) if entity_identity_gate else 0,
            "executable_count": len(executable_rules),
            "rejected_candidate_count": len(rejected),
            "validation_image_count": len(validation_facts),
            "normal_audit_image_count": len(audit_paths),
            "counterfactual_candidate_count": len(counterfactual_audit),
            "confirmed_pair_count": len(pairs),
            "unconfirmed_pair_count": len(counterfactual_audit) - len(pairs),
        },
        "decision": decision,
        "module_admission": module_admission,
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
            "greedy_control": (
                None if greedy is None else _outcome_summary(greedy)
            ),
            "evolved": None if evolved is None else _outcome_summary(evolved),
            "comparison": comparison,
            "multi_seed": multi_seed,
        },
        "posthoc_test": posthoc,
        "normal_audit_after_freeze": normal_audit,
        "caveats": [
            (
                "严格候选由 Qwen raw_response 完整 JSON 重放，并与 raw/typed "
                "ledger、support、prompt、模型权重及生成源码哈希逐项一致。"
                if qwen_provenance.valid else
                "Qwen audit provenance 未通过；本次仅为诊断实验，候选不得导出到主模型。"
            ),
            "实例 mask 是检测框实心区域，因此只声称框几何/计数/区域包含/质心关系可用。",
            "L_cf 的反事实是像素级编辑后重检测的结果，不是人工真值翻转；"
            "只有与规则真值无关的同一检测栈确认通过后才进入优化；"
            "它不是第二视觉模型的独立确认。",
            (
                "rule_matched_count_erase 只为精确 count_eq 候选构造实体擦除；"
                "目标 rule_id 仅作 provenance，确认器不读取规则真值。"
                if counterfactual_policy == "rule_matched_count_erase" else
                "generic 反事实没有按候选规则覆盖率配额生成。"
            ),
            "left_of(left_compartment, right_compartment) 与 "
            "disjoint(left_compartment, right_compartment) 由几何定义恒真，"
            "任何视觉干预都无法证伪它们；见 falsifiable 字段。",
            "事后测试集指标只用于报告，不参与搜索，也不用于调阈值。",
            "protocol_admission 会把 support 不一致或使用额外正常图的实验"
            "限制为诊断层，禁止写入严格 K-shot 主模型表。",
            "strict_k4 模式的 L_normal 是 support consistency，而不是独立"
            "验证误差；是否泛化只能由冻结后的 test 和后续多 seed 判断。",
            (
                "实体身份 gate 只做精确词表/支持观测拒绝，不做 alias、模糊"
                "匹配或自动语义纠错。"
                if entity_identity_gate else
                "实体身份 gate 未启用；该运行不能评估 Qwen 实体名称过滤效果。"
            ),
            "进化搜索若没有稳定超过同损失、无额外奖励的贪心对照，"
            "就不会被接入主模型。",
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
    parser.add_argument("--calibration-image-count", type=int, default=32)
    parser.add_argument("--normal-fit-count", type=int, default=31)
    parser.add_argument("--counterfactual-source-count", type=int, default=10)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--include-test-split", action="store_true")
    parser.add_argument("--baseline-training-manifest", type=Path, default=None)
    parser.add_argument(
        "--protocol-mode",
        choices=("validation_assisted", "strict_k4"),
        default="validation_assisted",
    )
    parser.add_argument(
        "--entity-identity-gate",
        action="store_true",
        default=False,
        help="Reject rules whose exact entity names are absent from the fixed "
             "perception ontology or insufficiently observed on K-shot support.",
    )
    parser.add_argument(
        "--entity-identity-min-support", type=float, default=0.75,
    )
    parser.add_argument(
        "--counterfactual-policy",
        choices=("generic", "rule_matched_count_erase"),
        default="generic",
    )
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
        calibration_image_count=args.calibration_image_count,
        normal_fit_count=args.normal_fit_count,
        counterfactual_source_count=args.counterfactual_source_count,
        max_validation=args.max_validation,
        include_test_split=args.include_test_split,
        baseline_training_manifest=args.baseline_training_manifest,
        protocol_mode=args.protocol_mode,
        entity_identity_gate=args.entity_identity_gate,
        entity_identity_min_support=args.entity_identity_min_support,
        counterfactual_policy=args.counterfactual_policy,
    )
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
