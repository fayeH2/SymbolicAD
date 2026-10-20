#!/usr/bin/env python3
"""Summarize support-size controls for the v7 executable logic experiment.

This script never selects a run or changes a score.  It reads every supplied
run, preserves negative controls, and reports how Qwen candidates become
groundable and finally accepted executable atoms as normal support grows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


METRICS = ("i_auroc", "i_aupr", "p_auroc", "p_aupr")


def _read(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def _load_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    metrics_path = run_dir / "metrics_v7_executable_logic.json"
    library_path = run_dir / "normality_rules_v7_executable_logic.json"
    metrics = _read(metrics_path)
    library = _read(library_path)
    if metrics.get("rule_library") != str(library_path):
        raise ValueError(f"Metric/library lineage mismatch in {run_dir}")
    baseline = metrics["baseline"]
    fused = metrics["neuro_symbolic"]
    delta = metrics["delta_neuro_symbolic_minus_baseline"]
    rules = []
    for row in library["candidate_audit"]:
        rules.append({
            "id": row["id"],
            "status": row["status"],
            "accepted": str(row["status"]).startswith("accepted_"),
            "llm_vote_rate": float(row["llm_vote_rate"]),
            "required_llm_vote_rate": float(row["required_llm_vote_rate"]),
            "upper": float(row["upper"]),
            "scale": float(row["scale"]),
            "crossfit_normal_fpr": float(row["crossfit_normal_fpr"]),
            "bootstrap_relative_interval_width": float(
                row["bootstrap"]["relative_interval_width"]
            ),
            "rejection_reasons": list(row["rejection_reasons"]),
        })
    return {
        "run_dir": str(run_dir),
        "rule_support_count": int(metrics["rule_support_count"]),
        "candidate_rule_count": int(metrics["candidate_rule_count"]),
        "groundable_rule_count": int(metrics["groundable_rule_count"]),
        "accepted_rule_count": int(metrics["accepted_rule_count"]),
        "accepted_rule_ids": list(metrics["accepted_rule_ids"]),
        "baseline": {key: float(baseline[key]) for key in METRICS},
        "neuro_symbolic": {key: float(fused[key]) for key in METRICS},
        "delta": {key: float(delta[key]) for key in METRICS},
        "normal_symbolic_trigger_rate": float(
            metrics["normal_symbolic_trigger_rate"]
        ),
        "anomaly_symbolic_trigger_rate": float(
            metrics["anomaly_symbolic_trigger_rate"]
        ),
        "rules": rules,
        "candidate_bank": metrics.get("candidate_bank"),
        "metrics_path": str(metrics_path),
        "library_path": str(library_path),
    }


def _top_candidates(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    bank = _read(Path(path))
    result = []
    for candidate in bank.get("candidates", []):
        rule = candidate["rule"]
        result.append({
            "predicate": rule["predicate"],
            "arguments": rule["arguments"],
            "expected": rule["expected"],
            "occurrence_count": int(candidate["occurrence_count"]),
            "occurrence_rate": float(
                candidate["occurrence_rate_among_expected_supports"]
            ),
        })
    return result


def summarize(run_dirs: list[Path]) -> dict[str, Any]:
    runs = sorted((_load_run(path) for path in run_dirs), key=lambda row: row["rule_support_count"])
    if not runs:
        raise ValueError("At least one --run-dir is required")
    reference = runs[0]["baseline"]
    for run in runs[1:]:
        if any(abs(run["baseline"][key] - reference[key]) > 1e-12 for key in METRICS):
            raise ValueError("Runs do not use the same frozen baseline predictions")
    all_rules = sorted({rule["id"] for run in runs for rule in run["rules"]})
    rule_evolution = []
    for rule_id in all_rules:
        row: dict[str, Any] = {"rule_id": rule_id, "by_support_count": {}}
        for run in runs:
            matched = next(rule for rule in run["rules"] if rule["id"] == rule_id)
            row["by_support_count"][str(run["rule_support_count"])] = matched
        rule_evolution.append(row)
    for run in runs:
        run["top_qwen_candidates"] = _top_candidates(run["candidate_bank"])
    return {
        "schema_version": "7.0-support-stability-summary",
        "comparison_policy": "all supplied runs retained, including negative controls",
        "same_frozen_baseline_verified": True,
        "metrics": list(METRICS),
        "runs": runs,
        "rule_evolution": rule_evolution,
        "interpretation": {
            "candidate_rules": "union of independently proposed Qwen semantics",
            "groundable_rules": "candidates covered by deterministic visual predicates",
            "accepted_rules": "groundable rules passing normal-only vote, cross-fit, dynamic-range, and redundancy checks",
            "support_below_10": "provisional only; not a stable rule library",
        },
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Executable logic support-stability summary",
        "",
        "All runs below use the same frozen 4-shot INP-Former predictions. Rule support count changes only the normal images used to propose, calibrate, and validate the static rule library.",
        "",
        "| rule supports | candidates | groundable | accepted | accepted IDs | normal trigger | anomaly trigger |",
        "|---:|---:|---:|---:|---|---:|---:|",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| {run['rule_support_count']} | {run['candidate_rule_count']} | "
            f"{run['groundable_rule_count']} | {run['accepted_rule_count']} | "
            f"{', '.join(run['accepted_rule_ids'])} | "
            f"{run['normal_symbolic_trigger_rate']:.4f} | "
            f"{run['anomaly_symbolic_trigger_rate']:.4f} |"
        )
    lines.extend([
        "",
        "| rule supports | method | I-AUROC | I-AUPR | P-AUROC | P-AUPR |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for run in summary["runs"]:
        lines.append(
            f"| {run['rule_support_count']} | baseline | "
            + " | ".join(_fmt(run["baseline"][key]) for key in METRICS)
            + " |"
        )
        lines.append(
            f"| {run['rule_support_count']} | executable logic | "
            + " | ".join(_fmt(run["neuro_symbolic"][key]) for key in METRICS)
            + " |"
        )
        lines.append(
            f"| {run['rule_support_count']} | delta | "
            + " | ".join(f"{run['delta'][key]:+.6f}" for key in METRICS)
            + " |"
        )
    lines.extend([
        "",
        "## Rule audit",
        "",
        "| supports | rule | status | Qwen vote | threshold | normal CV FPR | rejection |",
        "|---:|---|---|---:|---:|---:|---|",
    ])
    for run in summary["runs"]:
        for rule in run["rules"]:
            lines.append(
                f"| {run['rule_support_count']} | {rule['id']} | {rule['status']} | "
                f"{rule['llm_vote_rate']:.4f} | {rule['upper']:.6f} | "
                f"{rule['crossfit_normal_fpr']:.4f} | "
                f"{', '.join(rule['rejection_reasons']) or '-'} |"
            )
    lines.extend([
        "",
        "A 4-support library is explicitly marked provisional. The 30-support library is the reported stable experiment; unstable package-pose and dead symmetry candidates remain rejected instead of being forced into inference.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.run_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json.resolve()),
        "output_markdown": str(args.output_markdown.resolve()),
        "support_counts": [row["rule_support_count"] for row in summary["runs"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
