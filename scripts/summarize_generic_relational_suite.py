#!/usr/bin/env python3
"""Create a compact, auditable summary of frozen multi-dataset v4 runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _load(path: str | Path):
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-summary", nargs="+", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--specialized-cable")
    parser.add_argument("--static-smoke")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    validation = _load(args.validation)
    datasets, all_classes = [], []
    rejection_reasons: Counter[str] = Counter()
    total_candidates = 0
    for summary_name in args.suite_summary:
        summary = _load(summary_name)
        rows = summary["classes"]
        all_classes.extend(rows)
        proposed = sum(not row["proposal_abstained"] for row in rows)
        grounded = sum(
            not row["proposal_abstained"] and row["grounding_support_match_rate"] >= 0.75
            for row in rows
        )
        candidates = 0
        for row in rows:
            run_dir = Path(row["rule_library"]).resolve().parent
            audit = _load(run_dir / "candidate_audit_v4_generic.json")
            candidates += len(audit)
            for item in audit:
                if not item.get("accepted"):
                    rejection_reasons[str(item.get("reason", "unknown"))] += 1
        total_candidates += candidates
        datasets.append({
            "dataset": summary["dataset"],
            "classes": summary["class_count"],
            "llm_proposal_classes": proposed,
            "stable_grounder_classes": grounded,
            "candidate_rules": candidates,
            "accepted_rules": int(sum(row["selected_rule_count"] for row in rows)),
            "rule_covered_classes": summary["rule_covered_classes"],
            "fusion_enabled_classes": summary["fusion_enabled_classes"],
            "improved_classes": summary["improved_classes"],
            "degraded_classes": summary["degraded_classes"],
            "macro_i_auroc": {
                "baseline": summary["macro_baseline_i_auroc"],
                "fused": summary["macro_fused_i_auroc"],
                "delta": summary["macro_delta_i_auroc"],
            },
            "macro_i_aupr": {
                "baseline": summary["macro_baseline_i_aupr"],
                "fused": summary["macro_fused_i_aupr"],
                "delta": summary["macro_delta_i_aupr"],
            },
        })

    baseline = np.asarray([row["baseline"]["i_auroc"] for row in all_classes])
    fused = np.asarray([row["baseline_plus_rules"]["i_auroc"] for row in all_classes])
    nonzero = [
        {
            "dataset": row["dataset"],
            "class_name": row["class_name"],
            "selected_predicates": row["selected_predicates"],
            "baseline_i_auroc": row["baseline"]["i_auroc"],
            "fused_i_auroc": row["baseline_plus_rules"]["i_auroc"],
            "delta_i_auroc": row["delta_i_auroc"],
            "normal_trigger_rate": row["normal_trigger_rate"],
            "anomaly_trigger_rate": row["anomaly_trigger_rate"],
        }
        for row in all_classes if row["delta_i_auroc"] != 0
    ]
    output = {
        "schema_version": "4.0-generic-relational-multidataset-summary",
        "protocol": {
            "shot": 4,
            "support_seed": 1,
            "support_strategy": "inp_official",
            "llm": "/home/hjh/AgentModels/Qwen3-VL-8B-Thinking",
            "llm_queries": "four independent known-normal support images",
            "consensus": "at least 3/4 agree on within-image entity count",
            "rule_validation": "strict leave-one-support-out grounder refit; zero normal FPR",
            "test_time": "compiled static rules; no LLM",
            "test_labels_used_for_training_or_selection": False,
        },
        "independent_validation": {
            "all_checks_passed": validation["all_checks_passed"],
            "passed_checks": validation["passed_checks"],
            "total_checks": validation["total_checks"],
            "bootstrap_repeats": 2000,
        },
        "datasets": datasets,
        "overall_generic": {
            "classes": len(all_classes),
            "llm_proposal_classes": int(sum(not row["proposal_abstained"] for row in all_classes)),
            "stable_grounder_classes": int(sum(
                not row["proposal_abstained"] and row["grounding_support_match_rate"] >= 0.75
                for row in all_classes
            )),
            "candidate_rules": total_candidates,
            "accepted_rules": int(sum(row["selected_rule_count"] for row in all_classes)),
            "rule_covered_classes": int(sum(row["selected_rule_count"] > 0 for row in all_classes)),
            "fusion_enabled_classes": int(sum(row["fusion"]["enabled"] for row in all_classes)),
            "improved_classes": int(np.sum(fused > baseline)),
            "degraded_classes": int(np.sum(fused < baseline)),
            "macro_i_auroc": {
                "baseline": float(np.mean(baseline)),
                "fused": float(np.mean(fused)),
                "delta": float(np.mean(fused - baseline)),
            },
            "rejected_candidate_reasons": dict(sorted(rejection_reasons.items())),
        },
        "nonzero_generic_results": nonzero,
        "conclusion": (
            "The frozen generic mechanism is safe but does not establish broad positive "
            "generalization: coverage and effect size are too small. The principal bottleneck "
            "is semantic grounding, not rule retrieval."
        ),
    }
    if args.specialized_cable:
        cable = _load(args.specialized_cable)
        output["specialized_cable_exploratory"] = {
            "status": cable.get("evaluation_status"),
            "baseline_i_auroc": cable["baseline"]["i_auroc"],
            "fused_i_auroc": cable["baseline_plus_relations"]["i_auroc"],
            "delta_i_auroc": cable["delta_i_auroc"],
            "baseline_i_aupr": cable["baseline"]["i_aupr"],
            "fused_i_aupr": cable["baseline_plus_relations"]["i_aupr"],
            "delta_i_aupr": cable["delta_i_aupr"],
            "cable_swap": cable["per_defect_type"]["cable_swap"],
            "not_part_of_confirmatory_generic_macro": True,
        }
    if args.static_smoke:
        smoke = _load(args.static_smoke)
        output["static_inference_smoke"] = {
            "runtime_loaded_llm": smoke["runtime_loaded_llm"],
            "checkpoint_sha256_verified": smoke["checkpoint_sha256_verified"],
            "adapter_config_verified": smoke["adapter_config_verified"],
            "image_count": len(smoke["predictions"]),
        }
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
