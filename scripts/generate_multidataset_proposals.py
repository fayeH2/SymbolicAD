#!/usr/bin/env python3
"""Generate one static Qwen rule proposal per category with one model load.

This is deliberately dataset-agnostic.  The multi-GPU orchestrator starts one
process per dataset/shard and isolates it to a single visible GPU.  Qwen is
never loaded by the later test-time rule engine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from neurosymbolic_fsad.data import select_inpformer_support
from neurosymbolic_fsad.qwen_generic_relations import QwenGenericRelationalProposer
from neurosymbolic_fsad.qwen_relations import make_rgb_support_montage


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _consensus(proposer, class_name: str, support: list[Path]):
    protocol = (
        "four_independent_support_queries_exact_count"
        if len(support) == 4 else
        "independent_support_queries_exact_count"
    )
    individual, raw_outputs = [], []
    for path in support:
        item = proposer.propose(class_name, 1, path)
        raw_outputs.append(item.pop("raw_output", ""))
        individual.append(item)
    entities = [item.get("primary_entity") for item in individual]
    counts = [item.get("expected_count") for item in entities if isinstance(item, dict)]
    entity_ids = [item.get("id") for item in entities if isinstance(item, dict)]
    arrangements = [item.get("arrangement") for item in entities if isinstance(item, dict)]
    valid = (
        len(counts) == len(support)
        and len(set(counts)) == 1
        and len(set(entity_ids)) == 1
    )
    if valid:
        predicate_sets = [
            {row["predicate"] for row in item.get("rules", [])}
            for item in individual
        ]
        shared = set.intersection(*predicate_sets) if predicate_sets else set()
        rules = [
            row for row in individual[0].get("rules", [])
            if row["predicate"] in shared
        ]
        if not any(row["predicate"] == "count_eq" for row in rules):
            rules.insert(0, {
                "id": "consensus_count",
                "predicate": "count_eq",
                "importance": 0.9,
                "confidence": 0.9,
                "reason": "same within-image count in every independent support query",
            })
        primary = dict(entities[0])
        primary["arrangement"] = (
            arrangements[0] if len(set(arrangements)) == 1 else "free"
        )
        proposal = {
            "normality_summary": individual[0].get("normality_summary", ""),
            "primary_entity": primary,
            "rules": rules,
            "model": proposer.model_path,
            "consensus_protocol": protocol,
            "per_support_proposals": individual,
        }
    else:
        proposal = {
            "normality_summary": "abstained: independent support proposals disagree",
            "primary_entity": None,
            "rules": [],
            "model": proposer.model_path,
            "consensus_protocol": protocol,
            "per_support_proposals": individual,
        }
    return proposal, "\n\n===== SUPPORT QUERY =====\n\n".join(raw_outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--support-seed", type=int, default=1)
    parser.add_argument("--llm-model", default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking")
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--summary-name", default="proposal_summary.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root).expanduser().resolve() / args.dataset
    pending = []
    summaries = []
    for class_name in args.classes:
        class_output = root / class_name / f"k{args.shot}"
        proposal_path = class_output / "qwen_generic_relational_proposal.json"
        if args.resume and proposal_path.is_file():
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            row = {
                "class_name": class_name,
                "proposal": str(proposal_path),
                "abstained": proposal.get("primary_entity") is None,
                "primary_entity": proposal.get("primary_entity"),
                "predicates": [item["predicate"] for item in proposal.get("rules", [])],
                "resumed": True,
            }
            summaries.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        else:
            pending.append(class_name)

    proposer = None
    failures = []
    if pending:
        proposer = QwenGenericRelationalProposer(
            args.llm_model, args.llm_device, args.max_new_tokens
        )
    for class_name in pending:
        class_output = root / class_name / f"k{args.shot}"
        class_output.mkdir(parents=True, exist_ok=True)
        try:
            support = select_inpformer_support(
                args.data_root, args.dataset, class_name, args.shot, args.support_seed
            )
            make_rgb_support_montage(
                support, class_output / "normal_support_relational_montage.jpg"
            )
            proposal, raw = _consensus(proposer, class_name, support)
            raw_path = class_output / "qwen_generic_relational_raw.txt"
            proposal_path = class_output / "qwen_generic_relational_proposal.json"
            raw_path.write_text(raw + "\n", encoding="utf-8")
            _dump(proposal_path, proposal)
            row = {
                "class_name": class_name,
                "proposal": str(proposal_path),
                "abstained": proposal.get("primary_entity") is None,
                "primary_entity": proposal.get("primary_entity"),
                "predicates": [item["predicate"] for item in proposal.get("rules", [])],
                "resumed": False,
            }
            summaries.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        except Exception as exc:  # keep other categories running; nonzero at end
            row = {"class_name": class_name, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(row)
            summaries.append(row)
            print(json.dumps(row, ensure_ascii=False), file=sys.stderr, flush=True)

    _dump(root / f"k{args.shot}_{args.summary_name}", {
        "dataset": args.dataset,
        "shot": args.shot,
        "classes": summaries,
        "failures": failures,
    })
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
