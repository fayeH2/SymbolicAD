#!/usr/bin/env python3
"""Load Qwen once and induce generic relational proposals for many classes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from symbolicAD.data import (
    MPDD_CLASS_ORDER, MVTEC_CLASS_ORDER, VISA_CLASS_ORDER, select_inpformer_support,
)
from symbolicAD.qwen_generic_relations import (
    QwenGenericRelationalProposer,
    consensus_generic_relational_proposals,
)
from symbolicAD.qwen_relations import make_rgb_support_montage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mvtec", "visa", "mpdd"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--classes", nargs="*")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--support-seed", type=int, default=1)
    parser.add_argument("--llm-model", default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking")
    parser.add_argument("--llm-device", default="cuda:1")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--per-image-consensus", action="store_true")
    args = parser.parse_args()

    default_classes = {
        "mvtec": MVTEC_CLASS_ORDER,
        "visa": VISA_CLASS_ORDER,
        "mpdd": MPDD_CLASS_ORDER,
    }
    classes = args.classes or default_classes[args.dataset]
    root = Path(args.output_root).expanduser().resolve() / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    proposer = QwenGenericRelationalProposer(
        args.llm_model, args.llm_device, args.max_new_tokens
    )
    summaries = []
    for class_name in classes:
        class_output = root / class_name / f"k{args.shot}"
        class_output.mkdir(parents=True, exist_ok=True)
        support = select_inpformer_support(
            args.data_root, args.dataset, class_name, args.shot, args.support_seed
        )
        montage = make_rgb_support_montage(
            support, class_output / "normal_support_relational_montage.jpg"
        )
        if args.per_image_consensus:
            individual = []
            raw_outputs = []
            for path in support:
                item = proposer.propose(class_name, 1, path)
                raw_outputs.append(item.pop("raw_output", ""))
                individual.append(item)
            proposal = consensus_generic_relational_proposals(individual)
            proposal["model"] = proposer.model_path
            raw = "\n\n===== SUPPORT QUERY =====\n\n".join(raw_outputs)
        else:
            proposal = proposer.propose(class_name, len(support), montage)
            raw = proposal.pop("raw_output", "")
        (class_output / "qwen_generic_relational_raw.txt").write_text(
            raw + "\n", encoding="utf-8"
        )
        (class_output / "qwen_generic_relational_proposal.json").write_text(
            json.dumps(proposal, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries.append({
            "class_name": class_name,
            "proposal": str(class_output / "qwen_generic_relational_proposal.json"),
            "abstained": proposal.get("primary_entity") is None,
            "primary_entity": proposal.get("primary_entity"),
            "predicates": [item["predicate"] for item in proposal.get("rules", [])],
        })
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    (root / f"k{args.shot}_proposal_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
