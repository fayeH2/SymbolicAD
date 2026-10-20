#!/usr/bin/env python3
"""Run the frozen v4 generic protocol over a complete benchmark."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from neurosymbolic_fsad.data import MPDD_CLASS_ORDER, MVTEC_CLASS_ORDER, VISA_CLASS_ORDER
from neurosymbolic_fsad.experiment_generic_relational import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mvtec", "visa", "mpdd"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--proposal-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--inpformer-root", required=True)
    parser.add_argument("--classes", nargs="*")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    default_classes = {
        "mvtec": MVTEC_CLASS_ORDER,
        "visa": VISA_CLASS_ORDER,
        "mpdd": MPDD_CLASS_ORDER,
    }
    classes = args.classes or default_classes[args.dataset]
    results = []
    for class_name in classes:
        proposal = (
            Path(args.proposal_root) / args.dataset / class_name / "k4" /
            "qwen_generic_relational_proposal.json"
        )
        if not proposal.is_file():
            raise FileNotFoundError(proposal)
        output = Path(args.output_root) / args.dataset / class_name / "k4"
        cache = Path(args.cache_root) / args.dataset / class_name / "k4"
        payload = run(Namespace(
            dataset=args.dataset,
            data_root=args.data_root,
            class_name=class_name,
            shot=4,
            support_seed=1,
            support_strategy="inp_official",
            checkpoint=args.checkpoint,
            inpformer_root=args.inpformer_root,
            proposal_json=str(proposal),
            output_dir=str(output),
            prediction_cache_dir=str(cache),
            device=args.device,
            batch_size=args.batch_size,
            encoder="dinov2reg_vit_base_14",
            inp_num=6,
            input_size=448,
            crop_size=392,
            map_size=256,
            relation_image_size=256,
            minimum_rule_auc=0.75,
            maximum_rule_fpr=0.0,
            minimum_fusion_gain=0.02,
            llm_model="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking",
        ))
        results.append(payload)
        print(json.dumps({
            "class_name": class_name,
            "rules": payload["selected_rule_count"],
            "baseline_auroc": payload["baseline"]["i_auroc"],
            "fused_auroc": payload["baseline_plus_rules"]["i_auroc"],
            "delta_auroc": payload["delta_i_auroc"],
        }, ensure_ascii=False), flush=True)

    summary = {
        "schema_version": "4.0-generic-relational-suite",
        "dataset": args.dataset,
        "class_count": len(results),
        "rule_covered_classes": sum(item["selected_rule_count"] > 0 for item in results),
        "fusion_enabled_classes": sum(item["fusion"]["enabled"] for item in results),
        "improved_classes": sum(item["delta_i_auroc"] > 1e-12 for item in results),
        "degraded_classes": sum(item["delta_i_auroc"] < -1e-12 for item in results),
        "macro_baseline_i_auroc": float(np.mean([item["baseline"]["i_auroc"] for item in results])),
        "macro_fused_i_auroc": float(np.mean([item["baseline_plus_rules"]["i_auroc"] for item in results])),
        "macro_delta_i_auroc": float(np.mean([item["delta_i_auroc"] for item in results])),
        "macro_baseline_i_aupr": float(np.mean([item["baseline"]["i_aupr"] for item in results])),
        "macro_fused_i_aupr": float(np.mean([item["baseline_plus_rules"]["i_aupr"] for item in results])),
        "macro_delta_i_aupr": float(np.mean([item["delta_i_aupr"] for item in results])),
        "classes": results,
        "test_labels_used_for_training_or_selection": False,
    }
    output = Path(args.output_root) / args.dataset / "k4_suite_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
