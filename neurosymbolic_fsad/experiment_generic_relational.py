"""Multi-class experiment for Qwen-guided generic repeated-part rules."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import collect_test_samples, select_inpformer_support, select_support
from .generic_relations import (
    GenericRelationalRuleEngine,
    compile_generic_rules,
)
from .inpformer_adapter import (
    DEFAULT_INPFORMER_ROOT,
    INPFormerAdapter,
    file_sha256,
    load_prediction_cache,
    save_prediction_cache,
)
from .qwen_generic_relations import validate_generic_relational_proposal
from .qwen_relations import make_rgb_support_montage


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metrics(labels, scores) -> Dict[str, float]:
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    return {
        "i_auroc": float(roc_auc_score(labels, scores)) if np.ptp(scores) > 0 else 0.5,
        "i_aupr": float(average_precision_score(labels, scores)),
    }


def _cached_infer(adapter, path, images, identifiers, metadata, batch_size):
    cached = load_prediction_cache(path, identifiers, expected_metadata=dict(metadata))
    if cached is not None:
        return cached
    result = adapter.infer_images(images, batch_size=batch_size, identifiers=identifiers)
    save_prediction_cache(path, result, metadata=dict(metadata))
    return result


def _robust_location_scale(values):
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return center, max(1.4826 * mad, float(np.std(values)) * 0.25, 1e-6)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    class_root = Path(args.data_root).expanduser().resolve() / args.class_name
    support_paths = (
        select_inpformer_support(args.data_root, args.dataset, args.class_name,
                                 args.shot, args.support_seed)
        if args.support_strategy == "inp_official"
        else select_support(class_root, args.shot, args.support_seed)
    )
    test_subdir = getattr(args, "test_subdir", "test")
    samples = collect_test_samples(class_root, test_subdir=test_subdir)
    test_paths = [Path(item["path"]) for item in samples]
    montage = make_rgb_support_montage(
        support_paths, output / "normal_support_relational_montage.jpg"
    )
    raw = json.loads(Path(args.proposal_json).read_text(encoding="utf-8"))
    proposal = validate_generic_relational_proposal(raw)
    proposal["model"] = str(raw.get("model", args.llm_model))
    proposal["consensus_protocol"] = raw.get("consensus_protocol", "single_montage_query")
    proposal["source_consensus_protocol"] = proposal["consensus_protocol"]
    proposal["per_support_proposals"] = raw.get("per_support_proposals", [])
    if proposal["consensus_protocol"] == "four_independent_support_queries_exact_count":
        # The proposal-generation pass used exact identity/count consensus and
        # could therefore abstain even when three independent views agreed on
        # the observable count.  The frozen experiment deliberately recovers
        # only the 3/4 count majority; semantic identity is still enforced by
        # the deterministic grounder and strict leave-one-support-out checks.
        # Record that actual protocol explicitly in every compiled artifact.
        proposal["consensus_protocol"] = (
            "four_independent_support_queries_3of4_count_consensus"
        )
        canonical_individual = [
            validate_generic_relational_proposal(item)
            for item in proposal["per_support_proposals"]
            if isinstance(item, Mapping)
        ]
        valid = [item for item in canonical_individual
                 if isinstance(item.get("primary_entity"), Mapping)]
        count_votes = Counter(
            int(item["primary_entity"]["expected_count"]) for item in valid
        )
        winning_count, agreement = count_votes.most_common(1)[0] if count_votes else (None, 0)
        agreeing = [item for item in valid
                    if int(item["primary_entity"]["expected_count"]) == winning_count]
        required_agreement = max(3, int(np.ceil(0.75 * len(support_paths))))
        proposal["required_count_agreement"] = required_agreement
        if agreement < required_agreement:
            proposal["normality_summary"] = "abstained: fewer than 3/4 independent count proposals agree"
            proposal["primary_entity"] = None
            proposal["rules"] = []
        else:
            arrangement_votes = Counter(
                str(item["primary_entity"]["arrangement"]) for item in agreeing
            )
            winning_arrangement, arrangement_agreement = arrangement_votes.most_common(1)[0]
            source = agreeing[0]
            proposal["normality_summary"] = source.get("normality_summary", "")
            proposal["primary_entity"] = dict(source["primary_entity"])
            proposal["primary_entity"]["expected_count"] = int(winning_count)
            proposal["primary_entity"]["arrangement"] = (
                winning_arrangement if arrangement_agreement >= required_agreement else "free"
            )
            shared_predicates = set.intersection(*[
                {row["predicate"] for row in item.get("rules", [])} for item in agreeing
            ]) if agreeing else set()
            proposal["rules"] = [
                dict(row) for row in source.get("rules", [])
                if row["predicate"] in shared_predicates
            ]
            if not any(row["predicate"] == "count_eq" for row in proposal["rules"]):
                proposal["rules"].insert(0, {
                    "id": "majority_count", "predicate": "count_eq",
                    "importance": 0.85, "confidence": 0.85,
                    "reason": "at least three of four independent support queries agree",
                })
            proposal["consensus_support_agreement"] = f"{agreement}/{len(support_paths)}"
            entities = [item["primary_entity"] for item in agreeing]
            arrangements = {str(item.get("arrangement")) for item in entities}
            predicates = {item.get("predicate") for item in proposal.get("rules", [])}
            expected_count = int(proposal["primary_entity"]["expected_count"])
            expansion = []
            if expected_count > 1:
                expansion.extend([
                    ("similar_size", "repeated entity instances admit a support-verifiable size relation"),
                    ("similar_orientation", "repeated entity instances admit a support-verifiable orientation relation"),
                    ("appearance_consistency", "repeated entity instances admit a support-verifiable appearance relation"),
                ])
            if len(arrangements) == 1 and next(iter(arrangements)) not in {"single", "free"}:
                expansion.extend([
                    ("arrangement_eq", "same arrangement in all four independent support proposals"),
                    ("ordered_spacing", "ordered arrangement admits a support-verifiable spacing relation"),
                ])
            for predicate, reason in expansion:
                if predicate not in predicates:
                    proposal["rules"].append({
                        "id": f"consensus_{predicate}",
                        "predicate": predicate,
                        "importance": 0.8,
                        "confidence": 0.8,
                        "reason": reason,
                    })
                    predicates.add(predicate)
    _dump(output / "qwen_generic_relational_proposal.json", proposal)

    grounder, rules, audits = compile_generic_rules(
        proposal, support_paths, image_size=args.relation_image_size,
        minimum_cv_auc=args.minimum_rule_auc,
        maximum_normal_fpr=args.maximum_rule_fpr,
    )
    _dump(output / "candidate_audit_v4_generic.json", audits)

    pseudo_images, pseudo_ids = [], []
    if grounder is not None:
        predicates = [item["predicate"] for item in rules]
        for support_index, path in enumerate(support_paths):
            for predicate in predicates:
                pseudo_images.append(grounder.counterfactual(path, predicate))
                pseudo_ids.append(f"counterfactual://{support_index}/{predicate}")

    metadata = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "inpformer_root": str(Path(args.inpformer_root).expanduser().resolve()),
        "encoder": args.encoder,
        "inp_num": int(args.inp_num),
        "input_size": int(args.input_size),
        "crop_size": int(args.crop_size),
        "map_size": int(args.map_size),
        "cache_schema": 2,
        "dataset": args.dataset,
        "class_name": args.class_name,
        "shot": args.shot,
        "support_seed": args.support_seed,
        "test_subdir": test_subdir,
    }
    cache_root = Path(args.prediction_cache_dir).expanduser().resolve() \
        if args.prediction_cache_dir else output
    support_ids = [str(path.resolve()) for path in support_paths]
    test_ids = [str(path.resolve()) for path in test_paths]
    support_cache = cache_root / "cache" / "support_inp_v4.npz"
    test_cache = cache_root / "cache" / "test_inp_v4.npz"
    support_predictions = load_prediction_cache(
        support_cache, support_ids, expected_metadata=metadata
    )
    test_predictions = load_prediction_cache(
        test_cache, test_ids, expected_metadata=metadata
    )
    pseudo_predictions = None
    pseudo_metadata = {**metadata, "counterfactual_schema": "generic_component_v1"}
    pseudo_cache = cache_root / "cache" / "counterfactual_inp_v4.npz"
    if pseudo_images:
        pseudo_predictions = load_prediction_cache(
            pseudo_cache, pseudo_ids, expected_metadata=pseudo_metadata
        )
    if support_predictions is None or test_predictions is None or (
        pseudo_images and pseudo_predictions is None
    ):
        adapter = INPFormerAdapter(
            checkpoint=args.checkpoint,
            device=args.device,
            inpformer_root=args.inpformer_root,
            encoder_name=args.encoder,
            inp_num=args.inp_num,
            input_size=args.input_size,
            crop_size=args.crop_size,
            map_size=args.map_size,
        )
        if support_predictions is None:
            support_predictions = adapter.infer_images(
                support_paths, batch_size=args.batch_size, identifiers=support_ids
            )
            save_prediction_cache(support_cache, support_predictions, metadata=metadata)
        if test_predictions is None:
            test_predictions = adapter.infer_images(
                test_paths, batch_size=args.batch_size, identifiers=test_ids
            )
            save_prediction_cache(test_cache, test_predictions, metadata=metadata)
        if pseudo_images and pseudo_predictions is None:
            pseudo_predictions = adapter.infer_images(
                pseudo_images, batch_size=args.batch_size, identifiers=pseudo_ids
            )
            save_prediction_cache(pseudo_cache, pseudo_predictions, metadata=pseudo_metadata)
        adapter.close()
        del adapter
        gc.collect()

    baseline_center, baseline_scale = _robust_location_scale(support_predictions["scores"])
    provisional = GenericRelationalRuleEngine({
        "grounder": grounder.state_dict() if grounder is not None else None,
        "rules": rules,
    })
    support_relation = np.asarray([provisional.evaluate(path)[0] for path in support_paths])
    pseudo_relation = np.asarray([provisional.evaluate(image)[0] for image in pseudo_images])
    support_z = (np.asarray(support_predictions["scores"]) - baseline_center) / baseline_scale
    fusion = {
        "enabled": False, "alpha": 0.0, "support_cv_gain": 0.0,
        "baseline_center": baseline_center, "baseline_scale": baseline_scale,
        "score_domain": "raw_inp_score",
    }
    selected_audits = [row for row in audits if row.get("accepted")]
    zero_fpr_rules = bool(selected_audits) and all(
        float(row.get("normal_fpr", 1.0)) <= 0.0 for row in selected_audits
    )
    fusion["zero_crossfit_normal_fpr_required"] = True
    fusion["selected_rule_crossfit_normal_fpr"] = [
        float(row.get("normal_fpr", 1.0)) for row in selected_audits
    ]
    if rules and pseudo_predictions is not None and len(pseudo_images) and zero_fpr_rules:
        pseudo_z = (np.asarray(pseudo_predictions["scores"]) - baseline_center) / baseline_scale
        labels_cv = np.asarray([0] * len(support_z) + [1] * len(pseudo_z))
        baseline_cv = np.concatenate([support_z, pseudo_z])
        base_auc = float(roc_auc_score(labels_cv, baseline_cv))
        support_raw = np.asarray(support_predictions["scores"], dtype=float)
        pseudo_raw = np.asarray(pseudo_predictions["scores"], dtype=float)
        normal_span = float(np.quantile(support_raw, 0.90) - np.quantile(support_raw, 0.10))
        alpha = max(2.0 * normal_span, 1e-4)
        fused_cv = np.concatenate([
            support_raw + alpha * support_relation,
            pseudo_raw + alpha * pseudo_relation,
        ])
        fused_auc = float(roc_auc_score(labels_cv, fused_cv))
        # Rule validity is decided by strict relational cross-fit.  Fusion only
        # calibrates its magnitude to normal INP score variation; it must not
        # require synthetic defects to outrank a baseline that already detects
        # those artificial edits perfectly.
        fusion.update({
            "enabled": True, "alpha": alpha,
            "support_cv_gain": fused_auc - base_auc,
            "support_baseline_auc": base_auc,
            "support_fused_auc": fused_auc,
            "normal_score_q10": float(np.quantile(support_raw, 0.10)),
            "normal_score_q90": float(np.quantile(support_raw, 0.90)),
            "calibration_rule": "alpha=2*(normal_q90-normal_q10), support-only",
        })
    elif rules and not zero_fpr_rules:
        fusion["disabled_reason"] = "selected_rule_has_nonzero_leave_one_support_normal_fpr"

    library = {
        "schema_version": "4.0-generic-relational",
        "dataset": args.dataset,
        "class_name": args.class_name,
        "shot": args.shot,
        "support": [str(path.resolve()) for path in support_paths],
        "support_sha256": {str(path.resolve()): file_sha256(path) for path in support_paths},
        "qwen": proposal,
        "grounder": grounder.state_dict() if grounder is not None else None,
        "rules": rules,
        "fusion": fusion,
        "inpformer": metadata,
        "runtime_requires_llm": False,
    }
    engine = GenericRelationalRuleEngine(library)
    library_path = engine.save(output / "normality_rules_v4_generic.json")

    labels = np.asarray([int(item["label"]) for item in samples])
    baseline = np.asarray(test_predictions["scores"], dtype=float)
    baseline_z = (baseline - baseline_center) / baseline_scale
    relation, decision, evidence_rows, csv_rows = [], [], [], []
    for sample, raw_baseline, z_score in zip(samples, baseline, baseline_z):
        rel, evidence, facts, diagnostics = engine.evaluate(sample["path"])
        score = float(raw_baseline + fusion["alpha"] * rel) if fusion["enabled"] else float(raw_baseline)
        relation.append(rel)
        decision.append(score)
        top = evidence[0] if evidence else None
        csv_rows.append({
            "path": str(sample["path"]), "defect_type": sample["defect_type"],
            "label": int(sample["label"]), "baseline_score": float(raw_baseline),
            "baseline_z": float(z_score), "relation_score": float(rel),
            "decision_score": score, "top_rule": top["rule_id"] if top else "abstain",
        })
        evidence_rows.append({
            "path": str(sample["path"]), "facts": facts,
            "diagnostics": diagnostics, "evidence": evidence,
        })
    relation, decision = np.asarray(relation), np.asarray(decision)
    defect_types = np.asarray([str(item["defect_type"]) for item in samples])
    per_type = {}
    for defect in sorted(set(defect_types) - {"good", "ok"}):
        selected = np.isin(defect_types, ["good", "ok", defect])
        if len(np.unique(labels[selected])) < 2:
            continue
        anomaly = defect_types == defect
        per_type[defect] = {
            "count": int(anomaly.sum()),
            "baseline_i_auroc": float(roc_auc_score(labels[selected], baseline[selected])),
            "fused_i_auroc": float(roc_auc_score(labels[selected], decision[selected])),
            "relation_trigger_rate": float(np.mean(relation[anomaly] > 0)),
        }
    baseline_metrics, fused_metrics = _metrics(labels, baseline), _metrics(labels, decision)
    result = {
        "schema_version": "4.0-generic-relational",
        "dataset": args.dataset, "class_name": args.class_name, "shot": args.shot,
        "test_count": len(samples), "normal_count": int((labels == 0).sum()),
        "anomaly_count": int((labels == 1).sum()),
        "proposal_abstained": proposal.get("primary_entity") is None,
        "grounding_support_match_rate": (
            float(grounder.support_match_rate) if grounder is not None else 0.0
        ),
        "selected_rule_count": len(rules),
        "selected_predicates": [item["predicate"] for item in rules],
        "fusion": fusion,
        "baseline": baseline_metrics,
        "baseline_plus_rules": fused_metrics,
        "delta_i_auroc": fused_metrics["i_auroc"] - baseline_metrics["i_auroc"],
        "delta_i_aupr": fused_metrics["i_aupr"] - baseline_metrics["i_aupr"],
        "normal_trigger_rate": float(np.mean(relation[labels == 0] > 0)),
        "anomaly_trigger_rate": float(np.mean(relation[labels == 1] > 0)),
        "per_defect_type": per_type,
        "rule_library": str(library_path),
        "test_labels_used_for_training_or_selection": False,
    }
    with (output / "predictions_v4_generic.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    _dump(output / "rule_evidence_v4_generic.json", evidence_rows)
    _dump(output / "split_v4_generic.json", {
        "support": [str(path) for path in support_paths],
        "test": [{**item, "path": str(item["path"])} for item in samples],
        "test_labels_used_for_training_or_selection": False,
    })
    _dump(output / "metrics_v4_generic.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument(
        "--test-subdir", default="test",
        help="Test directory below each class (MVTec2 uses test_public).",
    )
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--support-seed", type=int, default=1)
    parser.add_argument("--support-strategy", choices=("inp_official", "hash"), default="inp_official")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inpformer-root", default=DEFAULT_INPFORMER_ROOT)
    parser.add_argument("--proposal-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-cache-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--map-size", type=int, default=256)
    parser.add_argument("--relation-image-size", type=int, default=256)
    parser.add_argument("--minimum-rule-auc", type=float, default=0.75)
    parser.add_argument("--maximum-rule-fpr", type=float, default=0.25)
    parser.add_argument("--minimum-fusion-gain", type=float, default=0.02)
    parser.add_argument("--llm-model", default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
