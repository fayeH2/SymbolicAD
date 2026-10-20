"""Controlled cable experiment for Qwen-guided relational normality rules."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .augmentations import build_calibration_views
from .cable_relations import (
    CableRelationalRuleEngine,
    CableSceneGrounder,
    build_relational_calibration_views,
    build_relational_library,
    compile_relational_rules,
)
from .data import collect_test_samples, select_inpformer_support, select_support
from .inpformer_adapter import (
    DEFAULT_INPFORMER_ROOT,
    INPFormerAdapter,
    file_sha256,
    load_prediction_cache,
    save_prediction_cache,
)
from .qwen_relations import (
    DEFAULT_MODEL_PATH,
    QwenRelationalProposer,
    make_rgb_support_montage,
    validate_relational_proposal,
)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    return {
        "i_auroc": float(roc_auc_score(labels, scores)) if np.ptp(scores) > 0 else 0.5,
        "i_aupr": float(average_precision_score(labels, scores)),
    }


def _cached_infer(
    adapter: INPFormerAdapter,
    path: Path,
    images: Sequence,
    identifiers: Sequence[str],
    metadata: Mapping[str, Any],
    batch_size: int,
):
    cached = load_prediction_cache(path, identifiers, expected_metadata=dict(metadata))
    if cached is not None:
        return cached
    result = adapter.infer_images(images, batch_size=batch_size, identifiers=identifiers)
    save_prediction_cache(path, result, metadata=dict(metadata))
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _image_sha256(image: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(image))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _extract_relation_rows(
    views, grounder: CableSceneGrounder,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[int], List[int], List[str]]:
    normal_rows: List[Dict[str, float]] = []
    pseudo_rows: List[Dict[str, float]] = []
    normal_groups: List[int] = []
    pseudo_groups: List[int] = []
    pseudo_families: List[str] = []
    for view in views:
        facts, _ = grounder.extract(view.image)
        if view.is_pseudo_anomaly:
            pseudo_rows.append(facts)
            pseudo_groups.append(int(view.source_index))
            pseudo_families.append(str(view.family))
        else:
            normal_rows.append(facts)
            normal_groups.append(int(view.source_index))
    return normal_rows, pseudo_rows, normal_groups, pseudo_groups, pseudo_families


def _build_grounder_crossfit_folds(
    support_paths: Sequence[Path], image_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Strict LOO: the held support never enters that fold's visual prototypes."""
    folds: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []
    for held_index, held_path in enumerate(support_paths):
        train_paths = [path for index, path in enumerate(support_paths) if index != held_index]
        fold_grounder = CableSceneGrounder(image_size=image_size).fit(train_paths)
        train_views = build_relational_calibration_views(train_paths, fold_grounder)
        held_views = build_relational_calibration_views([held_path], fold_grounder)
        train_rows, _, _, _, _ = _extract_relation_rows(train_views, fold_grounder)
        held_normal, held_pseudo, _, _, held_families = _extract_relation_rows(
            held_views, fold_grounder
        )
        folds.append({
            "fold": held_index,
            "train_normal_rows": train_rows,
            "held_normal_rows": held_normal,
            "held_pseudo_rows": held_pseudo,
            "held_pseudo_families": held_families,
        })
        manifest.append({
            "fold": held_index,
            "held_support": str(held_path.resolve()),
            "train_support": [str(path.resolve()) for path in train_paths],
            "train_normal_count": len(train_rows),
            "held_normal_count": len(held_normal),
            "held_pseudo_count": len(held_pseudo),
            "grounder_refit_without_held_support": True,
        })
    return folds, manifest


def _per_type_metrics(samples, labels, baseline, relation, decision):
    types = np.asarray([str(item["defect_type"]) for item in samples])
    result = {}
    for defect_type in sorted(set(types) - {"good", "ok"}):
        selected = np.isin(types, ["good", "ok", defect_type])
        if len(np.unique(labels[selected])) < 2:
            continue
        anomaly = types == defect_type
        result[defect_type] = {
            "count": int(anomaly.sum()),
            "baseline_i_auroc": float(roc_auc_score(labels[selected], baseline[selected])),
            "relational_i_auroc": float(roc_auc_score(labels[selected], relation[selected])),
            "fused_i_auroc": float(roc_auc_score(labels[selected], decision[selected])),
            "relation_trigger_rate": float(np.mean(relation[anomaly] > 0)),
        }
    return result


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    if args.class_name != "cable":
        raise ValueError(
            "The first controlled relational grounder is intentionally scoped to class=cable"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    llm_model_hash = getattr(args, "llm_model_hash", None)
    if llm_model_hash:
        llm_model_hash = str(llm_model_hash).strip().lower()
        if len(llm_model_hash) != 64 or any(
            character not in "0123456789abcdef" for character in llm_model_hash
        ):
            raise ValueError("--llm-model-hash must be a 64-character SHA-256 hex digest")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    class_root = Path(args.data_root).expanduser().resolve() / args.class_name
    support_paths = (
        select_inpformer_support(
            args.data_root, args.dataset, args.class_name, args.shot, args.support_seed
        )
        if args.support_strategy == "inp_official"
        else select_support(class_root, args.shot, args.support_seed)
    )
    test_samples = collect_test_samples(class_root)
    test_paths = [Path(item["path"]) for item in test_samples]

    montage = make_rgb_support_montage(
        support_paths, output / "normal_support_relational_montage.jpg"
    )
    if args.proposal_json:
        raw_payload = json.loads(Path(args.proposal_json).read_text(encoding="utf-8"))
        proposal = validate_relational_proposal(raw_payload)
        proposal["raw_output"] = json.dumps(raw_payload, ensure_ascii=False)
        proposal_source = "provided_json"
    else:
        proposer = QwenRelationalProposer(
            model_path=args.llm_model,
            device=args.llm_device,
            max_new_tokens=args.max_new_tokens,
        )
        proposal = proposer.propose(args.class_name, len(support_paths), montage)
        proposal_source = "qwen3_vl_relational"
        del proposer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    proposal["model"] = str(Path(args.llm_model).expanduser().resolve())
    (output / "qwen_relational_raw.txt").write_text(
        proposal.get("raw_output", "") + "\n", encoding="utf-8"
    )
    _dump(output / "qwen_relational_proposal.json", {
        key: value for key, value in proposal.items() if key != "raw_output"
    })

    grounder = CableSceneGrounder(image_size=args.relation_image_size).fit(support_paths)
    relation_views = build_relational_calibration_views(support_paths, grounder)
    crossfit_folds, crossfit_manifest = _build_grounder_crossfit_folds(
        support_paths, args.relation_image_size
    )

    generic_views = build_calibration_views(support_paths, seed=args.calibration_seed)
    generic_images = [item.image for item in generic_views]
    generic_digests = [_image_sha256(image) for image in generic_images]
    generic_ids = [
        f"{'pseudo' if item.is_pseudo_anomaly else 'normal'}://"
        f"{item.source_index}/{item.transform}/{digest}"
        for item, digest in zip(generic_views, generic_digests)
    ]
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
    cache_metadata = adapter.cache_metadata()
    support_hashes = {
        str(path.resolve()): file_sha256(path) for path in support_paths
    }
    generic_cache_metadata = {
        **cache_metadata,
        "cache_purpose": "generic_support_calibration_for_relational_v3",
        "dataset": args.dataset,
        "class_name": args.class_name,
        "support_strategy": args.support_strategy,
        "support_seed": args.support_seed,
        "calibration_seed": args.calibration_seed,
        "support_sha256": support_hashes,
        "augmentation_schema": "generic_calibration_views_v1",
        "image_sha256": generic_digests,
    }
    cache_root = (
        Path(args.prediction_cache_dir).expanduser().resolve()
        if args.prediction_cache_dir else output
    )
    support_predictions = _cached_infer(
        adapter,
        cache_root / "cache" / "support_inp.npz",
        support_paths,
        [str(path.resolve()) for path in support_paths],
        cache_metadata,
        args.batch_size,
    )
    test_predictions = _cached_infer(
        adapter,
        cache_root / "cache" / "test_inp.npz",
        test_paths,
        [str(path.resolve()) for path in test_paths],
        cache_metadata,
        args.batch_size,
    )
    generic_predictions = _cached_infer(
        adapter,
        cache_root / "cache" / "calibration_inp_relational_v3.npz",
        generic_images,
        generic_ids,
        generic_cache_metadata,
        args.batch_size,
    )
    adapter.close()
    del adapter
    gc.collect()

    normal_rows, pseudo_rows, normal_groups, pseudo_groups, pseudo_families = (
        _extract_relation_rows(relation_views, grounder)
    )
    relation_manifest = []
    for view in relation_views:
        relation_manifest.append({
            "source_index": view.source_index,
            "transform": view.transform,
            "label": int(view.is_pseudo_anomaly),
            "family": view.family,
        })

    allowed_families = tuple(
        item.strip() for item in args.allowed_rule_families.split(",") if item.strip()
    )
    rules, audits = compile_relational_rules(
        proposal,
        normal_rows,
        pseudo_rows,
        pseudo_families,
        normal_groups,
        pseudo_groups,
        allowed_families=allowed_families,
        minimum_cv_auc=args.minimum_rule_auc,
        maximum_normal_fpr=args.maximum_rule_fpr,
        crossfit_folds=crossfit_folds,
    )
    generic_normal = np.asarray([
        not item.is_pseudo_anomaly for item in generic_views
    ], dtype=bool)
    library = build_relational_library(
        dataset=args.dataset,
        class_name=args.class_name,
        support_paths=[str(path.resolve()) for path in support_paths],
        grounder=grounder,
        proposal=proposal,
        rules=rules,
        audits=audits,
        normal_baseline_scores=generic_predictions["scores"][generic_normal],
        pseudo_baseline_scores=generic_predictions["scores"][~generic_normal],
        metadata={
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_sha256": cache_metadata["checkpoint_sha256"],
            "support_sha256": support_hashes,
            "proposal_source": proposal_source,
            "proposal_sha256": _sha256_text(json.dumps(
                {key: value for key, value in proposal.items() if key != "raw_output"},
                sort_keys=True,
            )),
            "calibration_seed": args.calibration_seed,
            "support_seed": args.support_seed,
            "support_strategy": args.support_strategy,
            "allowed_rule_families": list(allowed_families),
            "rule_cv_protocol": "leave_one_support_out_refit_grounder",
            "adapter_config": {
                "inpformer_root": str(Path(args.inpformer_root).expanduser().resolve()),
                "encoder": args.encoder,
                "inp_num": args.inp_num,
                "input_size": args.input_size,
                "crop_size": args.crop_size,
                "map_size": args.map_size,
            },
            "qwen_model_aggregate_sha256": llm_model_hash,
            "llm_loaded_in_this_run": proposal_source == "qwen3_vl_relational",
            "proposal_replay_source": (
                str(Path(args.proposal_json).expanduser().resolve())
                if args.proposal_json else None
            ),
            "runtime_loads_llm": False,
            "grounder_training_device": "cpu_deterministic_cv",
            "baseline_training_device": args.device,
            "llm_induction_device": (
                args.llm_device if proposal_source == "qwen3_vl_relational" else None
            ),
        },
    )
    engine = CableRelationalRuleEngine(library)
    library_path = engine.save(output / "normality_rules_v3_relational.json")

    labels = np.asarray([int(item["label"]) for item in test_samples], dtype=int)
    baseline = np.asarray(test_predictions["scores"], dtype=float)
    relation_scores, decisions, rows, evidence_rows = [], [], [], []
    for sample, baseline_score in zip(test_samples, baseline):
        relation_score, evidence, facts, scene = engine.evaluate(sample["path"])
        decision = engine.decision_score(float(baseline_score), relation_score)
        relation_scores.append(relation_score)
        decisions.append(decision)
        top = evidence[0] if evidence else None
        rows.append({
            "path": str(sample["path"]),
            "defect_type": sample["defect_type"],
            "label": int(sample["label"]),
            "baseline_score": float(baseline_score),
            "relational_score": relation_score,
            "decision_score": decision,
            "top_rule": top["rule_id"] if top else "abstain",
            "top_violation": top["violation"] if top else 0.0,
        })
        evidence_rows.append({
            "path": str(sample["path"]),
            "facts": facts,
            "scene": scene,
            "evidence": evidence,
        })
    relation_scores = np.asarray(relation_scores, dtype=float)
    decisions = np.asarray(decisions, dtype=float)
    baseline_metrics = _metrics(labels, baseline)
    relation_metrics = _metrics(labels, relation_scores)
    fused_metrics = _metrics(labels, decisions)
    normal_mask = labels == 0
    result = {
        "schema_version": "3.0-relational",
        "evaluation_status": "exploratory_development",
        "dataset": args.dataset,
        "class_name": args.class_name,
        "shot": args.shot,
        "test_count": len(test_samples),
        "normal_count": int(normal_mask.sum()),
        "anomaly_count": int((~normal_mask).sum()),
        "rules": str(library_path),
        "qwen_proposal_source": proposal_source,
        "selected_rule_count": len(rules),
        "selected_predicates": [rule["predicate"] for rule in rules],
        "selected_facts": [rule["fact"] for rule in rules],
        "fusion": library["fusion"],
        "baseline": baseline_metrics,
        "relations_only": relation_metrics,
        "baseline_plus_relations": fused_metrics,
        "delta_i_auroc": fused_metrics["i_auroc"] - baseline_metrics["i_auroc"],
        "delta_i_aupr": fused_metrics["i_aupr"] - baseline_metrics["i_aupr"],
        "normal_relation_trigger_rate": float(np.mean(relation_scores[normal_mask] > 0)),
        "per_defect_type": _per_type_metrics(
            test_samples, labels, baseline, relation_scores, decisions
        ),
        "test_labels_used_for_training_or_selection": False,
        "caveat": (
            "The code path is support-only, but cable was chosen after prior benchmark "
            "diagnosis; treat this run as exploratory and confirm on a frozen new class/seed."
        ),
    }
    with (output / "predictions_v3_relational.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _dump(output / "rule_evidence_v3_relational.json", evidence_rows)
    _dump(output / "candidate_audit_v3_relational.json", audits)
    _dump(output / "relational_calibration_manifest.json", relation_manifest)
    _dump(output / "crossfit_manifest_v3_relational.json", crossfit_manifest)
    _dump(output / "split_v3_relational.json", {
        "support": [str(path.resolve()) for path in support_paths],
        "test": [{**item, "path": str(item["path"])} for item in test_samples],
        "test_labels_used_for_training_or_selection": False,
    })
    _dump(output / "metrics_v3_relational.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen-guided executable relational rules on top of INP-Former"
    )
    parser.add_argument("--dataset", default="mvtec")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--class-name", default="cable")
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--support-seed", type=int, default=1)
    parser.add_argument("--support-strategy", choices=["inp_official", "hash"], default="inp_official")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inpformer-root", default=DEFAULT_INPFORMER_ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-cache-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--llm-device", default="cuda:1")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--llm-model-hash",
        help="Optional precomputed aggregate SHA-256 of the local LLM directory",
    )
    parser.add_argument("--proposal-json")
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder", default="dinov2reg_vit_base_14")
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--map-size", type=int, default=256)
    parser.add_argument("--relation-image-size", type=int, default=256)
    parser.add_argument("--calibration-seed", type=int, default=20261013)
    parser.add_argument("--allowed-rule-families", default="identity")
    parser.add_argument("--minimum-rule-auc", type=float, default=0.75)
    parser.add_argument("--maximum-rule-fpr", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
