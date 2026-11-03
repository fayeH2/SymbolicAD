#!/usr/bin/env python3
"""Align Qwen rules with GroundingDINO facts and freeze a validated rule bank.

Pipeline
--------
audit JSON (Qwen raw rules)
  -> typed rule adaptation
  -> atomic entity extraction
  -> targeted GroundingDINO detection prompts
  -> semantic--visual entity alignment (SVEA)
  -> canonical entity rewriting
  -> normal-support compilation / validation
  -> frozen validated rule bank JSON

This script does not use test labels and does not run an LLM at test time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from symbolicAD.evaluation.protocol_admission import validate_qwen_audit_provenance
from symbolicAD.rules.alignment import (
    build_targeted_detector_vocabulary, build_validated_rule_bank,
)
from symbolicAD.rules.facts.calibration import calibrate_from_counts
from symbolicAD.rules.facts.open_vocab import GroundingDinoDetector, Vocabulary
from symbolicAD.rules.facts.scene_builder import SceneFactBuilder, SceneFactConfig
from symbolicAD.rules.generation.ontology import ontology_for_category
from symbolicAD.rules.qwen_adapter import adapt_qwen_candidates, payloads_from_audit
from symbolicAD.rules.rule_registry import default_rule_registry
from symbolicAD.reasoning.constraint_executor import (
    ConstraintOperator, combine_statuses,
)
from symbolicAD.reasoning.violation_result import RuleStatus


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _context_payload(context: Any) -> Dict[str, Any]:
    return {
        "scene_graph": context.scene.to_dict(),
        "metadata": context.metadata,
        "components": [
            {
                "instance_id": instance.instance_id,
                "entity_type": context.scene.component_type(index),
                "bbox": list(instance.bbox),
                "centroid": list(instance.centroid),
                "confidence": instance.confidence,
                "attributes": instance.attributes,
                "source": instance.source,
            }
            for index, instance in enumerate(context.scene.components)
        ],
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    category = str(args.category)
    ontology = ontology_for_category(category)
    if ontology is None:
        raise SystemExit(f"no ontology registered for category {category!r}")

    supports = tuple(path.expanduser().resolve() for path in args.support)
    if not supports:
        raise SystemExit("at least one --support image is required")
    if any(not path.is_file() for path in supports):
        missing = [str(path) for path in supports if not path.is_file()]
        raise SystemExit(f"missing support images: {missing}")

    audit_path = args.audit_json.expanduser().resolve()
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    if args.verify_provenance:
        replay = validate_qwen_audit_provenance(
            report, category=category, support_paths=supports,
        )
        if not replay.valid:
            raise SystemExit(
                "audit provenance rejected: " + ", ".join(replay.reason_codes)
            )

    payloads = payloads_from_audit(report)
    adaptations = adapt_qwen_candidates(payloads)
    rules = tuple(row.rule for row in adaptations if row.rule is not None)
    if not rules:
        raise SystemExit("audit contains no adaptable typed rules")

    # GroundingDINO is prompted only by the canonical entities that Qwen
    # actually used (plus the container coordinate frame).
    detector_map = build_targeted_detector_vocabulary(rules, ontology)
    vocabulary = Vocabulary(detector_map)
    builder = SceneFactBuilder(
        GroundingDinoDetector(device=args.device),
        SceneFactConfig(vocabulary=vocabulary),
    )

    records: Dict[str, Dict[str, Any]] = {}
    contexts = []
    counts_per_image = []
    support_payloads = []
    for path in supports:
        record = builder.extract(path)
        key = str(path.resolve())
        records[key] = record
        counts = builder.entity_counts(record)
        counts_per_image.append(counts)
        image = np.asarray(Image.open(path).convert("RGB"))
        context = builder.build_from_record(record, image=image)
        contexts.append(context)
        support_payloads.append({
            "path": key,
            "detection_record": record,
            "counts": {name: list(row) for name, row in counts.items()},
            "scene": _context_payload(context),
        })

    calibration = calibrate_from_counts(
        counts_per_image,
        split="aligned_qwen_support",
        source_images=[str(path) for path in supports],
    )
    calibrated_contexts = []
    for path, record in zip(supports, (records[str(p.resolve())] for p in supports)):
        image = np.asarray(Image.open(path).convert("RGB"))
        calibrated_contexts.append(
            builder.build_from_record(record, calibration=calibration, image=image)
        )

    bank = build_validated_rule_bank(
        rules,
        records,
        category=category,
        ontology=ontology,
        support_contexts=calibrated_contexts,
    )

    query_payloads = []
    registry = default_rule_registry()
    for query_path in args.query:
        query_path = query_path.expanduser().resolve()
        query_record = builder.extract(query_path)
        query_context = builder.build_from_record(
            query_record, calibration=calibration,
            image=np.asarray(Image.open(query_path).convert("RGB")),
        )
        results = []
        for rule in bank.accepted_rules:
            capability = registry.resolve(rule.canonical_predicate)
            if capability is None:
                continue
            results.append(capability.executor.execute(rule, query_context))
        status = combine_statuses(
            [row.status for row in results], ConstraintOperator.AND,
        ) if results else RuleStatus.UNKNOWN
        if status == RuleStatus.VIOLATED:
            score = max(row.violation_score for row in results)
        elif status == RuleStatus.SATISFIED:
            score = 0.0
        else:
            score = None
        query_payloads.append({
            "path": str(query_path),
            "counts": {
                name: list(row)
                for name, row in builder.entity_counts(query_record).items()
            },
            "scene": _context_payload(query_context),
            "result": {
                "status": status.value,
                "score": score,
                "rule_results": [row.to_dict() for row in results],
            },
        })

    payload = {
        "audit_json": str(audit_path),
        "supports": [str(path) for path in supports],
        "detector_vocabulary": {
            key: list(value) for key, value in detector_map.items()
        },
        "calibration": calibration.to_dict(),
        "support_facts": support_payloads,
        "rule_bank": bank.to_dict(),
        "query_evaluation": query_payloads,
    }
    output_dir = args.output_dir.expanduser().resolve()
    _write_json(output_dir / "validated_rule_bank.json", payload)
    _write_json(output_dir / "detection_records.json", records)
    _write_json(output_dir / "support_facts.json", support_payloads)
    if query_payloads:
        _write_json(output_dir / "query_evaluation.json", query_payloads)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--support", nargs="+", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--query", nargs="*", type=Path, default=[],
        help="optional test images to execute the frozen bank on (no LLM)",
    )
    parser.add_argument(
        "--verify-provenance", action=argparse.BooleanOptionalAction,
        default=True,
        help="fail closed unless the audit prompt/source/support provenance matches",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    bank = payload["rule_bank"]
    print(json.dumps({
        "accepted_rule_count": bank["summary"]["accepted_rule_count"],
        "quarantined_rule_count": bank["summary"]["quarantined_rule_count"],
        "accepted_rule_ids": [
            rule["rule_id"] for rule in bank["accepted_rules"]
        ],
        "entity_bindings": {
            name: {
                "canonical_id": row["canonical_id"],
                "status": row["status"],
                "reason_codes": row["reason_codes"],
                "candidate_count": row["candidate_count"],
            }
            for name, row in bank["entity_bindings"].items()
        },
        "detector_vocabulary": payload["detector_vocabulary"],
        "query_evaluation": [
            {
                "path": row["path"],
                "status": row["result"]["status"],
                "score": row["result"]["score"],
            }
            for row in payload["query_evaluation"]
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
