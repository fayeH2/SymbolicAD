#!/usr/bin/env python3
"""Generate a provenance-complete Qwen rule audit from explicit K-shot supports.

The LLM only proposes candidates.  Exact typed adaptation and executor
availability are recorded here; search and test evaluation happen in a later,
separately frozen stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from symbolicAD.rules.generation.ontology import ontology_for_category
from symbolicAD.rules.generation.ontology_prompt import build_rule_generation_prompt
from symbolicAD.rules.generation.qwen_interface import QwenVLRuleGenerator
from symbolicAD.evaluation.protocol_admission import (
    model_artifact_manifest,
    rule_generation_source_manifest,
)
from symbolicAD.rules.qwen_adapter import adapt_qwen_rule
from symbolicAD.rules.rule_registry import default_rule_registry
from symbolicAD.rules.search.symbolicdet_evolution import hard_constraint_reasons


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--support", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--model", default="/home/hjh/AgentModels/Qwen3-VL-8B-Thinking"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--force-json-prefill", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    supports = tuple(path.expanduser().resolve() for path in args.support)
    if len(supports) < 1 or len(set(supports)) != len(supports):
        raise ValueError("support paths must be a non-empty unique list")
    if any(not path.is_file() for path in supports):
        missing = [str(path) for path in supports if not path.is_file()]
        raise FileNotFoundError(f"missing support images: {missing}")
    support_rows = [
        {"path": str(path), "sha256": _sha256(path)} for path in supports
    ]
    support_hashes = [row["sha256"] for row in support_rows]
    if len(support_hashes) != len(set(support_hashes)):
        raise ValueError("support images must also have unique content hashes")
    support_fingerprint = hashlib.sha256(
        json.dumps(
            support_hashes, separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    model_root = Path(args.model).expanduser().resolve()
    config_path = model_root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    generator = QwenVLRuleGenerator(
        model_path=str(model_root), device=args.device,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
        force_json_prefill=args.force_json_prefill,
    )
    prompt = build_rule_generation_prompt(args.category, len(supports))
    rules = generator.generate_rules(
        args.category, supports, k_shot=len(supports), prompt=prompt
    )
    registry = default_rule_registry()
    candidates = []
    executable_count = 0
    for qwen_rule in rules:
        payload = qwen_rule.to_dict()
        adaptation = adapt_qwen_rule(payload)
        typed = adaptation.rule
        reasons = (
            ["MALFORMED_QWEN_RULE"] if typed is None
            else list(hard_constraint_reasons((typed,), registry))
        )
        executable = typed is not None and not reasons
        executable_count += int(executable)
        candidates.append({
            "rule_id": adaptation.source_rule_id,
            "description": qwen_rule.description,
            "source_predicate": adaptation.source_predicate,
            "adaptation_status": adaptation.status.value,
            "adaptation_reason": adaptation.reason_code,
            "typed_rule": None if typed is None else typed.to_dict(),
            "compile_status": "EXECUTABLE" if executable else "REJECTED",
            "compile_reason": "READY_FOR_VISUAL_VALIDATION" if executable else (
                reasons[0] if reasons else adaptation.reason_code
            ),
            "hard_constraint_reasons": reasons,
            "raw_qwen_rule": payload,
        })

    raw_response = generator.last_raw_response
    ontology = ontology_for_category(args.category)
    report = {
        "schema_version": "qwen-rule-admission-audit-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "category": args.category,
        "model": str(model_root),
        "model_config_sha256": _sha256(config_path),
        "model_artifacts_sha256": model_artifact_manifest(model_root),
        "source_sha256": rule_generation_source_manifest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "ontology": None if ontology is None else ontology.to_dict(),
        "ontology_sha256": (
            None if ontology is None else hashlib.sha256(
                json.dumps(
                    ontology.to_dict(), ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        ),
        "generation_config": {
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking_requested": bool(args.enable_thinking),
            "thinking_control_supported": generator.thinking_control_supported,
            "force_json_prefill": bool(args.force_json_prefill),
            "do_sample": False,
        },
        "support_fingerprint": support_fingerprint,
        "supports": support_rows,
        "raw_response": raw_response,
        "raw_response_sha256": hashlib.sha256(
            raw_response.encode("utf-8")
        ).hexdigest(),
        "summary": {
            "candidate_count": len(candidates),
            "executable_count": executable_count,
            "hard_rejected_count": len(candidates) - executable_count,
            "search_started": False,
            "test_accessed": False,
            "status": "ok" if candidates else "parse_error",
        },
        "candidates": candidates,
    }
    _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report["summary"], sort_keys=True), flush=True)
    return 0 if candidates else 2


if __name__ == "__main__":
    raise SystemExit(main())
