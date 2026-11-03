"""Raw-response replay is the only trusted strict Qwen candidate source."""

from __future__ import annotations

import copy
import hashlib
import json

from symbolicAD.evaluation.protocol_admission import (
    model_artifact_manifest,
    rule_generation_source_manifest,
    validate_qwen_audit_provenance,
)
from symbolicAD.rules.generation.dsl import (
    QWEN_RULE_GENERATION_PROMPT,
    parse_qwen_rules_strict,
)
from symbolicAD.rules.qwen_adapter import adapt_qwen_rule


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_report(tmp_path):
    supports = []
    for index in range(4):
        path = tmp_path / f"support_{index}.png"
        path.write_bytes(f"support-{index}".encode())
        supports.append(path.resolve())
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"test"}')
    (model / "model.safetensors").write_bytes(b"test weights")
    raw = (
        '[{"rule_id":"count_part","description":"one part",'
        '"predicate_name":"COUNT","predicate_args":{"entity_type":"part"},'
        '"predicate_params":{"expected_count":1,"tolerance":0},'
        '"confidence":0.9}]'
    )
    parsed = parse_qwen_rules_strict(raw, "demo")
    evidence = [
        {"support_image": str(path), "method": "qwen_generation"}
        for path in supports
    ]
    parsed[0].support_evidence = evidence
    payload = parsed[0].to_dict()
    adaptation = adapt_qwen_rule(payload)
    support_rows = [
        {"path": str(path), "sha256": _sha(path)} for path in supports
    ]
    support_hashes = [row["sha256"] for row in support_rows]
    fingerprint = hashlib.sha256(json.dumps(
        support_hashes, separators=(",", ":"), sort_keys=True,
    ).encode()).hexdigest()
    prompt = QWEN_RULE_GENERATION_PROMPT.format(category="demo", k_shot=4)
    report = {
        "schema_version": "qwen-rule-admission-audit-v2",
        "category": "demo",
        "model": str(model.resolve()),
        "model_config_sha256": _sha(model / "config.json"),
        "model_artifacts_sha256": model_artifact_manifest(model),
        "source_sha256": rule_generation_source_manifest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "generation_config": {
            "do_sample": False,
            "force_json_prefill": True,
        },
        "support_fingerprint": fingerprint,
        "supports": support_rows,
        "raw_response": raw,
        "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "summary": {
            "candidate_count": 1,
            "executable_count": 1,
            "hard_rejected_count": 0,
            "search_started": False,
            "test_accessed": False,
            "status": "ok",
        },
        "candidates": [{
            "rule_id": "count_part",
            "description": "one part",
            "source_predicate": "COUNT",
            "adaptation_status": adaptation.status.value,
            "adaptation_reason": adaptation.reason_code,
            "typed_rule": adaptation.rule.to_dict(),
            "compile_status": "EXECUTABLE",
            "compile_reason": "READY_FOR_VISUAL_VALIDATION",
            "hard_constraint_reasons": [],
            "raw_qwen_rule": payload,
        }],
    }
    return report, tuple(supports)


def test_valid_v2_replays_raw_and_readapts_typed(tmp_path):
    report, supports = _valid_report(tmp_path)
    replay = validate_qwen_audit_provenance(
        report, category="demo", support_paths=supports,
    )
    assert replay.valid
    assert [rule.rule_id for rule in replay.fresh_rules] == ["count_part"]


def test_tampered_typed_rule_is_rejected(tmp_path):
    report, supports = _valid_report(tmp_path)
    tampered = copy.deepcopy(report)
    tampered["candidates"][0]["typed_rule"]["parameters"]["expected"] = 2
    replay = validate_qwen_audit_provenance(
        tampered, category="demo", support_paths=supports,
    )
    assert not replay.valid
    assert "TYPED_RULE_ADAPTER_MISMATCH" in replay.reason_codes


def test_support_order_and_legacy_schema_are_rejected(tmp_path):
    report, supports = _valid_report(tmp_path)
    reordered = validate_qwen_audit_provenance(
        report, category="demo", support_paths=tuple(reversed(supports)),
    )
    assert not reordered.valid
    assert "ORDERED_SUPPORT_MANIFEST_MISMATCH" in reordered.reason_codes
    legacy = copy.deepcopy(report)
    legacy["schema_version"] = "qwen-rule-admission-audit-v1"
    replay = validate_qwen_audit_provenance(
        legacy, category="demo", support_paths=supports,
    )
    assert not replay.valid
    assert "AUDIT_SCHEMA_NOT_V2" in replay.reason_codes
