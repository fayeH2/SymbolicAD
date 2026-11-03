"""Fail-closed protocol admission for few-shot rule experiments.

This module does not score anomalies.  It prevents a diagnostic experiment
that consumed additional normal images or mismatched supports from being
silently promoted into the strict K-shot model table.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from symbolicAD.rules.generation.dsl import parse_qwen_rules_strict
from symbolicAD.rules.generation.ontology_prompt import build_rule_generation_prompt
from symbolicAD.rules.qwen_adapter import adapt_qwen_rule
from symbolicAD.rules.rule_schema import RuleSpec


@dataclass(frozen=True)
class ProtocolAdmission:
    expected_shot: int
    rule_support_count: int
    qwen_support_matches: bool
    qwen_audit_provenance_valid: bool
    baseline_support_matches: Optional[bool]
    baseline_artifact_valid: bool
    extra_normal_count: int
    test_detection_after_rule_freeze: bool
    rule_mechanism_protocol_valid: bool
    strict_kshot_protocol_eligible: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_shot": self.expected_shot,
            "rule_support_count": self.rule_support_count,
            "qwen_support_matches": self.qwen_support_matches,
            "qwen_audit_provenance_valid": self.qwen_audit_provenance_valid,
            "baseline_support_matches": self.baseline_support_matches,
            "baseline_artifact_valid": self.baseline_artifact_valid,
            "extra_normal_count": self.extra_normal_count,
            "test_detection_after_rule_freeze": (
                self.test_detection_after_rule_freeze
            ),
            "rule_mechanism_protocol_valid": self.rule_mechanism_protocol_valid,
            "strict_kshot_protocol_eligible": self.strict_kshot_protocol_eligible,
            "reason_codes": list(self.reason_codes),
            "experiment_tier": (
                "STRICT_KSHOT_CANDIDATE"
                if self.strict_kshot_protocol_eligible
                else "VALIDATION_ASSISTED_DIAGNOSTIC"
            ),
        }


@dataclass(frozen=True)
class QwenAuditReplay:
    """Fail-closed result of replaying candidates from the raw LLM response."""

    valid: bool
    reason_codes: Tuple[str, ...]
    fresh_rules: Tuple[RuleSpec, ...]
    fresh_payloads: Tuple[Mapping[str, Any], ...]
    parsed_candidate_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "reason_codes": list(self.reason_codes),
            "parsed_candidate_count": self.parsed_candidate_count,
            "fresh_rule_ids": [rule.rule_id for rule in self.fresh_rules],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def model_artifact_manifest(model_root: Path) -> Dict[str, str]:
    """Hash every generation-affecting model/processor artifact.

    Weight shards are intentionally included.  A config-only hash cannot
    distinguish two checkpoints with the same architecture.
    """
    root = model_root.expanduser().resolve()
    fixed = {
        "config.json", "generation_config.json", "chat_template.json",
        "preprocessor_config.json", "video_preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "model.safetensors.index.json",
    }
    paths = [
        path for path in root.iterdir()
        if path.is_file() and (path.name in fixed or path.suffix == ".safetensors")
    ]
    return {path.name: _sha256_file(path) for path in sorted(paths)}


def rule_generation_source_manifest() -> Dict[str, str]:
    project = Path(__file__).resolve().parents[2]
    relative_paths = (
        "scripts/generate_qwen_rule_audit.py",
        "symbolicAD/rules/generation/dsl.py",
        "symbolicAD/rules/generation/ontology.py",
        "symbolicAD/rules/generation/ontology_prompt.py",
        "symbolicAD/rules/generation/qwen_interface.py",
        "symbolicAD/rules/qwen_adapter.py",
    )
    return {
        relative: _sha256_file(project / relative) for relative in relative_paths
    }


def _fingerprint_hashes(hashes: Sequence[str]) -> str:
    encoded = json.dumps(
        list(hashes), separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_qwen_audit_provenance(
    report: Mapping[str, Any],
    *,
    category: str,
    support_paths: Sequence[Path],
) -> QwenAuditReplay:
    """Replay a v2 audit from raw JSON and verify every stored derivative.

    Stored ``raw_qwen_rule`` and ``typed_rule`` rows are comparisons only.
    Returned rules are freshly parsed and adapted from ``raw_response``.
    """
    reasons = []
    supports = tuple(path.expanduser().resolve() for path in support_paths)
    if report.get("schema_version") != "qwen-rule-admission-audit-v2":
        reasons.append("AUDIT_SCHEMA_NOT_V2")
    if str(report.get("category", "")) != str(category):
        reasons.append("AUDIT_CATEGORY_MISMATCH")
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    if summary.get("status") != "ok":
        reasons.append("AUDIT_STATUS_NOT_OK")
    if summary.get("search_started") is not False or summary.get("test_accessed") is not False:
        reasons.append("AUDIT_NOT_PROPOSAL_ONLY")

    support_rows = report.get("supports")
    actual_hashes = []
    if len(supports) == 0 or any(not path.is_file() for path in supports):
        reasons.append("SUPPORT_FILE_MISSING")
    else:
        actual_hashes = [_sha256_file(path) for path in supports]
        if len(actual_hashes) != len(set(actual_hashes)):
            reasons.append("DUPLICATE_SUPPORT_CONTENT")
    expected_rows = [
        {"path": str(path), "sha256": digest}
        for path, digest in zip(supports, actual_hashes)
    ]
    if support_rows != expected_rows:
        reasons.append("ORDERED_SUPPORT_MANIFEST_MISMATCH")
    if actual_hashes and report.get("support_fingerprint") != _fingerprint_hashes(actual_hashes):
        reasons.append("SUPPORT_FINGERPRINT_MISMATCH")

    prompt = build_rule_generation_prompt(category, len(supports))
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if report.get("prompt_sha256") != prompt_sha:
        reasons.append("PROMPT_SHA256_MISMATCH")
    raw_response = report.get("raw_response")
    if not isinstance(raw_response, str) or not raw_response.strip():
        reasons.append("RAW_RESPONSE_MISSING")
        raw_response = ""
    raw_sha = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
    if report.get("raw_response_sha256") != raw_sha:
        reasons.append("RAW_RESPONSE_SHA256_MISMATCH")

    model_root = Path(str(report.get("model", ""))).expanduser()
    if not model_root.is_dir():
        reasons.append("MODEL_ROOT_MISSING")
    else:
        current_model_manifest = model_artifact_manifest(model_root)
        recorded_model_manifest = report.get("model_artifacts_sha256")
        config_path = model_root / "config.json"
        if (
            not config_path.is_file()
            or report.get("model_config_sha256") != _sha256_file(config_path)
        ):
            reasons.append("MODEL_CONFIG_SHA256_MISMATCH")
        if not any(name.endswith(".safetensors") for name in current_model_manifest):
            reasons.append("MODEL_WEIGHTS_MISSING")
        if recorded_model_manifest != current_model_manifest:
            reasons.append("MODEL_ARTIFACT_MANIFEST_MISMATCH")
    if report.get("source_sha256") != rule_generation_source_manifest():
        reasons.append("RULE_GENERATION_SOURCE_MISMATCH")
    generation = report.get("generation_config")
    if not isinstance(generation, Mapping) or generation.get("do_sample") is not False:
        reasons.append("NONDETERMINISTIC_GENERATION_CONFIG")
    if not isinstance(generation, Mapping) or generation.get("force_json_prefill") is not True:
        reasons.append("STRICT_JSON_PREFILL_NOT_RECORDED")

    parsed = parse_qwen_rules_strict(raw_response, category)
    if not parsed:
        reasons.append("STRICT_RAW_RESPONSE_PARSE_FAILED")
    expected_evidence = [
        {"support_image": str(path), "method": "qwen_generation"}
        for path in supports
    ]
    fresh_payloads = []
    for rule in parsed:
        rule.support_evidence = list(expected_evidence)
        fresh_payloads.append(rule.to_dict())

    stored_candidates = report.get("candidates")
    if not isinstance(stored_candidates, list):
        stored_candidates = []
        reasons.append("CANDIDATE_LEDGER_MISSING")
    if len(stored_candidates) != len(fresh_payloads):
        reasons.append("CANDIDATE_COUNT_MISMATCH")

    fresh_rules = []
    executable_count = 0
    if len(stored_candidates) == len(fresh_payloads):
        # Imports stay local so protocol classification remains lightweight.
        from symbolicAD.rules.rule_registry import default_rule_registry
        from symbolicAD.rules.search.symbolicdet_evolution import hard_constraint_reasons

        registry = default_rule_registry()
        for stored, payload in zip(stored_candidates, fresh_payloads):
            try:
                raw_matches = _canonical_json(stored.get("raw_qwen_rule")) == _canonical_json(payload)
            except (TypeError, ValueError):
                raw_matches = False
            if not raw_matches:
                reasons.append("RAW_RESPONSE_CANDIDATE_MISMATCH")
                continue
            adaptation = adapt_qwen_rule(payload)
            typed = adaptation.rule
            typed_payload = None if typed is None else typed.to_dict()
            try:
                typed_matches = _canonical_json(stored.get("typed_rule")) == _canonical_json(typed_payload)
            except (TypeError, ValueError):
                typed_matches = False
            if not typed_matches:
                reasons.append("TYPED_RULE_ADAPTER_MISMATCH")
            if stored.get("adaptation_status") != adaptation.status.value:
                reasons.append("ADAPTATION_STATUS_MISMATCH")
            if stored.get("adaptation_reason") != adaptation.reason_code:
                reasons.append("ADAPTATION_REASON_MISMATCH")
            hard_reasons = (
                ["MALFORMED_QWEN_RULE"] if typed is None
                else list(hard_constraint_reasons((typed,), registry))
            )
            executable = typed is not None and not hard_reasons
            executable_count += int(executable)
            if stored.get("hard_constraint_reasons") != hard_reasons:
                reasons.append("HARD_CONSTRAINT_LEDGER_MISMATCH")
            expected_compile = "EXECUTABLE" if executable else "REJECTED"
            if stored.get("compile_status") != expected_compile:
                reasons.append("COMPILE_STATUS_MISMATCH")
            if typed is not None:
                fresh_rules.append(typed)

    if summary.get("candidate_count") != len(fresh_payloads):
        reasons.append("SUMMARY_CANDIDATE_COUNT_MISMATCH")
    if summary.get("executable_count") != executable_count:
        reasons.append("SUMMARY_EXECUTABLE_COUNT_MISMATCH")
    if summary.get("hard_rejected_count") != len(fresh_payloads) - executable_count:
        reasons.append("SUMMARY_REJECTED_COUNT_MISMATCH")

    unique_reasons = tuple(dict.fromkeys(reasons))
    valid = not unique_reasons
    return QwenAuditReplay(
        valid=valid,
        reason_codes=unique_reasons,
        fresh_rules=tuple(fresh_rules) if valid else (),
        fresh_payloads=tuple(fresh_payloads) if valid else (),
        parsed_candidate_count=len(parsed),
    )


def assess_strict_kshot_protocol(
    *,
    rule_support_hashes: Iterable[str],
    qwen_support_hash_sets: Sequence[Iterable[str]],
    qwen_audit_provenance_valid: bool,
    baseline_support_hashes: Optional[Iterable[str]],
    baseline_artifact_valid: bool,
    extra_normal_hashes: Iterable[str],
    test_detection_after_rule_freeze: bool,
    expected_shot: int = 4,
) -> ProtocolAdmission:
    """Classify a rule experiment without consulting anomaly/test labels."""
    expected_shot = int(expected_shot)
    if expected_shot < 1:
        raise ValueError("expected_shot must be positive")
    rule_support = frozenset(str(value) for value in rule_support_hashes)
    qwen_sets = tuple(
        frozenset(str(value) for value in values)
        for values in qwen_support_hash_sets
    )
    baseline = (
        None if baseline_support_hashes is None
        else frozenset(str(value) for value in baseline_support_hashes)
    )
    # Only images outside the declared K supports consume an extra normal-data
    # budget.  Reusing a support for a support-derived counterfactual is allowed.
    extra_normal = frozenset(
        str(value) for value in extra_normal_hashes
    ) - rule_support

    qwen_matches = bool(qwen_sets) and all(
        values == rule_support for values in qwen_sets
    )
    baseline_matches = None if baseline is None else baseline == rule_support
    reasons = []
    if len(rule_support) != expected_shot:
        reasons.append("RULE_SUPPORT_COUNT_MISMATCH")
    if not qwen_matches:
        reasons.append("QWEN_SUPPORT_PROVENANCE_MISMATCH")
    if not qwen_audit_provenance_valid:
        reasons.append("QWEN_AUDIT_PROVENANCE_INVALID")
    if baseline is None:
        reasons.append("BASELINE_SUPPORT_NOT_SUPPLIED")
    elif not baseline_matches:
        reasons.append("BASELINE_SUPPORT_MISMATCH")
    if not baseline_artifact_valid:
        reasons.append("BASELINE_ARTIFACT_PROVENANCE_INVALID")
    if extra_normal:
        reasons.append("EXTRA_NORMAL_IMAGES_USED_FOR_CALIBRATION_OR_SEARCH")
    if not test_detection_after_rule_freeze:
        reasons.append("TEST_PIXELS_ACCESSED_BEFORE_RULE_FREEZE")

    mechanism_valid = bool(
        qwen_matches and qwen_audit_provenance_valid
        and test_detection_after_rule_freeze
    )
    strict = bool(
        len(rule_support) == expected_shot
        and qwen_matches
        and qwen_audit_provenance_valid
        and baseline_matches is True
        and baseline_artifact_valid
        and not extra_normal
        and test_detection_after_rule_freeze
    )
    return ProtocolAdmission(
        expected_shot=expected_shot,
        rule_support_count=len(rule_support),
        qwen_support_matches=qwen_matches,
        qwen_audit_provenance_valid=bool(qwen_audit_provenance_valid),
        baseline_support_matches=baseline_matches,
        baseline_artifact_valid=bool(baseline_artifact_valid),
        extra_normal_count=len(extra_normal),
        test_detection_after_rule_freeze=bool(
            test_detection_after_rule_freeze
        ),
        rule_mechanism_protocol_valid=mechanism_valid,
        strict_kshot_protocol_eligible=strict,
        reason_codes=tuple(reasons),
    )


__all__ = [
    "ProtocolAdmission", "QwenAuditReplay", "assess_strict_kshot_protocol",
    "validate_qwen_audit_provenance", "model_artifact_manifest",
    "rule_generation_source_manifest",
]
