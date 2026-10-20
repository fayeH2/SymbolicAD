"""Qwen-guided candidates for an executable transistor normality DSL.

This module is deliberately isolated from the existing v4/v5 rule paths.  It
does one job: query a local Qwen3-VL model independently for each known-normal
support image, validate the answer against a closed typed vocabulary, and
aggregate equal candidates by a canonical semantic key.  Candidate frequency
is evidence for the later support-only stability compiler; it is not itself a
test-time anomaly score.

The LLM is never needed after the candidate bank has been written.  In
particular, free-form text (``reason`` and ``normality_summary``) is excluded
from canonical keys and therefore cannot silently change executable semantics.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from PIL import Image

from .qwen_rules import DEFAULT_MODEL_PATH, extract_json_object


SCHEMA_VERSION = "neurosymbolic-fsad.logic-candidates.v7"

# This vocabulary is intentionally small.  Every member has a deterministic
# grounding contract that a later visual predicate implementation can satisfy.
ALLOWED_PREDICATES: Tuple[str, ...] = (
    "object_count",
    "part_count",
    "attached_to",
    "relative_position",
    "ordered_spacing",
    "mirror_symmetry",
    "similar_width",
    "similar_orientation",
    "continuity",
    "package_shape",
    "package_appearance",
    "package_pose",
)

_RULE_KEYS = {"id", "predicate", "arguments", "expected", "confidence", "reason"}
_PAYLOAD_KEYS = {"normality_summary", "rules"}

_ARGUMENT_SPECS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "object_count": {"entity": ("transistor",)},
    "part_count": {"whole": ("transistor",), "part": ("lead",)},
    "attached_to": {"part": ("lead",), "whole": ("package",)},
    "relative_position": {
        "subject": ("lead_group",),
        "reference": ("package",),
        "relation": ("above", "below", "left_of", "right_of", "around"),
    },
    "ordered_spacing": {
        "entity": ("lead",),
        "axis": ("horizontal", "vertical"),
    },
    "mirror_symmetry": {
        "entity": ("transistor", "package", "lead_group"),
        "axis": ("horizontal", "vertical"),
    },
    "similar_width": {"entity": ("lead",)},
    "similar_orientation": {"entity": ("lead",)},
    "continuity": {"entity": ("lead", "package_outline")},
    "package_shape": {
        "entity": ("package",),
        "shape": ("rectangular", "square", "circular", "rounded_rectangle"),
    },
    "package_appearance": {
        "entity": ("package",),
        "property": ("uniform_surface", "consistent_color", "intact_printing"),
    },
    "package_pose": {
        "entity": ("package",),
        "pose": ("upright", "horizontal", "vertical", "centered"),
    },
}

_FIXED_EXPECTED: Dict[str, Any] = {
    "attached_to": "all",
    "relative_position": True,
    "ordered_spacing": "uniform",
    "mirror_symmetry": True,
    "similar_width": True,
    "similar_orientation": True,
    "continuity": "continuous",
    "package_shape": True,
    "package_appearance": True,
    "package_pose": True,
}


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(f"{context} has invalid fields ({', '.join(details)})")


def _validate_text(value: Any, field: str, maximum: int, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if nonempty and not text:
        raise ValueError(f"{field} must not be empty")
    if len(text) > maximum:
        raise ValueError(f"{field} exceeds {maximum} characters")
    return text


def _validate_expected(predicate: str, value: Any) -> Any:
    if predicate in {"object_count", "part_count"}:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{predicate}.expected must be an integer")
        upper = 8 if predicate == "object_count" else 12
        if not 1 <= value <= upper:
            raise ValueError(f"{predicate}.expected must be in [1, {upper}]")
        return int(value)
    expected = _FIXED_EXPECTED[predicate]
    # Equality is deliberate: the JSON string "true" must not pass as true.
    if type(value) is not type(expected) or value != expected:  # noqa: E721
        raise ValueError(f"{predicate}.expected must equal {expected!r}")
    return value


def _validate_arguments(predicate: str, arguments: Any) -> Dict[str, str]:
    if not isinstance(arguments, Mapping):
        raise ValueError(f"{predicate}.arguments must be an object")
    specification = _ARGUMENT_SPECS[predicate]
    _require_exact_keys(arguments, set(specification), f"{predicate}.arguments")
    result: Dict[str, str] = {}
    for name, allowed in specification.items():
        value = arguments[name]
        if not isinstance(value, str):
            raise ValueError(f"{predicate}.arguments.{name} must be a string")
        canonical = value.strip().lower().replace("-", "_").replace(" ", "_")
        if canonical not in allowed:
            raise ValueError(
                f"{predicate}.arguments.{name}={canonical!r}; allowed={list(allowed)}"
            )
        result[name] = canonical
    return result


def rule_semantics(rule: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the executable portion of a validated rule.

    IDs, confidence scores, and language rationales are provenance rather than
    logic, so they are intentionally omitted.
    """

    predicate = str(rule["predicate"])
    arguments = dict(rule["arguments"])
    return {
        "predicate": predicate,
        "arguments": {key: arguments[key] for key in sorted(arguments)},
        "expected": rule["expected"],
    }


def canonical_rule_key(rule: Mapping[str, Any]) -> str:
    """Create a stable, lossless key for executable rule semantics."""

    return json.dumps(
        rule_semantics(rule), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def validate_logic_candidate(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Strictly validate one per-image Qwen response.

    Unlike the older permissive proposal schemas, malformed or unknown rules
    are not silently discarded.  A response either conforms completely or is
    recorded as a failed support query by the generation script.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("Qwen payload must be a JSON object")
    _require_exact_keys(payload, _PAYLOAD_KEYS, "payload")
    summary = _validate_text(payload["normality_summary"], "normality_summary", 1000)
    rows = payload["rules"]
    if not isinstance(rows, list):
        raise ValueError("rules must be a JSON list")
    if not 1 <= len(rows) <= 20:
        raise ValueError("rules must contain between 1 and 20 candidates")

    rules: List[Dict[str, Any]] = []
    seen = set()
    for index, item in enumerate(rows):
        if not isinstance(item, Mapping):
            raise ValueError(f"rules[{index}] must be an object")
        _require_exact_keys(item, _RULE_KEYS, f"rules[{index}]")
        rule_id = _validate_text(item["id"], f"rules[{index}].id", 100)
        predicate = item["predicate"]
        if not isinstance(predicate, str):
            raise ValueError(f"rules[{index}].predicate must be a string")
        predicate = predicate.strip().lower()
        if predicate not in ALLOWED_PREDICATES:
            raise ValueError(
                f"rules[{index}].predicate={predicate!r} is outside the DSL"
            )
        arguments = _validate_arguments(predicate, item["arguments"])
        expected = _validate_expected(predicate, item["expected"])
        confidence = item["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"rules[{index}].confidence must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.05 <= confidence <= 1.0:
            raise ValueError(f"rules[{index}].confidence must be finite and in [0.05, 1]")
        reason = _validate_text(item["reason"], f"rules[{index}].reason", 500)
        rule = {
            "id": rule_id,
            "predicate": predicate,
            "arguments": arguments,
            "expected": expected,
            "confidence": confidence,
            "reason": reason,
        }
        key = canonical_rule_key(rule)
        if key in seen:
            raise ValueError(f"rules[{index}] duplicates canonical rule {key}")
        seen.add(key)
        rules.append(rule)
    return {"normality_summary": summary, "rules": rules}


def aggregate_candidate_proposals(
    proposals: Sequence[Mapping[str, Any]],
    *,
    sample_ids: Optional[Sequence[str]] = None,
    expected_support_count: Optional[int] = None,
    class_name: str = "transistor",
) -> Dict[str, Any]:
    """Aggregate independently validated proposals by canonical rule key."""

    if sample_ids is None:
        sample_ids = [f"support_{index:04d}" for index in range(len(proposals))]
    if len(sample_ids) != len(proposals):
        raise ValueError("sample_ids and proposals must have the same length")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")
    validated = [validate_logic_candidate(proposal) for proposal in proposals]
    expected_count = len(validated) if expected_support_count is None else int(expected_support_count)
    if expected_count < len(validated):
        raise ValueError("expected_support_count cannot be smaller than successful proposals")

    buckets: Dict[str, List[Tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for sample_id, proposal in zip(sample_ids, validated):
        for rule in proposal["rules"]:
            buckets[canonical_rule_key(rule)].append((sample_id, rule))

    predicate_rank = {name: index for index, name in enumerate(ALLOWED_PREDICATES)}
    candidates: List[Dict[str, Any]] = []
    denominator = len(validated)
    for key, observations in buckets.items():
        first = observations[0][1]
        confidences = [float(rule["confidence"]) for _, rule in observations]
        candidates.append({
            "canonical_key": key,
            "rule": rule_semantics(first),
            "occurrence_count": len(observations),
            "occurrence_rate_among_successes": (
                float(len(observations) / denominator) if denominator else 0.0
            ),
            "occurrence_rate_among_expected_supports": (
                float(len(observations) / expected_count) if expected_count else 0.0
            ),
            "confidence": {
                "minimum": min(confidences),
                "mean": float(sum(confidences) / len(confidences)),
                "maximum": max(confidences),
            },
            "sample_ids": [sample_id for sample_id, _ in observations],
            "source_rule_ids": [str(rule["id"]) for _, rule in observations],
            "reasons": [str(rule["reason"]) for _, rule in observations],
        })
    candidates.sort(key=lambda item: (
        -int(item["occurrence_count"]),
        predicate_rank[str(item["rule"]["predicate"])],
        str(item["canonical_key"]),
    ))
    return {
        "schema_version": SCHEMA_VERSION,
        "class_name": class_name,
        "expected_support_count": expected_count,
        "successful_support_count": len(validated),
        "complete": len(validated) == expected_count,
        "candidate_count": len(candidates),
        "candidate_semantics": (
            "candidate occurrence is proposal evidence only; a support-only stability "
            "compiler must select the final executable rules"
        ),
        "allowed_predicates": list(ALLOWED_PREDICATES),
        "normality_summaries": [proposal["normality_summary"] for proposal in validated],
        "candidates": candidates,
    }


def build_logic_candidate_prompt(class_name: str = "transistor") -> str:
    """Build the closed-vocabulary prompt for one normal support image."""

    vocabulary = {
        predicate: {name: list(values) for name, values in arguments.items()}
        for predicate, arguments in _ARGUMENT_SPECS.items()
    }
    expected_contracts = {
        "object_count": {"type": "integer", "minimum": 1, "maximum": 8},
        "part_count": {"type": "integer", "minimum": 1, "maximum": 12},
        **{
            predicate: {"type": type(value).__name__, "literal": value}
            for predicate, value in _FIXED_EXPECTED.items()
        },
    }
    return f"""
You are observing ONE known-normal image from industrial anomaly-detection class
"{class_name}". Propose visible candidate facts for a STATIC, EXECUTABLE
normality program. This query concerns this image only. A later deterministic
compiler will aggregate many independent normal-image queries and reject facts
that are not stable, so include every clearly visible candidate but never guess
an occluded or ambiguous property.

Use ONLY the typed DSL below. Each predicate has the exact argument fields and
allowed values shown. For a count predicate, `expected` is the visible integer
count inside the stated range. For every other predicate it must equal the
`literal` in EXPECTED_CONTRACTS (JSON booleans remain booleans). Natural-language
object names, numeric thresholds, defect claims, test-image information, and
predicates outside this DSL are forbidden.

ARGUMENTS:
{json.dumps(vocabulary, ensure_ascii=False, indent=2)}

EXPECTED_CONTRACTS:
{json.dumps(expected_contracts, ensure_ascii=False, indent=2)}

Operational meanings:
- object_count / part_count: exact visible instance counts.
- attached_to: every visible lead is connected to the package.
- relative_position: the lead group lies in the named coarse relation to package.
- ordered_spacing: consecutive lead centers form a consistently ordered spacing.
- mirror_symmetry: the named entity is approximately symmetric about the axis.
- similar_width / similar_orientation: all visible leads agree on that property.
- continuity: each lead (or the package outline) is visibly unbroken.
- package_shape / package_appearance / package_pose: the stated observable package
  property holds. `centered` means centered in the image, not inside itself.

Return ONLY one JSON object with exactly these fields and 1--20 unique rules:
{{
  "normality_summary": "short description of only what is visible",
  "rules": [
    {{
      "id": "short_unique_snake_case_id",
      "predicate": "part_count",
      "arguments": {{"whole": "transistor", "part": "lead"}},
      "expected": 3,
      "confidence": 0.9,
      "reason": "brief image-grounded observation"
    }}
  ]
}}

Do not duplicate equivalent rules. Confidence must be a finite number in
[0.05, 1.0]. This image is normal, but that does not make an invisible relation
true; omit anything the pixels do not support.
""".strip()


class QwenLogicCandidateProposer:
    """Reusable local Qwen3-VL proposer for independent support queries."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "auto",
        max_new_tokens: int = 1536,
        max_validation_attempts: int = 2,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_path = str(Path(model_path).expanduser().resolve())
        if not Path(self.model_path).is_dir():
            raise FileNotFoundError(f"Local Qwen model is missing: {self.model_path}")
        if device == "auto":
            target = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            target = device
        dtype = torch.bfloat16 if target.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        kwargs = {"local_files_only": True, "low_cpu_mem_usage": True}
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path, dtype=dtype, **kwargs
            )
        except TypeError:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype=dtype, **kwargs
            )
        self.model = self.model.to(target).eval()
        self.target_device = target
        self.max_new_tokens = int(max_new_tokens)
        if self.max_new_tokens < 128:
            raise ValueError("max_new_tokens must be at least 128")
        self.max_validation_attempts = int(max_validation_attempts)
        if not 1 <= self.max_validation_attempts <= 4:
            raise ValueError("max_validation_attempts must be in [1, 4]")
        self.last_raw_output = ""

    def _generate(self, image: Image.Image, prompt: str) -> str:
        import torch

        prefix = '{"normality_summary":"'
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": prefix}],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs = {
            name: value.to(model_device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        continuation = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return prefix + continuation

    def propose(
        self, image_path: Union[str, Path], class_name: str = "transistor"
    ) -> Dict[str, Any]:
        """Propose and strictly validate candidates for one support image."""

        base_prompt = build_logic_candidate_prompt(class_name)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB").copy()

        validation_error = ""
        for attempt in range(1, self.max_validation_attempts + 1):
            prompt = base_prompt
            if validation_error:
                prompt += f"""

STRICT VALIDATION FEEDBACK FROM YOUR PREVIOUS ATTEMPT:
{validation_error}
Regenerate the entire JSON object. Every rule must contain exactly these six
fields: id, predicate, arguments, expected, confidence, reason. Check every
rule, especially the final rule, before ending the JSON.
"""
            raw = self._generate(image, prompt)
            self.last_raw_output = raw
            try:
                proposal = validate_logic_candidate(extract_json_object(raw))
            except ValueError as exc:
                validation_error = str(exc)[:1000]
                if attempt == self.max_validation_attempts:
                    raise ValueError(
                        "Qwen response failed strict logic-candidate validation "
                        f"after {attempt} attempt(s): {validation_error}"
                    ) from exc
                continue
            proposal["schema_version"] = SCHEMA_VERSION
            proposal["class_name"] = class_name
            proposal["model"] = self.model_path
            proposal["validation_attempts_used"] = attempt
            proposal["raw_output"] = raw
            return proposal
        raise AssertionError("unreachable validation-attempt loop")
