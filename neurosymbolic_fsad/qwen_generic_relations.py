"""Class-agnostic Qwen proposal schema for grounded normality candidates.

Qwen is used only during library construction.  It proposes a *candidate
pool*, never an executable answer.  A deterministic support-fitted compiler
later rejects ungrounded candidates, removes intervention-redundant rules and
searches a logic program.  This separation prevents fluent language variants
from being mistaken for independent symbolic evidence.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

import numpy as np
import torch
from PIL import Image

from .qwen_rules import DEFAULT_MODEL_PATH, extract_json_object


ALLOWED_ARRANGEMENTS = {"single", "row", "column", "grid", "ring", "free"}
ALLOWED_PREDICATE_ORDER = (
    "count_eq",
    "arrangement_eq",
    "similar_size",
    "similar_orientation",
    "ordered_spacing",
    "appearance_consistency",
)
ALLOWED_PREDICATES = set(ALLOWED_PREDICATE_ORDER)

PREDICATE_FAMILIES = {
    "count_eq": "cardinality",
    "arrangement_eq": "spatial_configuration",
    "ordered_spacing": "spatial_configuration",
    "similar_size": "scale_geometry",
    "similar_orientation": "orientation_geometry",
    "appearance_consistency": "appearance",
}


def build_generic_relational_prompt(class_name: str, support_count: int) -> str:
    panel_instruction = (
        "The attachment is ONE image. Count entities inside this image."
        if support_count == 1 else
        "The attachment is a 2-by-2 contact sheet of FOUR SEPARATE images. "
        "Count independently inside each panel."
    )
    return f"""
You are inducing a STATIC NORMALITY PROGRAM from exactly {support_count}
known-normal images of visual anomaly-detection class "{class_name}".  The
image may come from an industrial or medical benchmark; use only visible
evidence and do not assume domain-specific anatomy or component names.
{panel_instruction}

Find at most ONE repeated, visually separable foreground entity type that a
classical connected-component detector could ground without text at test time.
Examples are repeated product pieces, pins, teeth, holes, caps, or modules.
When one product contains two or more visible repeated subparts, choose those
subparts (for example three terminal pins), NOT the whole product with
expected_count=1.  A single whole object is acceptable only when it has no
separable repeated subparts.
Do not choose texture cells, printed letters, highlights, shadows, or background
patterns.  If there is no such invariant entity, set primary_entity to null and
return an empty rules list.  This abstention is important.

CRITICAL COUNT RULE: black tile boundaries and panel labels are not scene
content. `expected_count` is always a within-image count. Never use the number
of supplied support images as expected_count.

For a valid entity, use a lowercase snake_case entity id and a concrete
observable visual description. Do not invent thresholds. Propose 1--5
candidate normality rules, but only when the corresponding property is visibly
stable in every support image. Only use these predicates:
count_eq, arrangement_eq, similar_size, similar_orientation, ordered_spacing,
appearance_consistency.  arrangement must be one of:
single, row, column, grid, ring, free.

ORTHOGONALITY REQUIREMENT: each candidate should cover a different failure
axis. Do not rephrase the same rule. In particular arrangement_eq and
ordered_spacing are alternative spatial-configuration candidates; emit at
most one of them. Count, position, size/proportion, orientation and appearance
are different axes. Do not emit count_eq merely because an entity exists, and
do not emit arrangement_eq for a single entity unless its absolute position is
visibly fixed. The downstream compiler may still reject any candidate.

Return ONLY one JSON object:
{{
  "normality_summary": "short visible description",
  "primary_entity": {{
    "id": "entity_name",
    "expected_count": 4,
    "arrangement": "grid",
    "visual_description": "observable color/shape cues"
  }},
  "rules": [
    {{"id": "unique_id", "predicate": "count_eq",
      "importance": 0.9, "confidence": 0.9,
      "reason": "invariant visible in all supports"}}
  ]
}}

Never refer to defects or test images.  Never claim an invariant that is not
visible in every support panel.
""".strip()


def _probability(value: Any, default: float = 0.8) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not np.isfinite(number):
        number = default
    return float(np.clip(number, 0.05, 1.0))


def validate_generic_relational_proposal(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = str(payload.get("normality_summary", ""))[:1000]
    entity = payload.get("primary_entity")
    if not isinstance(entity, Mapping):
        return {"normality_summary": summary, "primary_entity": None, "rules": []}
    entity_id = str(entity.get("id", "")).strip().lower().replace(" ", "_")
    expected = entity.get("expected_count")
    arrangement = str(entity.get("arrangement", "free")).strip().lower()
    if (
        not entity_id
        or not isinstance(expected, (int, float))
        or int(expected) < 1
        or int(expected) > 32
        or arrangement not in ALLOWED_ARRANGEMENTS
    ):
        return {"normality_summary": summary, "primary_entity": None, "rules": []}
    visual_description = str(entity.get("visual_description", ""))[:500]
    semantic_text = f"{entity_id} {visual_description}".lower().replace("_", " ")
    # The generic connected-component compiler cannot ground texture primitives.
    # Rejecting them here prevents a parameter search from accidentally finding
    # an arbitrary set of four blobs that appears to confirm a hallucinated rule.
    banned_texture_terms = (
        "texture", "grain", "speckle", "mesh cell", "surface pattern",
        "woven thread", "weave cell", "background pattern", "repeating grid",
        "repeating pattern", "mesh opening",
    )
    if any(term in semantic_text for term in banned_texture_terms):
        return {"normality_summary": summary, "primary_entity": None, "rules": []}
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    part_nouns = r"segments?|parts?|pins?|leads?|cores?|pieces?|objects?|components?|arms?|teeth|holes?"
    mentioned_counts = []
    for word, number in number_words.items():
        if re.search(rf"\b{word}\b(?:\s+\w+){{0,3}}\s+(?:{part_nouns})\b", semantic_text):
            mentioned_counts.append(number)
    for match in re.finditer(rf"\b(\d+)\b(?:\s+\w+){{0,3}}\s+(?:{part_nouns})\b", semantic_text):
        mentioned_counts.append(int(match.group(1)))
    if mentioned_counts and int(expected) not in mentioned_counts:
        return {"normality_summary": summary, "primary_entity": None, "rules": []}
    canonical_entity = {
        "id": entity_id[:100],
        "expected_count": int(expected),
        "arrangement": arrangement,
        "visual_description": visual_description,
    }
    rules = []
    seen = set()
    rows = payload.get("rules", [])
    if not isinstance(rows, list):
        rows = []
    for index, item in enumerate(rows):
        if not isinstance(item, Mapping):
            continue
        predicate = str(item.get("predicate", "")).strip().lower()
        if predicate not in ALLOWED_PREDICATES or predicate in seen:
            continue
        seen.add(predicate)
        rules.append({
            "id": str(item.get("id") or f"generic_rel_{index + 1}")[:100],
            "predicate": predicate,
            "importance": _probability(item.get("importance")),
            "confidence": _probability(item.get("confidence")),
            "reason": str(item.get("reason", ""))[:500],
            "family": PREDICATE_FAMILIES[predicate],
        })
    return {
        "normality_summary": summary,
        "primary_entity": canonical_entity,
        "rules": rules,
    }


def consensus_generic_relational_proposals(
    proposals: Sequence[Mapping[str, Any]],
    *,
    minimum_fraction: float = 0.75,
) -> Dict[str, Any]:
    """Build a support-consensus candidate pool without forcing any rule.

    Entity wording is deliberately not used as a hard vote: synonymous names
    from separate LLM calls are common, while the deterministic visual compiler
    is responsible for grounding identity. Count and rule presence must reach
    support consensus. No query image or anomaly label is used.
    """
    canonical = [validate_generic_relational_proposal(item) for item in proposals]
    support_count = len(canonical)
    required = max(1, int(np.ceil(float(minimum_fraction) * support_count)))
    valid = [
        item for item in canonical
        if isinstance(item.get("primary_entity"), Mapping)
    ]
    protocol = "independent_support_orthogonal_candidate_consensus"
    if not valid:
        return {
            "normality_summary": "abstained: no independently grounded entity proposal",
            "primary_entity": None,
            "rules": [],
            "consensus_protocol": protocol,
            "required_support_agreement": required,
            "per_support_proposals": canonical,
        }
    count_votes = Counter(
        int(item["primary_entity"]["expected_count"]) for item in valid
    )
    winning_count, count_agreement = count_votes.most_common(1)[0]
    agreeing = [
        item for item in valid
        if int(item["primary_entity"]["expected_count"]) == winning_count
    ]
    if count_agreement < required:
        return {
            "normality_summary": "abstained: support proposals disagree on entity count",
            "primary_entity": None,
            "rules": [],
            "consensus_protocol": protocol,
            "required_support_agreement": required,
            "count_votes": dict(count_votes),
            "per_support_proposals": canonical,
        }
    arrangement_votes = Counter(
        str(item["primary_entity"]["arrangement"]) for item in agreeing
    )
    arrangement, arrangement_agreement = arrangement_votes.most_common(1)[0]
    if arrangement_agreement < required:
        arrangement = "free"
    predicate_votes = Counter(
        predicate
        for item in agreeing
        for predicate in {row["predicate"] for row in item.get("rules", [])}
    )
    accepted = {
        predicate for predicate, votes in predicate_votes.items()
        if votes >= required
    }
    source_rows = {
        row["predicate"]: row
        for item in agreeing
        for row in item.get("rules", [])
        if row["predicate"] in accepted
    }
    rules = [
        source_rows[predicate] for predicate in ALLOWED_PREDICATE_ORDER
        if predicate in source_rows
    ]
    source = agreeing[0]
    primary = dict(source["primary_entity"])
    primary["expected_count"] = int(winning_count)
    primary["arrangement"] = arrangement
    return {
        "normality_summary": source.get("normality_summary", ""),
        "primary_entity": primary,
        "rules": rules,
        "consensus_protocol": protocol,
        "required_support_agreement": required,
        "count_votes": dict(count_votes),
        "predicate_support_votes": dict(predicate_votes),
        "per_support_proposals": canonical,
    }


class QwenGenericRelationalProposer:
    """One reusable local model instance for many dataset categories."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "auto",
        max_new_tokens: int = 1024,
    ):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_path = str(Path(model_path).expanduser().resolve())
        target = device
        if target == "auto":
            target = "cuda:1" if torch.cuda.device_count() > 1 else (
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        dtype = torch.bfloat16 if target.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
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

    def propose(self, class_name: str, support_count: int, montage_path: Union[str, Path]) -> Dict[str, Any]:
        prompt = build_generic_relational_prompt(class_name, support_count)
        prefix = '{"normality_summary":"'
        with Image.open(montage_path) as handle:
            image = handle.convert("RGB")
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": prefix}]},
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
        inputs = {key: value.to(model_device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        continuation = self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        raw = prefix + continuation
        proposal = validate_generic_relational_proposal(extract_json_object(raw))
        proposal["raw_output"] = raw
        proposal["model"] = self.model_path
        return proposal
