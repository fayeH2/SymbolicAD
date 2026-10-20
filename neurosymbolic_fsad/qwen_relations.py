"""Offline Qwen3-VL discovery of executable entity--relation rules.

The model is used only while building a rule library from known-normal support
images.  Runtime inference consumes the compiled JSON library and never loads
the LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

import numpy as np
import torch
from PIL import Image, ImageDraw

from .qwen_rules import DEFAULT_MODEL_PATH, extract_json_object


ALLOWED_ENTITIES = {"outer_sheath", "conductor", "copper_core"}
ALLOWED_PREDICATES = {
    "count_eq",
    "all_inside",
    "one_to_one_inside",
    "distinct_attribute_count_eq",
    "attribute_multiset_eq",
    "triangular_layout",
    "similar_size",
    "centered_group",
    "relative_position",
}
ALLOWED_IDENTITIES = {"blue", "brown", "yellow_green"}
ALLOWED_RELATIONS = {"above", "below", "left_of", "right_of"}


def make_rgb_support_montage(
    image_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    tile_size: int = 336,
) -> Path:
    """Create the RGB-only full-image view used by the relational grounder."""
    if not image_paths:
        raise ValueError("At least one support image is required")
    columns = 2
    rows = int(np.ceil(len(image_paths) / columns))
    label_height = 30
    canvas = Image.new(
        "RGB", (columns * tile_size, rows * (tile_size + label_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(image_paths):
        with Image.open(path) as handle:
            aligned = handle.convert("RGB")
        aligned.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (tile_size, tile_size), "black")
        panel.paste(
            aligned,
            ((tile_size - aligned.width) // 2, (tile_size - aligned.height) // 2),
        )
        row, column = divmod(index, columns)
        left, top = column * tile_size, row * (tile_size + label_height)
        canvas.paste(panel, (left, top))
        draw.text((left + 6, top + tile_size + 6), f"known-normal support {index + 1}", fill="black")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)
    return output


def build_relational_prompt(class_name: str, support_count: int) -> str:
    vocabulary = {
        "entities": sorted(ALLOWED_ENTITIES),
        "predicates": {
            "count_eq": "an entity type has an exact normal count",
            "all_inside": "every subject lies inside the object container",
            "one_to_one_inside": "each object/part has exactly one matched contained part",
            "distinct_attribute_count_eq": "the set has an exact number of distinct identities",
            "attribute_multiset_eq": "normal samples contain exactly the listed identity multiset",
            "triangular_layout": "three parts form a non-degenerate triangular arrangement",
            "similar_size": "instances of one entity type have comparable size",
            "centered_group": "the part group is centered within its container",
            "relative_position": "one identity has a stable direction relative to another",
        },
        "canonical_color_identities": sorted(ALLOWED_IDENTITIES),
        "relative_relations": sorted(ALLOWED_RELATIONS),
    }
    return f"""
You are inducing a STATIC NORMALITY PROGRAM for few-shot industrial anomaly detection.
The attached montage contains exactly {support_count} known-normal images of class
"{class_name}".  Inspect the visible objects and recover entities, counts, attributes,
and relations that are invariant across these normal examples.

This is not a request for vague appearance adjectives or anomaly-map statistics.
Prefer executable compositional facts such as exact part count, distinct identity
count, expected identity multiset, containment, one-to-one correspondence, relative
position, layout, and size ratios.  State only relations visibly supported by every
normal image.  Do not invent numeric thresholds; a support-only verifier will learn
them.  Do not mention defects or test images.

Use only this executable vocabulary:
{json.dumps(vocabulary, ensure_ascii=False, indent=2)}

Return ONLY one JSON object with this exact shape:
{{
  "normality_summary": "concise compositional description",
  "entities": [
    {{"id": "one allowed entity", "expected_count": 1,
      "visual_description": "observable grounding cues"}}
  ],
  "rules": [
    {{"id": "unique_rule_id", "predicate": "one allowed predicate",
      "subject": "entity or canonical color identity", "object": null,
      "attribute": null, "relation": null, "target": null,
      "values": [], "importance": 0.9, "confidence": 0.9,
      "reason": "short image-grounded rationale"}}
  ]
}}

Field conventions:
- count_eq: subject=entity, target=integer.
- all_inside/one_to_one_inside: subject and object are allowed entities.
- distinct_attribute_count_eq: subject=conductor, attribute=color_identity,
  target=integer.
- attribute_multiset_eq: subject=conductor, attribute=color_identity, values are
  canonical color identities.
- triangular_layout/similar_size/centered_group: subject=conductor.
- relative_position: subject and object are canonical color identities and relation
  is above/below/left_of/right_of.

Generate every clearly visible invariant, but never duplicate a predicate with the
same arguments.
""".strip()


def _canonical_identity(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "yellow": "yellow_green",
        "green": "yellow_green",
        "yellowgreen": "yellow_green",
        "yellow_green": "yellow_green",
        "blue": "blue",
        "brown": "brown",
    }
    return aliases.get(text, text)


def _finite_probability(value: Any, default: float = 0.8) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not np.isfinite(number):
        number = default
    return float(np.clip(number, 0.05, 1.0))


def validate_relational_proposal(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Strictly validate and canonicalize the LLM program before compilation."""
    entity_rows = payload.get("entities", [])
    if not isinstance(entity_rows, list):
        entity_rows = []
    entities = []
    seen_entities = set()
    for item in entity_rows:
        if not isinstance(item, Mapping):
            continue
        entity_id = str(item.get("id", "")).strip().lower()
        if entity_id not in ALLOWED_ENTITIES or entity_id in seen_entities:
            continue
        seen_entities.add(entity_id)
        expected = item.get("expected_count")
        entities.append({
            "id": entity_id,
            "expected_count": int(expected) if isinstance(expected, (int, float)) else None,
            "visual_description": str(item.get("visual_description", ""))[:500],
        })

    rule_rows = payload.get("rules", [])
    if not isinstance(rule_rows, list):
        raise ValueError("Qwen relational JSON must contain a rules list")
    rules = []
    seen = set()
    for index, item in enumerate(rule_rows):
        if not isinstance(item, Mapping):
            continue
        predicate = str(item.get("predicate", "")).strip().lower()
        if predicate not in ALLOWED_PREDICATES:
            continue
        subject = str(item.get("subject", "")).strip().lower()
        object_value = item.get("object")
        object_name = None if object_value is None else str(object_value).strip().lower()
        attribute = item.get("attribute")
        attribute = None if attribute is None else str(attribute).strip().lower()
        relation = item.get("relation")
        relation = None if relation is None else str(relation).strip().lower()
        target = item.get("target")
        values = item.get("values", [])
        values = values if isinstance(values, list) else []

        if predicate in {"count_eq", "all_inside", "one_to_one_inside",
                         "distinct_attribute_count_eq", "attribute_multiset_eq",
                         "triangular_layout", "similar_size", "centered_group"}:
            if subject not in ALLOWED_ENTITIES:
                continue
        if predicate in {"all_inside", "one_to_one_inside"} and object_name not in ALLOWED_ENTITIES:
            continue
        if predicate in {"distinct_attribute_count_eq", "attribute_multiset_eq"}:
            if subject != "conductor" or attribute != "color_identity":
                continue
        if predicate == "attribute_multiset_eq":
            values = [_canonical_identity(value) for value in values]
            if not values or any(value not in ALLOWED_IDENTITIES for value in values):
                continue
            values = sorted(values)
        if predicate == "relative_position":
            subject = _canonical_identity(subject)
            object_name = _canonical_identity(object_name)
            if subject not in ALLOWED_IDENTITIES or object_name not in ALLOWED_IDENTITIES:
                continue
            if relation not in ALLOWED_RELATIONS:
                continue
        if predicate in {"count_eq", "distinct_attribute_count_eq"}:
            if not isinstance(target, (int, float)):
                continue
            target = int(target)

        key = (predicate, subject, object_name, attribute, relation,
               target, tuple(values))
        if key in seen:
            continue
        seen.add(key)
        rules.append({
            "id": str(item.get("id") or f"rel_{index + 1}")[:100],
            "predicate": predicate,
            "subject": subject,
            "object": object_name,
            "attribute": attribute,
            "relation": relation,
            "target": target,
            "values": values,
            "importance": _finite_probability(item.get("importance", 0.8)),
            "confidence": _finite_probability(item.get("confidence", 0.8)),
            "reason": str(item.get("reason", ""))[:500],
        })
    if not rules:
        raise ValueError("Qwen proposed no executable relational rules")
    return {
        "normality_summary": str(payload.get("normality_summary", ""))[:1200],
        "entities": entities,
        "rules": rules,
    }


class QwenRelationalProposer:
    """Run the local Qwen3-VL model once during offline rule induction."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "auto",
        max_new_tokens: int = 2048,
    ):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_path = str(Path(model_path).expanduser())
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        if device == "auto":
            target = "cuda:1" if torch.cuda.device_count() > 1 else (
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            target = device
        dtype = torch.bfloat16 if target.startswith("cuda") else torch.float32
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
        self.max_new_tokens = max_new_tokens
        self.last_raw_output = ""

    def propose(
        self,
        class_name: str,
        support_count: int,
        montage_path: Union[str, Path],
    ) -> Dict[str, Any]:
        prompt = build_relational_prompt(class_name, support_count)
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
        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        generated = ids[:, inputs["input_ids"].shape[1]:]
        continuation = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        raw = prefix + continuation
        self.last_raw_output = raw
        proposal = validate_relational_proposal(extract_json_object(raw))
        proposal["raw_output"] = raw
        return proposal
