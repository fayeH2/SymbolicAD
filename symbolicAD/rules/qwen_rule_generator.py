"""Qwen3-VL based rule generation with deterministic rule-bank assembly.

The LLM sees the normal support images and returns a compact structured
description (components, counts, relations, joint appearance variants).  A
deterministic builder turns that description into the executable atom DSL; the
compiler later calibrates counts and builds the hierarchy.

This module is intentionally the only place that talks to Qwen.  The rest of
the pipeline consumes cached JSON, so query-time inference never calls the LLM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schema import ComponentSymbol, LogicNode, RelationAtom, RuleKB

DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "configs" / "rules"
QWEN_PROMPT_VERSION = "v1"

_SPATIAL = {"Inside", "LeftOf", "RightOf", "Above", "Below", "CenterAligned"}
_PLACEHOLDERS = {
    "parentorempty", "camelcase", "child", "parent", "name", "component",
    "componentname", "shortrole", "shortprompt", "max2shortprompts",
}


class QwenRuleGenerationError(RuntimeError):
    """Raised when Qwen output cannot be turned into a rule bank."""


# --------------------------------------------------------------------------- #
# JSON extraction / repair
# --------------------------------------------------------------------------- #
class _SplitGroups(dict):
    """Marker produced by the duplicate-key JSON hook."""


def _object_pairs_hook(pairs):
    keys = [key for key, _ in pairs]
    if keys.count("group") > 1:
        groups: List[Dict[str, Any]] = []
        current: Dict[str, Any] = {}
        for key, value in pairs:
            if key == "group" and current:
                groups.append(current)
                current = {}
            current[key] = value
        if current:
            groups.append(current)
        return _SplitGroups(__split_groups__=groups)
    result: Dict[str, Any] = {}
    for key, value in pairs:
        result[key] = value
    return result


def _expand_split_groups(value):
    if isinstance(value, _SplitGroups):
        return [ _expand_split_groups(item) for item in value["__split_groups__"] ]
    if isinstance(value, dict):
        return {key: _expand_split_groups(item) for key, item in value.items()}
    if isinstance(value, list):
        expanded: List[Any] = []
        for item in value:
            converted = _expand_split_groups(item)
            if isinstance(converted, list) and isinstance(item, _SplitGroups):
                expanded.extend(converted)
            else:
                expanded.append(converted)
        return expanded
    return value


def _close_truncated_json(text: str) -> str:
    """Best-effort closing of a truncated JSON object."""
    in_string = False
    escape = False
    stack: List[str] = []
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]" and stack:
            stack.pop()
    if in_string:
        text += '"'
    for opener in reversed(stack):
        text += "}" if opener == "{" else "]"
    return text


def _extract_json(raw: str, prefix: str = "") -> Dict[str, Any]:
    text = prefix + raw if prefix else raw
    start = text.find("{")
    if start < 0:
        raise QwenRuleGenerationError("Qwen response contains no JSON object")
    candidate = text[start:]
    end = candidate.rfind("}")
    if end >= 0:
        candidate = candidate[: end + 1]
    try:
        parsed = json.loads(candidate, object_pairs_hook=_object_pairs_hook)
    except json.JSONDecodeError:
        candidate = _close_truncated_json(candidate)
        parsed = json.loads(candidate, object_pairs_hook=_object_pairs_hook)
    parsed = _expand_split_groups(parsed)
    if not isinstance(parsed, dict):
        raise QwenRuleGenerationError("Qwen JSON root is not an object")
    return parsed


# --------------------------------------------------------------------------- #
# Description repair
# --------------------------------------------------------------------------- #
def _sanitise_name(name: str, fallback: str) -> str:
    if str(name).strip().lower() in _PLACEHOLDERS:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", str(name).replace(" ", "_"))
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = "C" + cleaned
    return cleaned[:48]


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = [str(item) for item in value]
    else:
        values = [str(value)]
    result, seen = [], set()
    for item in values:
        item = item.strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result


def repair_description(description: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Make Qwen's structured description internally consistent."""
    components: List[Dict[str, Any]] = []
    by_name: Dict[str, Dict[str, Any]] = {}

    for index, item in enumerate(description.get("components", []) or []):
        if not isinstance(item, dict):
            continue
        name = _sanitise_name(item.get("name", ""), f"Component{index}")
        if name in by_name:
            continue
        prompts = _string_list(item.get("prompts", []))
        if not prompts:
            prompts = [f"a {name}"]
        count = item.get("count")
        count_range = item.get("count_range")
        if isinstance(count, (int, float)) and not isinstance(count, bool):
            count = int(round(float(count)))
        else:
            count = None
        if isinstance(count_range, (list, tuple)) and len(count_range) == 2:
            low, high = int(round(float(count_range[0]))), int(round(float(count_range[1])))
            count_range = [max(0, min(low, high)), max(low, high)]
        else:
            count_range = None
        entry = {
            "name": name,
            "role": str(item.get("role", "")).strip(),
            "part_of": _sanitise_name(item.get("part_of", ""), ""),
            "prompts": prompts,
            "count": count,
            "count_range": count_range,
        }
        components.append(entry)
        by_name[name] = entry

    # Add parents referenced by part_of but omitted by the model.
    for entry in list(components):
        parent = entry["part_of"]
        if not parent or parent in by_name or parent.lower() in _PLACEHOLDERS:
            if parent.lower() in _PLACEHOLDERS:
                entry["part_of"] = ""
            continue
        parent_entry = {
            "name": parent,
            "role": "container",
            "part_of": "",
            "prompts": [f"a {parent.replace('_', ' ')}", f"the {parent.replace('_', ' ')} in the scene"],
            "count": None,
            "count_range": None,
        }
        components.insert(0, parent_entry)
        by_name[parent] = parent_entry
    for entry in components:
        if entry["part_of"] and entry["part_of"] not in by_name:
            entry["part_of"] = ""

    # Relations: keep only valid spatial relations among known components.
    valid_containment = {
        (entry["name"], entry["part_of"])
        for entry in components
        if entry.get("part_of") and entry["part_of"] in by_name
    }
    relations: List[Dict[str, str]] = []
    seen_relations = set()
    for item in description.get("relations", []) or []:
        if not isinstance(item, dict):
            continue
        relation = str(item.get("type", item.get("relation", ""))).strip()
        a = _sanitise_name(item.get("a", ""), "")
        b = _sanitise_name(item.get("b", ""), "")
        if relation not in _SPATIAL or a not in by_name or b not in by_name or a == b:
            continue
        if relation == "Inside" and (a, b) not in valid_containment and (b, a) in valid_containment:
            # The model emitted the inverse containment; the builder already
            # adds the correct Inside(child, parent) from part_of.
            continue
        key = (relation, a, b)
        if key in seen_relations:
            continue
        seen_relations.add(key)
        relations.append({"type": relation, "a": a, "b": b})

    # Attribute variants: flatten duplicate keys and rename collisions.
    variants: List[Dict[str, Any]] = []
    seen_variants = set()
    for group_index, item in enumerate(description.get("attribute_variants", []) or []):
        if len(variants) >= 6:
            break
        if not isinstance(item, dict):
            continue
        group = str(item.get("group", f"variant{group_index}")).strip() or f"variant{group_index}"
        conditions = []
        for condition in item.get("conditions", []) or []:
            if not isinstance(condition, dict):
                continue
            component = _sanitise_name(condition.get("component", ""), "")
            attribute = str(condition.get("attribute", "")).strip()
            value = str(condition.get("value", "")).strip()
            prompts = _string_list(condition.get("prompts", []))
            if (
                component not in by_name
                or not attribute
                or not value
                or attribute.lower() in _PLACEHOLDERS
                or value.lower() in _PLACEHOLDERS
            ):
                continue
            if not prompts:
                prompts = [f"a {value} {attribute}", f"{value}"]
            conditions.append(
                {
                    "component": component,
                    "attribute": attribute,
                    "value": value,
                    "prompts": prompts,
                }
            )
        if not conditions:
            continue
        key = frozenset(
            (condition["component"], condition["attribute"], condition["value"])
            for condition in conditions
        )
        if key in seen_variants:
            continue
        seen_variants.add(key)
        variants.append(
            {
                "group": f"{group}_{len(variants)}",
                "conditions": conditions,
            }
        )

    return {
        "category": category,
        "components": components,
        "relations": relations,
        "attribute_variants": variants,
    }


# --------------------------------------------------------------------------- #
# Description -> executable RuleKB
# --------------------------------------------------------------------------- #
def build_rule_kb_from_description(description: Dict[str, Any], category: str) -> RuleKB:
    description = repair_description(description, category)
    components = [
        ComponentSymbol(
            name=item["name"],
            prompts=item["prompts"],
            role=item.get("role", ""),
            part_of=item.get("part_of", ""),
        )
        for item in description["components"]
    ]
    component_names = {item.name for item in components}
    atoms: List[RelationAtom] = []
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"R{counter}"

    for item in description["components"]:
        name = item["name"]
        count = item.get("count")
        count_range = item.get("count_range")
        if count_range is not None:
            low, high = count_range
            atoms.append(
                RelationAtom(
                    atom_id=next_id(),
                    expression=f"CountInRange({name},{low},{high})",
                    kind="CountInRange",
                    args=[name, low, high],
                    normal_prompt=f"between {low} and {high} {name}",
                    violation_prompt=f"wrong number of {name}",
                )
            )
        elif isinstance(count, int) and count > 0:
            atoms.append(
                RelationAtom(
                    atom_id=next_id(),
                    expression=f"CountEq({name},{count})",
                    kind="CountEq",
                    args=[name, count],
                    normal_prompt=f"exactly {count} {name}",
                    violation_prompt=f"wrong number of {name}",
                )
            )

    # Structural containment inferred from part_of.
    contained = set()
    for item in description["components"]:
        parent = item.get("part_of")
        if parent and parent in component_names:
            key = (item["name"], parent)
            if key in contained:
                continue
            contained.add(key)
            atoms.append(
                RelationAtom(
                    atom_id=next_id(),
                    expression=f"Inside({item['name']},{parent})",
                    kind="SpatialRelation",
                    args=[item["name"], parent],
                    params={"relation": "Inside"},
                    normal_prompt=f"{item['name']} inside {parent}",
                    violation_prompt=f"{item['name']} outside {parent}",
                )
            )

    for relation in description["relations"]:
        expression = f"{relation['type']}({relation['a']},{relation['b']})"
        atoms.append(
            RelationAtom(
                atom_id=next_id(),
                expression=expression,
                kind="SpatialRelation",
                args=[relation["a"], relation["b"]],
                params={"relation": relation["type"]},
                normal_prompt=f"{relation['a']} {relation['type']} {relation['b']}",
                violation_prompt=f"{relation['a']} not {relation['type']} {relation['b']}",
            )
        )

    for variant in description["attribute_variants"]:
        for condition in variant["conditions"]:
            expression = (
                f"AttributeMatch({condition['component']},"
                f"{condition['attribute']},{condition['value']})"
            )
            atoms.append(
                RelationAtom(
                    atom_id=next_id(),
                    expression=expression,
                    kind="AttributeMatch",
                    args=[
                        condition["component"],
                        condition["attribute"],
                        condition["value"],
                    ],
                    params={
                        "group": variant["group"],
                        "attribute_prompts": condition["prompts"],
                    },
                    normal_prompt=f"{condition['component']} with {condition['attribute']} {condition['value']}",
                    violation_prompt=f"{condition['component']} with wrong {condition['attribute']}",
                )
            )

    if not atoms:
        raise QwenRuleGenerationError("Qwen description produced no executable atoms")

    formula = LogicNode(
        op="AND",
        children=[LogicNode.leaf(atom.atom_id) for atom in atoms],
    )
    return RuleKB(
        category=category,
        components=components,
        atoms=atoms,
        formula=formula,
        metadata={
            "source": "qwen3-vl-8b-thinking",
            "prompt_version": QWEN_PROMPT_VERSION,
            "description": description,
        },
    )


# --------------------------------------------------------------------------- #
# Qwen backend
# --------------------------------------------------------------------------- #
def _rule_generation_prompt(category: str, image_count: int) -> str:
    return f"""Inspect these {image_count} normal support images of ONE product category and describe its normal structure as compact JSON.

Product category: {category}

Output ONE JSON object with exactly these keys: category, components, relations, attribute_variants.

Schema:
{{
 "category": "{category}",
 "components": [
   {{"name":"CamelCase","role":"short role","part_of":"ParentOrEmpty","prompts":["max 2 short prompts"],"count":1,"count_range":null}}
 ],
 "relations": [
   {{"type":"Inside|Above|Below|LeftOf|RightOf|CenterAligned","a":"Child","b":"Parent"}}
 ],
 "attribute_variants": [
   {{"group":"variant","conditions":[
     {{"component":"Name","attribute":"fruit","value":"orange","prompts":["max 2 prompts"]}},
     {{"component":"Name","attribute":"color","value":"yellow","prompts":["max 2 prompts"]}}
   ]}}
 ]
}}

Strict requirements:
- List every important visible part, including the main container, closures and labels.
- Separate visually distinct parts with different roles into separate components.
- Separate a large picture/fruit label from a small text label into two components.
- Counts must be visible integers. Use count_range [low, high] only when the count really varies.
- Relations must include containment (Inside child parent) and the vertical order of the two labels. In image coordinates, the component higher up is Above the lower component. Include each relation only once.
- attribute_variants: ONLY for valid normal appearance combinations (e.g. fruit type and liquid colour). The support images may show several valid variants; enumerate ALL distinct fruit/colour combinations you can see (for example cherry+red, banana+white, orange+yellow). At most 4 unique entries. Each entry MUST be a separate object in the array; never repeat the same variant and never repeat the group key inside one object. Put fruit attributes on the fruit-label component and colour attributes on the Liquid component. Do NOT add texture, clarity, material, background, border, font, shape, size or position attributes.
- Never copy schema placeholders (ParentOrEmpty, CamelCase, Child, Parent, Name, ComponentName, short role, max 2 prompts) into the output.
- Do not describe scratches, dirt, contamination or breakage.
- Keep prompts short.
- Valid JSON only, no markdown, no extra keys. Output the JSON immediately."""


def _resize_support(images: Sequence, max_side: int) -> List:
    from PIL import Image

    resized = []
    for image in images:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        resized.append(image)
    return resized


class QwenVLBackend:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:1",
        max_image_side: int = 768,
        max_new_tokens: int = 3000,
    ):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.device = device
        self.max_image_side = int(max_image_side)
        self.max_new_tokens = int(max_new_tokens)
        self.model_path = str(model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

    def generate(self, category: str, support_images: Sequence) -> str:
        images = _resize_support(support_images, self.max_image_side)
        prompt_text = _rule_generation_prompt(category, len(images))
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": prompt_text}],
            },
            {
                "role": "assistant",
                "content": '{"category":"' + category + '","components":[',
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        inputs = self.processor(
            text=[prompt], images=images, return_tensors="pt"
        ).to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return '{"category":"' + category + '","components":[' + raw


_BACKENDS: Dict[Tuple[str, str, int, int], QwenVLBackend] = {}


def get_backend(config) -> QwenVLBackend:
    qwen = config.qwen
    key = (
        str(qwen.model_path),
        str(qwen.device),
        int(qwen.max_image_side),
        int(qwen.max_new_tokens),
    )
    if key not in _BACKENDS:
        _BACKENDS[key] = QwenVLBackend(
            model_path=qwen.model_path,
            device=qwen.device,
            max_image_side=qwen.max_image_side,
            max_new_tokens=qwen.max_new_tokens,
        )
    return _BACKENDS[key]


def generate_rules_with_qwen(
    category: str,
    support_images: Sequence,
    config,
    force: bool = False,
) -> RuleKB:
    cache_dir = config.cache_dir / config.qwen.cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    shot = int(getattr(config, "shot", 0))
    stem = f"{category}_k{shot}"
    rule_path = cache_dir / f"{stem}.json"
    raw_path = cache_dir / f"{stem}.raw.txt"
    description_path = cache_dir / f"{stem}.description.json"

    if rule_path.is_file() and not force:
        return RuleKB.load(rule_path)

    backend = get_backend(config)
    raw = backend.generate(category, support_images)
    raw_path.write_text(raw, encoding="utf-8")
    try:
        description = _extract_json(raw)
        rule_kb = build_rule_kb_from_description(description, category)
    except Exception as exc:
        raise QwenRuleGenerationError(
            f"Qwen rule generation failed for {category}; raw response saved to {raw_path}: {exc}"
        ) from exc

    rule_kb.metadata["support_images"] = [str(path) for path in support_images]
    rule_kb.metadata["support_shot"] = shot
    rule_kb.metadata["raw_response_path"] = str(raw_path)
    description_path.write_text(
        json.dumps(rule_kb.metadata["description"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rule_kb.save(rule_path)
    return rule_kb


# --------------------------------------------------------------------------- #
# Public loader (Qwen first, template fallback only if explicitly allowed)
# --------------------------------------------------------------------------- #
def cache_path(category: str, rule_cache_dir: str | Path) -> Path:
    """Return the JSON path inside an already-resolved rule cache directory."""
    return Path(rule_cache_dir) / f"{category}.json"


def load_or_generate_rules(
    category: str,
    cache_dir: str | Path,
    template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
    force: bool = False,
    support_images: Optional[Sequence] = None,
    config=None,
    allow_template_fallback: Optional[bool] = None,
) -> RuleKB:
    qwen_enabled = bool(
        config is not None
        and getattr(config, "qwen", None) is not None
        and config.qwen.enabled
        and support_images
    )
    if qwen_enabled:
        try:
            return generate_rules_with_qwen(
                category, support_images, config, force=force
            )
        except Exception as exc:
            fallback = (
                config.qwen.allow_template_fallback
                if allow_template_fallback is None
                else allow_template_fallback
            )
            if not fallback:
                raise
            print(f"[SymbolicAD] Qwen rule generation failed: {exc}; using template fallback")

    cached = cache_path(category, cache_dir)
    if cached.is_file() and not force:
        return RuleKB.load(cached)
    template = Path(template_dir) / f"{category}.json"
    if template.is_file():
        rule_kb = RuleKB.load(template)
    else:
        rule_kb = _fallback_rule_kb(category)
    rule_kb.category = rule_kb.category or category
    rule_kb.save(cached)
    return rule_kb


def _fallback_rule_kb(category: str) -> RuleKB:
    phrase = category.replace("_", " ")
    payload = {
        "category": category,
        "components": [
            {"name": category, "prompts": [f"a photo of a {phrase}"], "synonyms": []}
        ],
        "atoms": [
            {
                "id": "R1",
                "expression": f"Exist({category})",
                "normal_prompt": f"a normal {phrase}",
                "violation_prompt": f"a missing {phrase}",
            }
        ],
        "formula": {"op": "AND", "children": ["R1"]},
    }
    return RuleKB.from_dict(payload)


def write_template(
    category: str, payload: dict, template_dir: str | Path = DEFAULT_TEMPLATE_DIR
) -> Path:
    template_dir = Path(template_dir)
    template_dir.mkdir(parents=True, exist_ok=True)
    path = template_dir / f"{category}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path
