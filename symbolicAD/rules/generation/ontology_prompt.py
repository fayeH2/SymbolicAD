"""Build the Qwen rule-generation prompt from the ontology and predicate registry.

The prompt is intentionally derived from executable code, not maintained as a
separate hand-written capability list.  If an executor is not registered, its
predicate is not advertised to Qwen.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from .dsl import QWEN_RULE_GENERATION_PROMPT
from .ontology import CategoryOntology, ontology_for_category
from ..rule_registry import default_rule_registry


# Qwen-facing predicate contract.  ``typed_predicate`` is the exact registry
# capability that the adapter must resolve after generation.
_PREDICATE_CONTRACTS: Sequence[Dict[str, Any]] = (
    {
        "predicate_name": "COUNT",
        "typed_predicate": "count_eq",
        "arguments": {"entity_type": "<canonical instance id>"},
        "parameters": {"expected_count": "<non-negative integer>",
                       "tolerance": "<non-negative integer>"},
        "when_to_use": "global count of instance-granularity entities",
    },
    {
        "predicate_name": "COUNT",
        "typed_predicate": "count_in",
        "arguments": {"entity_type": "<canonical id>",
                       "target_entity_type": "<canonical region/container id>"},
        "parameters": {"expected_count": "<non-negative integer>",
                       "tolerance": "<non-negative integer>"},
        "when_to_use": "count of entities observed inside one named region/container",
    },
    {
        "predicate_name": "EXISTS",
        "typed_predicate": "exists",
        "arguments": {"entity_type": "<canonical id>",
                       "target_entity_type": "<optional canonical region id>"},
        "parameters": {"minimum_count": 1},
        "when_to_use": "presence, especially for region-granularity objects",
    },
    {
        "predicate_name": "LEFT_OF",
        "typed_predicate": "left_of",
        "arguments": {"entity_type": "<canonical source id>",
                       "target_entity_type": "<canonical target id>"},
        "parameters": {"threshold": "<non-negative normalized margin>"},
        "when_to_use": "directional horizontal relation",
    },
    {
        "predicate_name": "ABOVE",
        "typed_predicate": "above",
        "arguments": {"entity_type": "<canonical source id>",
                       "target_entity_type": "<canonical target id>"},
        "parameters": {"threshold": "<non-negative normalized margin>"},
        "when_to_use": "directional vertical relation",
    },
    {
        "predicate_name": "ALIGNED_VERTICAL",
        "typed_predicate": "aligned_vertical",
        "arguments": {"entity_type": "<canonical source id>",
                       "target_entity_type": "<canonical target id>"},
        "parameters": {"threshold": "<non-negative normalized x-spread>"},
        "when_to_use": "two reliably grounded entities share a vertical column",
    },
    {
        "predicate_name": "DISJOINT",
        "typed_predicate": "disjoint",
        "arguments": {"entity_type": "<canonical source id>",
                       "target_entity_type": "<canonical target id>"},
        "parameters": {"threshold": "<non-negative overlap tolerance>"},
        "when_to_use": "two instance masks must not overlap",
    },
    {
        "predicate_name": "LENGTH",
        "typed_predicate": "length_range",
        "arguments": {"entity_type": "<canonical id>"},
        "parameters": {"expected_value": "<number>", "tolerance": "<non-negative number>"},
        "when_to_use": "measured PCA-axis length is stable in normal supports",
    },
    {
        "predicate_name": "ASPECT_RATIO",
        "typed_predicate": "aspect_range",
        "arguments": {"entity_type": "<canonical id>"},
        "parameters": {"expected_value": "<number>", "tolerance": "<non-negative number>"},
        "when_to_use": "shape aspect ratio is stable in normal supports",
    },
    {
        "predicate_name": "COLOR_H",
        "typed_predicate": "color_range",
        "arguments": {"entity_type": "<canonical id>"},
        "parameters": {"expected_value": "<number>", "tolerance": "<non-negative number>"},
        "when_to_use": "hue is stable in normal supports",
    },
)


def predicate_contracts(
    registered_predicates: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Return only contracts whose exact typed predicate is registered."""
    if registered_predicates is None:
        registered_predicates = default_rule_registry().predicates()
    allowed = {str(name).strip().lower() for name in registered_predicates}
    return [
        dict(contract)
        for contract in _PREDICATE_CONTRACTS
        if contract["typed_predicate"] in allowed
    ]


QWEN_ONTOLOGY_RULE_PROMPT = """
You are an expert in industrial visual inspection. Analyze the provided normal
reference images of product category "{category}" and induce a category-level
normality program.

The images show K={k_shot} normal sample(s). The program must describe what is
expected to remain true for normal instances of this category. Do not write
defect-specific rules and do not use test labels.

OUTPUT CONTRACT:
Return only one compact JSON array with between 4 and 10 rule objects.
Do not expose analysis, chain-of-thought, prose, Markdown, or code fences.
The array is interpreted as the conjunction (AND) of its rule objects.

ONTOLOGY: the ONLY allowed entities are listed below. Use the canonical_id
exactly. You may use a declared alias, but prefer the canonical_id.
{ontology_json}

PREDICATE REGISTRY: the ONLY allowed predicates are listed below. Use the exact
predicate_name, argument names and parameter names shown here.
{predicate_json}

Rules:
1. Never invent an entity outside the ontology. If you cannot name an observed
   object with a canonical_id, omit the rule.
2. Never invent a predicate outside this list. In particular, CONTAINS,
   TOUCHING, NEAR, REQUIRES, FORBIDS, RIGHT_OF, BELOW and CUSTOM are not
   available in this registry.
3. Instance-granularity entities use COUNT / EXISTS / relation predicates.
   Region-granularity entities (for example granola) must use EXISTS or a
   region-scoped COUNT with target_entity_type; do not treat one region as many
   texture instances.
4. For an instance count inside a compartment, set target_entity_type to the
   compartment canonical_id. Do not write a global COUNT for a statement that
   is only true inside one compartment.
5. Never use COUNT with expected_count = 0. "Not detected" is not evidence of
   absence. If you need to express absence, use EXISTS with a region and an
   explicit support justification, or omit the rule.
6. Do not write a count rule for the container entity itself. The container is
   a coordinate frame, not a normality atom.
7. Prefer relations when two distinct entities are reliably observed in BOTH
   supports. If a stable directional or alignment relation exists, include at
   least one LEFT_OF, ABOVE or ALIGNED_VERTICAL rule instead of returning only
   counts.
8. Do not repeat the same predicate and argument tuple. If a global count and a
   compartment-scoped count express the same invariant, keep only the
   compartment-scoped rule.
9. Do not encode incidental single-image details as production invariants.
   When evidence is insufficient, either omit the rule or mark it with
   "evidence_status": "candidate".
10. Each object must use these fields:
    rule_id, predicate_name, predicate_args, predicate_params, description,
    confidence (0.0-1.0), reasoning (one short evidence sentence).
11. Output a valid JSON array. No trailing text.

Example shapes (replace placeholders with canonical entities and measured
values; do not copy these examples as facts):
[
  {{
    "rule_id": "R1",
    "predicate_name": "COUNT",
    "predicate_args": {{"entity_type": "<instance_entity>",
                       "target_entity_type": "<compartment_or_region>"}},
    "predicate_params": {{"expected_count": 2, "tolerance": 0}},
    "description": "two instances inside one compartment",
    "confidence": 0.9,
    "reasoning": "both normal supports show the same in-region count",
    "evidence_status": "observed"
  }},
  {{
    "rule_id": "R2",
    "predicate_name": "LEFT_OF",
    "predicate_args": {{"entity_type": "<source_entity>",
                       "target_entity_type": "<target_entity>"}},
    "predicate_params": {{"threshold": 0.1}},
    "description": "directional relation between two entities",
    "confidence": 0.9,
    "reasoning": "the same direction holds in both normal supports",
    "evidence_status": "observed"
  }}
]
"""


def build_rule_generation_prompt(
    category: str,
    k_shot: int,
    ontology: Optional[CategoryOntology] = None,
    *,
    registered_predicates: Optional[Sequence[str]] = None,
) -> str:
    """Return the ontology-constrained prompt, or the legacy prompt if absent."""
    selected = ontology or ontology_for_category(category)
    if selected is None:
        return QWEN_RULE_GENERATION_PROMPT.format(
            category=category, k_shot=int(k_shot)
        )
    contracts = predicate_contracts(registered_predicates)
    if not contracts:
        raise ValueError("predicate registry has no Qwen-advertisable predicates")
    return QWEN_ONTOLOGY_RULE_PROMPT.format(
        category=category,
        k_shot=int(k_shot),
        ontology_json=json.dumps(
            selected.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
        ),
        predicate_json=json.dumps(
            contracts, ensure_ascii=False, indent=2, sort_keys=True
        ),
    )


__all__ = [
    "QWEN_ONTOLOGY_RULE_PROMPT", "build_rule_generation_prompt",
    "predicate_contracts", "_PREDICATE_CONTRACTS",
]
