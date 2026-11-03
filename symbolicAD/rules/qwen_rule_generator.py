"""Rule-bank loading / caching and the Stage-4 Qwen hook.

Stage one uses tracked JSON templates under ``configs/rules`` and copies them
into ``cache/rules``.  Qwen generation is intentionally not called here yet:
the pipeline must remain runnable and deterministic until Stage 4 lands.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import RuleKB

DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "configs" / "rules"


def cache_path(category: str, cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "rules" / f"{category}.json"


def load_or_generate_rules(
    category: str,
    cache_dir: str | Path,
    template_dir: str | Path = DEFAULT_TEMPLATE_DIR,
    force: bool = False,
) -> RuleKB:
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
            {
                "name": category,
                "prompts": [f"a photo of a {phrase}"],
                "synonyms": [],
            }
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


def generate_rules_with_qwen(*args, **kwargs) -> RuleKB:
    """Placeholder for Stage 4.

    The eventual implementation will send support images, the category name and
    the allowed atom vocabulary to Qwen3-VL and cache the validated JSON here.
    """
    raise NotImplementedError(
        "Qwen3-VL rule generation is a Stage-4 feature. "
        "Use configs/rules/*.json for now (Stage 1-3)."
    )


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
