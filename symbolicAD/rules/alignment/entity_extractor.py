"""Extract the atomic entity names referenced by a rule bank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

from ..rule_schema import RuleSpec


@dataclass(frozen=True)
class AtomicEntityUse:
    source_name: str
    rule_ids: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "rule_ids": list(self.rule_ids),
        }


def extract_atomic_entities(rules: Iterable[RuleSpec]) -> Tuple[AtomicEntityUse, ...]:
    """Return unique entity names in first-seen order with their rule IDs."""
    order = []
    uses: Dict[str, list[str]] = {}
    for rule in rules:
        for entity in rule.entity_types:
            name = str(entity)
            if not name:
                continue
            if name not in uses:
                uses[name] = []
                order.append(name)
            if rule.rule_id not in uses[name]:
                uses[name].append(rule.rule_id)
    return tuple(
        AtomicEntityUse(name, tuple(uses[name])) for name in order
    )


__all__ = ["AtomicEntityUse", "extract_atomic_entities"]
