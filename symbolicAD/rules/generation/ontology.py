"""Category-level entity ontology shared by rule generation and perception.

The ontology is the single source of truth for:

* which canonical entity IDs Qwen may use;
* which aliases are allowed during semantic-visual alignment;
* the granularity of each entity (container / instance / region / part);
* the GroundingDINO detector phrases.

It deliberately performs exact, declared alias matching.  It never does fuzzy
string matching or silently maps an unknown entity (for example ``apple``) to a
visually similar canonical entity (for example ``peach``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


def normalize_entity_name(name: str) -> str:
    """Normalise case, underscores/hyphens and repeated whitespace only."""
    text = str(name).strip().lower()
    for separator in ("_", "-", "/"):
        text = text.replace(separator, " ")
    return " ".join(text.split())


@dataclass(frozen=True)
class EntitySpec:
    canonical_id: str
    granularity: str
    aliases: Tuple[str, ...] = ()
    detector_prompts: Tuple[str, ...] = ()
    optional_attributes: Tuple[str, ...] = ()
    # Derived entities are deterministic scene geometry (for example the
    # left/right compartments split from the container).  They are valid rule
    # operands but are never sent to GroundingDINO as detector prompts.
    derived: bool = False

    def __post_init__(self) -> None:
        if self.granularity not in {"container", "instance", "region", "part"}:
            raise ValueError(f"invalid granularity: {self.granularity}")
        if not self.canonical_id.strip():
            raise ValueError("canonical_id cannot be empty")
        if not self.detector_prompts and not self.derived:
            raise ValueError(
                f"{self.canonical_id} must declare at least one detector prompt"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "granularity": self.granularity,
            "aliases": list(self.aliases),
            "detector_prompts": list(self.detector_prompts),
            "optional_attributes": list(self.optional_attributes),
            "derived": bool(self.derived),
        }


@dataclass(frozen=True)
class Canonicalization:
    source_name: str
    canonical_id: Optional[str]
    status: str
    reason_code: str
    matched_alias: Optional[str] = None

    @property
    def accepted(self) -> bool:
        return self.status in {"ALIGNED", "ALIGNED_WITH_CANONICALIZATION"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "canonical_id": self.canonical_id,
            "status": self.status,
            "reason_code": self.reason_code,
            "matched_alias": self.matched_alias,
        }


@dataclass(frozen=True)
class CategoryOntology:
    category: str
    entities: Mapping[str, EntitySpec]
    supertypes: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.category.strip():
            raise ValueError("category cannot be empty")
        if not self.entities:
            raise ValueError("ontology needs at least one entity")
        for canonical, spec in self.entities.items():
            if canonical != spec.canonical_id:
                raise ValueError(
                    f"entity key {canonical!r} does not match canonical_id "
                    f"{spec.canonical_id!r}"
                )
        for supertype, members in self.supertypes.items():
            missing = [member for member in members if member not in self.entities]
            if missing:
                raise ValueError(
                    f"supertype {supertype!r} references unknown members: {missing}"
                )

    def canonicalize(self, name: str) -> Canonicalization:
        source = str(name).strip()
        normalized = normalize_entity_name(source)
        if not normalized:
            return Canonicalization(source, None, "UNRESOLVED", "EMPTY_ENTITY_NAME")

        for canonical_id in self.entities:
            if normalize_entity_name(canonical_id) == normalized:
                return Canonicalization(
                    source, canonical_id, "ALIGNED", "CANONICAL_ID"
                )

        matches = []
        for canonical_id, spec in self.entities.items():
            alias_map = {
                normalize_entity_name(alias): alias
                for alias in spec.aliases
            }
            if normalized in alias_map:
                matches.append((canonical_id, alias_map[normalized]))
        if len(matches) == 1:
            canonical_id, alias = matches[0]
            return Canonicalization(
                source, canonical_id, "ALIGNED_WITH_CANONICALIZATION",
                "DECLARED_ALIAS", matched_alias=alias,
            )
        if len(matches) > 1:
            return Canonicalization(source, None, "AMBIGUOUS", "AMBIGUOUS_ALIAS")

        for supertype, members in self.supertypes.items():
            if normalize_entity_name(supertype) == normalized:
                return Canonicalization(
                    source, supertype, "ALIGNED_WITH_CANONICALIZATION",
                    "DECLARED_SUPERTYPE", matched_alias=supertype,
                )
        return Canonicalization(source, None, "UNRESOLVED", "ONTOLOGY_MISMATCH")

    def entity(self, canonical_id: str) -> Optional[EntitySpec]:
        return self.entities.get(str(canonical_id))

    def granularity(self, canonical_id: str) -> Optional[str]:
        if canonical_id in self.supertypes:
            return "instance"
        spec = self.entity(canonical_id)
        return None if spec is None else spec.granularity

    def is_derived(self, canonical_id: str) -> bool:
        spec = self.entity(canonical_id)
        return bool(spec is not None and spec.derived)

    def detector_vocabulary(
        self, canonical_ids: Optional[Sequence[str]] = None
    ) -> Dict[str, Tuple[str, ...]]:
        """Return the GroundingDINO phrase map for selected canonical entities."""
        selected = list(canonical_ids) if canonical_ids is not None else list(self.entities)
        vocabulary: Dict[str, Tuple[str, ...]] = {}
        for canonical_id in selected:
            if canonical_id in self.supertypes:
                continue
            spec = self.entity(canonical_id)
            if spec is not None and not spec.derived:
                vocabulary[canonical_id] = tuple(spec.detector_prompts)
        return vocabulary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "entities": {
                canonical_id: spec.to_dict()
                for canonical_id, spec in self.entities.items()
            },
            "supertypes": {
                name: list(members) for name, members in self.supertypes.items()
            },
        }


DEFAULT_BREAKFAST_ONTOLOGY = CategoryOntology(
    category="breakfast_box",
    entities={
        "bento_box": EntitySpec(
            canonical_id="bento_box",
            granularity="container",
            aliases=(
                "bento_box", "bento box", "lunch box", "lunchbox",
                "breakfast box", "breakfast_box", "tray",
            ),
            detector_prompts=("bento box", "lunch box", "bento", "lunchbox"),
        ),
        "orange": EntitySpec(
            canonical_id="orange",
            granularity="instance",
            aliases=("orange", "oranges"),
            detector_prompts=("orange",),
        ),
        "peach": EntitySpec(
            canonical_id="peach",
            granularity="instance",
            aliases=("peach", "peaches"),
            detector_prompts=("peach",),
        ),
        "granola": EntitySpec(
            canonical_id="granola",
            granularity="region",
            aliases=("granola", "cereal", "muesli", "granola region"),
            detector_prompts=("granola", "cereal", "muesli"),
        ),
        "almond": EntitySpec(
            canonical_id="almond",
            granularity="instance",
            aliases=("almond", "almonds"),
            detector_prompts=("almond",),
        ),
        "banana_slice": EntitySpec(
            canonical_id="banana_slice",
            granularity="instance",
            aliases=(
                "banana_slice", "banana slice", "banana slices",
                "banana", "bananas", "dried banana", "dried banana slice",
            ),
            detector_prompts=("banana",),
        ),
        "left_compartment": EntitySpec(
            canonical_id="left_compartment",
            granularity="region",
            aliases=("left_compartment", "left compartment", "left region"),
            derived=True,
        ),
        "right_compartment": EntitySpec(
            canonical_id="right_compartment",
            granularity="region",
            aliases=("right_compartment", "right compartment", "right region"),
            derived=True,
        ),
    },
    supertypes={"fruit": ("peach", "orange")},
)


_ONTOLOGY_BY_CATEGORY: Dict[str, CategoryOntology] = {
    DEFAULT_BREAKFAST_ONTOLOGY.category: DEFAULT_BREAKFAST_ONTOLOGY,
}


def ontology_for_category(category: str) -> Optional[CategoryOntology]:
    return _ONTOLOGY_BY_CATEGORY.get(str(category).strip().lower())


def register_ontology(ontology: CategoryOntology) -> None:
    _ONTOLOGY_BY_CATEGORY[ontology.category.strip().lower()] = ontology


__all__ = [
    "EntitySpec", "Canonicalization", "CategoryOntology",
    "DEFAULT_BREAKFAST_ONTOLOGY", "ontology_for_category", "register_ontology",
    "normalize_entity_name",
]
