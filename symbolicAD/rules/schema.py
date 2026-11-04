"""Explicit rule knowledge base shared by the logic tree and grounding code."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ALLOWED_ATOM_KINDS = (
    "Exist",
    "CountEq",
    "CountInRange",
    "CountRatio",
    "OneToOne",
    "SpatialRelation",
    "AttributeMatch",
)

SPATIAL_RELATIONS = (
    "Inside",
    "LeftOf",
    "RightOf",
    "Above",
    "Below",
    "CenterAligned",
)

ALLOWED_RELATIONS = ALLOWED_ATOM_KINDS + SPATIAL_RELATIONS

_CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$")


class RuleSchemaError(ValueError):
    """Raised when a rule document cannot be interpreted at all."""


@dataclass
class ComponentSymbol:
    name: str
    prompts: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    role: str = ""
    part_of: str = ""

    def all_prompts(self) -> List[str]:
        values = list(self.prompts) + list(self.synonyms)
        seen, result = set(), []
        for value in values:
            value = str(value).strip()
            if value and value.lower() not in seen:
                seen.add(value.lower())
                result.append(value)
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComponentSymbol":
        name = str(payload.get("name", "")).strip()
        if not name:
            raise RuleSchemaError("component entry is missing a name")
        prompts = _as_string_list(payload.get("prompts", []))
        synonyms = _as_string_list(payload.get("synonyms", []))
        role = str(payload.get("role", "")).strip()
        part_of = str(payload.get("part_of", "")).strip()
        return cls(
            name=name,
            prompts=prompts,
            synonyms=synonyms,
            role=role,
            part_of=part_of,
        )


@dataclass
class RelationAtom:
    atom_id: str
    expression: str
    kind: str
    args: List[Any] = field(default_factory=list)
    weight: float = 1.0
    normal_prompt: str = ""
    violation_prompt: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def components(self) -> List[str]:
        """Return only the arguments that reference component symbols."""
        if self.kind in {"Exist", "CountEq", "CountInRange", "AttributeMatch"}:
            return [str(self.args[0])] if self.args else []
        if self.kind in {"CountRatio", "OneToOne", "SpatialRelation"}:
            return [str(value) for value in self.args[:2]]
        return [str(value) for value in self.args if isinstance(value, str)]

    @property
    def spatial_relation(self) -> Optional[str]:
        if self.kind == "SpatialRelation":
            return str(self.params.get("relation", ""))
        return None

    @property
    def attribute(self) -> Optional[str]:
        if self.kind == "AttributeMatch" and len(self.args) >= 2:
            return str(self.args[1])
        return None

    @property
    def attribute_value(self) -> Optional[str]:
        if self.kind == "AttributeMatch" and len(self.args) >= 3:
            return str(self.args[2])
        return None

    @property
    def attribute_prompts(self) -> List[str]:
        prompts = _as_string_list(self.params.get("attribute_prompts", []))
        if prompts:
            return prompts
        value = self.attribute_value or ""
        attribute = self.attribute or "attribute"
        return [f"a {value} {attribute}", f"{value}", f"{attribute} {value}"]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RelationAtom":
        atom_id = str(payload.get("id", "")).strip()
        expression = str(payload.get("expression", "")).strip()
        if not atom_id:
            raise RuleSchemaError("atom entry is missing an id")
        if not expression:
            raise RuleSchemaError(f"atom {atom_id} is missing an expression")
        kind, args, params = parse_expression(expression)
        weight = float(payload.get("weight", 1.0))
        normal_prompt = str(payload.get("normal_prompt", "")).strip()
        violation_prompt = str(payload.get("violation_prompt", "")).strip()
        params.update(dict(payload.get("params", {}) or {}))
        return cls(
            atom_id=atom_id,
            expression=expression,
            kind=kind,
            args=args,
            weight=weight,
            normal_prompt=normal_prompt,
            violation_prompt=violation_prompt,
            params=params,
        )


@dataclass
class LogicNode:
    op: str
    children: List["LogicNode"] = field(default_factory=list)
    atom_id: str = ""

    @classmethod
    def leaf(cls, atom_id: str) -> "LogicNode":
        return cls(op="LEAF", atom_id=str(atom_id))

    @classmethod
    def from_dict(cls, payload: Any) -> "LogicNode":
        if isinstance(payload, str):
            return cls.leaf(payload)
        if not isinstance(payload, Mapping):
            raise RuleSchemaError(f"unsupported logic node: {payload!r}")
        op = str(payload.get("op", "")).upper()
        if op == "LEAF":
            atom_id = payload.get("atom_id") or payload.get("id") or payload.get("atom")
            if not atom_id:
                raise RuleSchemaError("LEAF node is missing atom_id")
            return cls.leaf(str(atom_id))
        if op not in {"AND", "OR", "NOT"}:
            raise RuleSchemaError(f"unsupported logic operator: {op!r}")
        children = [cls.from_dict(item) for item in payload.get("children", [])]
        if op == "NOT" and len(children) != 1:
            raise RuleSchemaError("NOT node must have exactly one child")
        if not children:
            raise RuleSchemaError(f"{op} node has no children")
        return cls(op=op, children=children)

    def to_dict(self) -> Dict[str, Any]:
        if self.op == "LEAF":
            return {"op": "LEAF", "atom_id": self.atom_id}
        return {"op": self.op, "children": [child.to_dict() for child in self.children]}

    def atom_ids(self) -> List[str]:
        if self.op == "LEAF":
            return [self.atom_id]
        result: List[str] = []
        for child in self.children:
            result.extend(child.atom_ids())
        return result


@dataclass
class PromptLexicon:
    component_prompts: Dict[str, List[str]] = field(default_factory=dict)
    atom_prompts: Dict[str, Dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromptLexicon":
        component_prompts = {
            str(key): _as_string_list(value)
            for key, value in dict(payload.get("component_prompts", {}) or {}).items()
        }
        atom_prompts = {
            str(key): {
                "normal": str(dict(value).get("normal", "")),
                "violation": str(dict(value).get("violation", "")),
            }
            for key, value in dict(payload.get("atom_prompts", {}) or {}).items()
        }
        return cls(component_prompts=component_prompts, atom_prompts=atom_prompts)


@dataclass
class RuleKB:
    category: str
    components: List[ComponentSymbol]
    atoms: List[RelationAtom]
    formula: LogicNode
    lexicon: PromptLexicon = field(default_factory=PromptLexicon)
    support_evidence: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def component(self, name: str) -> Optional[ComponentSymbol]:
        for item in self.components:
            if item.name == name:
                return item
        return None

    def atom(self, atom_id: str) -> Optional[RelationAtom]:
        for item in self.atoms:
            if item.atom_id == atom_id:
                return item
        return None

    def prompts_for(self, component: str) -> List[str]:
        symbol = self.component(component)
        prompts = symbol.all_prompts() if symbol else []
        prompts = prompts or list(self.lexicon.component_prompts.get(component, []))
        if not prompts:
            prompts = [component.replace("_", " ")]
        return prompts

    def atom_prompts(self, atom_id: str) -> Dict[str, str]:
        atom = self.atom(atom_id)
        result = {
            "normal": atom.normal_prompt if atom else "",
            "violation": atom.violation_prompt if atom else "",
        }
        for key, value in self.lexicon.atom_prompts.get(atom_id, {}).items():
            if value and not result.get(key):
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuleKB":
        category = str(payload.get("category", "")).strip()
        components = [ComponentSymbol.from_dict(item) for item in payload.get("components", [])]
        atoms = [RelationAtom.from_dict(item) for item in payload.get("atoms", [])]
        formula_payload = payload.get("formula")
        if formula_payload is None:
            formula = LogicNode(op="AND", children=[LogicNode.leaf(atom.atom_id) for atom in atoms])
        else:
            formula = LogicNode.from_dict(formula_payload)
        lexicon = PromptLexicon.from_dict(payload.get("lexicon", {}) or {})
        return cls(
            category=category,
            components=components,
            atoms=atoms,
            formula=formula,
            lexicon=lexicon,
            support_evidence=dict(payload.get("support_evidence", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "components": [
                {
                    "name": item.name,
                    "prompts": item.prompts,
                    "synonyms": item.synonyms,
                    **({"role": item.role} if item.role else {}),
                    **({"part_of": item.part_of} if item.part_of else {}),
                }
                for item in self.components
            ],
            "atoms": [
                {
                    "id": item.atom_id,
                    "expression": item.expression,
                    "weight": item.weight,
                    "normal_prompt": item.normal_prompt,
                    "violation_prompt": item.violation_prompt,
                    **({"params": item.params} if item.params else {}),
                }
                for item in self.atoms
            ],
            "formula": self.formula.to_dict(),
            "lexicon": {
                "component_prompts": self.lexicon.component_prompts,
                "atom_prompts": self.lexicon.atom_prompts,
            },
            "support_evidence": self.support_evidence,
            "metadata": self.metadata,
        }

    @classmethod
    def load(cls, path: str | Path) -> "RuleKB":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _split_args(raw: str) -> List[str]:
    parts, depth, buffer = [], 0, []
    for char in raw:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
    tail = "".join(buffer).strip()
    if tail:
        parts.append(tail)
    return parts


def _coerce_arg(token: str) -> Any:
    token = token.strip().strip('"').strip("'")
    try:
        if re.fullmatch(r"[+-]?\d+", token):
            return int(token)
        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", token):
            return float(token)
    except ValueError:
        pass
    return token


def parse_expression(expression: str) -> tuple[str, List[Any], Dict[str, Any]]:
    """Parse ``CountEq(Screw, 4)`` / ``Inside(Screw, Slot)`` into a typed atom."""
    match = _CALL_RE.match(str(expression))
    if not match:
        raise RuleSchemaError(f"cannot parse atom expression: {expression!r}")
    name, raw_args = match.group(1), match.group(2)
    args = [_coerce_arg(token) for token in _split_args(raw_args)]
    params: Dict[str, Any] = {}

    if name in SPATIAL_RELATIONS:
        if len(args) != 2:
            raise RuleSchemaError(f"{name} expects two components, got {args}")
        params["relation"] = name
        return "SpatialRelation", args, params

    if name not in ALLOWED_ATOM_KINDS:
        raise RuleSchemaError(f"unsupported atom kind: {name!r}")

    if name == "Exist":
        if len(args) != 1:
            raise RuleSchemaError("Exist expects one component")
    elif name == "CountEq":
        if len(args) != 2:
            raise RuleSchemaError("CountEq expects (Component, K)")
    elif name == "CountInRange":
        if len(args) != 3:
            raise RuleSchemaError("CountInRange expects (Component, low, high)")
        low, high = float(args[1]), float(args[2])
        if low < 0 or high < low:
            raise RuleSchemaError("CountInRange requires 0 <= low <= high")
    elif name == "CountRatio":
        if len(args) != 3:
            raise RuleSchemaError("CountRatio expects (A, B, ratio)")
    elif name == "OneToOne":
        if len(args) != 2:
            raise RuleSchemaError("OneToOne expects (A, B)")
    elif name == "AttributeMatch":
        if len(args) != 3:
            raise RuleSchemaError("AttributeMatch expects (Component, attribute, value)")
        if not str(args[0]).strip() or not str(args[1]).strip() or not str(args[2]).strip():
            raise RuleSchemaError("AttributeMatch requires non-empty component/attribute/value")
    elif name == "SpatialRelation":
        if len(args) != 3:
            raise RuleSchemaError("SpatialRelation expects (A, B, Relation)")
        relation = str(args[2])
        if relation not in SPATIAL_RELATIONS:
            raise RuleSchemaError(f"unsupported spatial relation: {relation!r}")
        params["relation"] = relation
        args = args[:2]
    return name, args, params
