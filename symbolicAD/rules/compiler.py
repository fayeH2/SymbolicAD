"""Compile flat atom lists into a support-calibrated hierarchical rule tree.

The compiler is deliberately deterministic and inspectable.  It does not call an
LLM; it turns the generator output into a usable rule bank by:

* removing duplicate expressions;
* removing atoms implied by stronger atoms (for example ``OneToOne(A,B)`` when
  both counts are fixed to one);
* repairing ``CountEq`` values from normal support statistics, or converting
  unstable counts into ``CountInRange``;
* adding missing count rules for components that are required by other atoms;
* grouping atoms into ``existence / count / correspondence / spatial /
  attribute`` sub-goals and building a nested ``AND`` tree (attribute
  alternatives use ``OR``).
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .schema import LogicNode, RelationAtom, RuleKB


@dataclass
class CompileReport:
    removed_duplicates: List[Dict[str, object]] = field(default_factory=list)
    removed_redundant: List[Dict[str, object]] = field(default_factory=list)
    repaired_counts: List[Dict[str, object]] = field(default_factory=list)
    repaired_relations: List[Dict[str, object]] = field(default_factory=list)
    added_atoms: List[Dict[str, object]] = field(default_factory=list)
    coverage_issues: List[str] = field(default_factory=list)
    groups: Dict[str, List[str]] = field(default_factory=dict)
    tree_depth: int = 0
    deduplicated: bool = True
    count_repaired: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "removed_duplicates": self.removed_duplicates,
            "removed_redundant": self.removed_redundant,
            "repaired_counts": self.repaired_counts,
            "repaired_relations": self.repaired_relations,
            "added_atoms": self.added_atoms,
            "coverage_issues": self.coverage_issues,
            "groups": self.groups,
            "tree_depth": self.tree_depth,
        }


def _normalise_expression(expression: str) -> str:
    return re.sub(r"\s+", "", str(expression)).lower()


def _make_expression(kind: str, args: List[object], params: Dict[str, object]) -> str:
    if kind == "SpatialRelation":
        relation = str(params.get("relation", "Inside"))
        return f"{relation}({args[0]},{args[1]})"
    if kind == "AttributeMatch":
        return f"AttributeMatch({args[0]},{args[1]},{args[2]})"
    if kind == "CountInRange":
        return f"CountInRange({args[0]},{_number(args[1])},{_number(args[2])})"
    if kind == "CountEq":
        return f"CountEq({args[0]},{_number(args[1])})"
    if kind == "CountRatio":
        return f"CountRatio({args[0]},{args[1]},{_number(args[2])})"
    if kind == "OneToOne":
        return f"OneToOne({args[0]},{args[1]})"
    if kind == "Exist":
        return f"Exist({args[0]})"
    return str(args)


def _number(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def _tree_depth(node: LogicNode) -> int:
    if node.op == "LEAF":
        return 1
    return 1 + max((_tree_depth(child) for child in node.children), default=0)


def _group_node(atoms: List[RelationAtom], op: str = "AND") -> Optional[LogicNode]:
    if not atoms:
        return None
    if len(atoms) == 1:
        return LogicNode.leaf(atoms[0].atom_id)
    return LogicNode(op=op, children=[LogicNode.leaf(atom.atom_id) for atom in atoms])


def _unique_id(existing: Iterable[str], prefix: str) -> str:
    existing = set(existing)
    index = 0
    while f"{prefix}{index}" in existing:
        index += 1
    return f"{prefix}{index}"


def _attribute_tree(atoms: List[RelationAtom]) -> Optional[LogicNode]:
    if not atoms:
        return None
    explicit: Dict[str, List[RelationAtom]] = {}
    implicit: Dict[Tuple[str, str], List[RelationAtom]] = {}
    for atom in atoms:
        group = str(atom.params.get("group", "")).strip()
        if group:
            explicit.setdefault(group, []).append(atom)
        else:
            implicit.setdefault((atom.components[0], atom.attribute or ""), []).append(atom)

    children: List[LogicNode] = []
    if explicit:
        group_nodes = [
            _group_node(group_atoms, op="AND") for group_atoms in explicit.values()
        ]
        group_nodes = [node for node in group_nodes if node is not None]
        if group_nodes:
            children.append(
                group_nodes[0]
                if len(group_nodes) == 1
                else LogicNode(op="OR", children=group_nodes)
            )
    if implicit:
        implicit_nodes = [
            _group_node(group_atoms, op="OR") for group_atoms in implicit.values()
        ]
        implicit_nodes = [node for node in implicit_nodes if node is not None]
        if implicit_nodes:
            children.append(
                implicit_nodes[0]
                if len(implicit_nodes) == 1
                else LogicNode(op="AND", children=implicit_nodes)
            )
    if not children:
        return None
    return children[0] if len(children) == 1 else LogicNode(op="AND", children=children)


def compile_rule_kb(
    rule_kb: RuleKB,
    support_memory=None,
    config=None,
) -> Tuple[RuleKB, CompileReport]:
    """Return a compiled copy of ``rule_kb`` and an audit report."""
    compiled = copy.deepcopy(rule_kb)
    report = CompileReport()
    compiler = getattr(config, "compiler", None)
    if compiler is not None and not getattr(compiler, "enabled", True):
        report.deduplicated = False
        report.count_repaired = False
        return compiled, report

    drop_implied = bool(getattr(compiler, "drop_implied_exist", True))
    drop_redundant_one_to_one = bool(
        getattr(compiler, "drop_redundant_one_to_one", True)
    )
    repair_counts = bool(getattr(compiler, "repair_counts", False))
    repair_relation_direction = bool(
        getattr(compiler, "repair_relation_direction", False)
    )
    add_missing_counts = bool(getattr(compiler, "add_missing_counts", False))
    stable_std = float(getattr(compiler, "stable_count_std", 0.5))
    range_sigma = float(getattr(compiler, "count_in_range_sigma", 1.0))
    max_range_width = int(getattr(compiler, "max_count_range_width", 6))

    # ------------------------------------------------------------ deduplicate
    deduplicated: List[RelationAtom] = []
    seen = set()
    duplicates = set()
    for atom in compiled.atoms:
        key = _normalise_expression(atom.expression)
        if key in seen:
            duplicates.add(atom.atom_id)
            report.removed_duplicates.append(
                {"atom_id": atom.atom_id, "expression": atom.expression}
            )
            continue
        seen.add(key)
        deduplicated.append(atom)
    compiled.atoms = deduplicated

    # ------------------------------------- remove inverse containment atoms
    valid_containment = {
        (symbol.name, symbol.part_of)
        for symbol in compiled.components
        if symbol.part_of
    }
    containment_kept: List[RelationAtom] = []
    containment_seen = set()
    for atom in compiled.atoms:
        if (
            atom.kind == "SpatialRelation"
            and atom.spatial_relation == "Inside"
            and len(atom.components) == 2
        ):
            a, b = atom.components
            if (a, b) not in valid_containment and (b, a) in valid_containment:
                report.removed_redundant.append(
                    {
                        "atom_id": atom.atom_id,
                        "expression": atom.expression,
                        "reason": "inverse_containment",
                    }
                )
                continue
            if (b, a) in containment_seen and (a, b) not in valid_containment:
                report.removed_redundant.append(
                    {
                        "atom_id": atom.atom_id,
                        "expression": atom.expression,
                        "reason": "duplicate_inverse_containment",
                    }
                )
                continue
            containment_seen.add((a, b))
        containment_kept.append(atom)
    compiled.atoms = containment_kept

    # -------------------------------------------------------- repair counts
    if support_memory is not None and repair_counts:
        for atom in compiled.atoms:
            if atom.kind not in {"CountEq", "CountInRange"} or not atom.components:
                continue
            memory = support_memory.components.get(atom.components[0])
            if memory is None:
                continue
            expected = float(memory.expected_count)
            std = float(memory.count_std)
            if atom.kind == "CountEq":
                target = float(atom.args[1])
                if expected < 0.5 and target >= 1:
                    report.repaired_counts.append(
                        {
                            "atom_id": atom.atom_id,
                            "component": atom.components[0],
                            "old": target,
                            "new": target,
                            "support_mean": expected,
                            "support_std": std,
                            "mode": "skipped_zero_support",
                        }
                    )
                    continue
                if abs(expected - target) <= max(0.5, std):
                    report.repaired_counts.append(
                        {
                            "atom_id": atom.atom_id,
                            "component": atom.components[0],
                            "old": target,
                            "new": target,
                            "support_mean": expected,
                            "support_std": std,
                            "mode": "kept_within_tolerance",
                        }
                    )
                    continue
                if std <= stable_std and abs(expected - round(expected)) <= 0.5:
                    new_target = int(round(expected))
                    if new_target != int(target):
                        atom.args[1] = new_target
                        atom.expression = _make_expression(atom.kind, atom.args, atom.params)
                        report.repaired_counts.append(
                            {
                                "atom_id": atom.atom_id,
                                "component": atom.components[0],
                                "old": target,
                                "new": new_target,
                                "support_mean": expected,
                                "support_std": std,
                                "mode": "count_eq",
                            }
                        )
                    continue
                if (
                    expected >= 0.5
                    and (
                        target < expected - 2.0 * max(std, 0.5)
                        or target > expected + 2.0 * max(std, 0.5)
                    )
                ):
                    low = max(0, int(math.floor(expected - range_sigma * std)))
                    high = int(math.ceil(expected + range_sigma * std))
                    if high - low > max_range_width:
                        high = low + max_range_width
                    atom.kind = "CountInRange"
                    atom.args = [atom.components[0], low, high]
                    atom.expression = _make_expression(atom.kind, atom.args, atom.params)
                    report.repaired_counts.append(
                        {
                            "atom_id": atom.atom_id,
                            "component": atom.components[0],
                            "old": target,
                            "new": [low, high],
                            "support_mean": expected,
                            "support_std": std,
                            "mode": "count_in_range",
                        }
                    )
                else:
                    report.repaired_counts.append(
                        {
                            "atom_id": atom.atom_id,
                            "component": atom.components[0],
                            "old": target,
                            "new": target,
                            "support_mean": expected,
                            "support_std": std,
                            "mode": "kept_unreliable_support_count",
                        }
                    )
            else:
                low, high = float(atom.args[1]), float(atom.args[2])
                if expected < 0.5:
                    report.repaired_counts.append(
                        {
                            "atom_id": atom.atom_id,
                            "component": atom.components[0],
                            "old": [low, high],
                            "new": [low, high],
                            "support_mean": expected,
                            "support_std": std,
                            "mode": "skipped_zero_support",
                        }
                    )
                    continue
                if expected < low or expected > high:
                    new_low = min(low, int(math.floor(expected)))
                    new_high = max(high, int(math.ceil(expected)))
                    if new_high - new_low > max_range_width:
                        new_low = max(0, int(math.floor(expected)))
                        new_high = new_low + max_range_width
                    atom.args = [atom.components[0], new_low, new_high]
                    atom.expression = _make_expression(atom.kind, atom.args, atom.params)
                    report.repaired_counts.append(
                        {
                            "atom_id": atom.atom_id,
                            "component": atom.components[0],
                            "old": [low, high],
                            "new": [new_low, new_high],
                            "support_mean": expected,
                            "support_std": std,
                            "mode": "range_repair",
                        }
                    )

        # ------------------------------------------- add missing count rules
        if add_missing_counts:
            counted_components = {
                atom.components[0]
                for atom in compiled.atoms
                if atom.kind in {"CountEq", "CountInRange", "CountRatio"}
                and atom.components
            }
            referenced_components = {
                component
                for atom in compiled.atoms
                for component in atom.components
            }
            existing_ids = {atom.atom_id for atom in compiled.atoms}
            for symbol in compiled.components:
                name = symbol.name
                if name in counted_components or name not in referenced_components:
                    continue
                memory = support_memory.components.get(name)
                if memory is None:
                    continue
                expected = float(memory.expected_count)
                std = float(memory.count_std)
                if expected < 0.5:
                    report.coverage_issues.append(
                        f"component {name!r} is referenced but support count is 0"
                    )
                    continue
                if std > stable_std or abs(expected - round(expected)) > 0.5:
                    report.coverage_issues.append(
                        f"component {name!r} has unstable support count "
                        f"{expected:.2f}+/-{std:.2f}; no automatic count rule added"
                    )
                    continue
                atom_id = _unique_id(existing_ids, "AUTO")
                existing_ids.add(atom_id)
                if std <= stable_std:
                    args = [name, int(round(expected))]
                    kind = "CountEq"
                    expression = _make_expression(kind, args, {})
                else:
                    low = max(0, int(math.floor(expected - range_sigma * std)))
                    high = int(math.ceil(expected + range_sigma * std))
                    if high - low > max_range_width:
                        high = low + max_range_width
                    args = [name, low, high]
                    kind = "CountInRange"
                    expression = _make_expression(kind, args, {})
                compiled.atoms.append(
                    RelationAtom(
                        atom_id=atom_id,
                        expression=expression,
                        kind=kind,
                        args=args,
                        weight=1.0,
                        normal_prompt=f"{_number(args[1])} {name}",
                        violation_prompt=f"wrong number of {name}",
                        params={},
                    )
                )
                report.added_atoms.append(
                    {
                        "atom_id": atom_id,
                        "expression": expression,
                        "reason": "missing_count_rule",
                        "support_mean": expected,
                        "support_std": std,
                    }
                )

    # ------------------------------------- repair directional relation signs
    if support_memory is not None and repair_relation_direction:
        for atom in compiled.atoms:
            if atom.kind != "SpatialRelation" or len(atom.components) < 2:
                continue
            relation = atom.spatial_relation or "Inside"
            if relation not in {"Above", "Below", "LeftOf", "RightOf"}:
                continue
            a, b = atom.components[0], atom.components[1]
            memory = support_memory.relation(relation, a, b)
            if memory is None or memory.sample_count < 2:
                continue
            dx = float(memory.mean_geometry[0])
            dy = float(memory.mean_geometry[1])
            margin = 0.01
            flipped = None
            if relation == "Above" and dy < -margin:
                flipped = "Below"
            elif relation == "Below" and dy > margin:
                flipped = "Above"
            elif relation == "LeftOf" and dx < -margin:
                flipped = "RightOf"
            elif relation == "RightOf" and dx > margin:
                flipped = "LeftOf"
            if flipped:
                old_expression = atom.expression
                atom.params["relation"] = flipped
                atom.expression = f"{flipped}({a},{b})"
                report.repaired_relations.append(
                    {
                        "atom_id": atom.atom_id,
                        "old": old_expression,
                        "new": atom.expression,
                        "mean_dx": dx,
                        "mean_dy": dy,
                        "reason": "support_geometry_direction",
                    }
                )

    # ------------------------------------------------ remove implied atoms
    count_bounds: Dict[str, Tuple[float, float]] = {}
    exact_counts: Dict[str, float] = {}
    for atom in compiled.atoms:
        if atom.kind == "CountEq" and atom.components:
            value = float(atom.args[1])
            count_bounds[atom.components[0]] = (value, value)
            exact_counts[atom.components[0]] = value
        elif atom.kind == "CountInRange" and atom.components:
            count_bounds[atom.components[0]] = (float(atom.args[1]), float(atom.args[2]))

    kept: List[RelationAtom] = []
    removed_ids = set(duplicates)
    for atom in compiled.atoms:
        remove_reason = ""
        if (
            drop_implied
            and atom.kind == "Exist"
            and atom.components
            and atom.components[0] in count_bounds
            and count_bounds[atom.components[0]][0] >= 1.0
        ):
            remove_reason = "implied_by_count"
        elif (
            drop_redundant_one_to_one
            and atom.kind == "OneToOne"
            and len(atom.components) == 2
        ):
            a, b = atom.components
            if exact_counts.get(a) == 1.0 and exact_counts.get(b) == 1.0:
                remove_reason = "redundant_one_to_one_with_unit_counts"
        if remove_reason:
            removed_ids.add(atom.atom_id)
            report.removed_redundant.append(
                {
                    "atom_id": atom.atom_id,
                    "expression": atom.expression,
                    "reason": remove_reason,
                }
            )
            continue
        kept.append(atom)
    compiled.atoms = kept

    # ------------------------------------------------------- build the tree
    groups: Dict[str, List[RelationAtom]] = {
        "existence": [],
        "count": [],
        "correspondence": [],
        "spatial": [],
        "attribute": [],
    }
    for atom in compiled.atoms:
        if atom.kind == "Exist":
            groups["existence"].append(atom)
        elif atom.kind in {"CountEq", "CountInRange", "CountRatio"}:
            groups["count"].append(atom)
        elif atom.kind == "OneToOne":
            groups["correspondence"].append(atom)
        elif atom.kind == "SpatialRelation":
            groups["spatial"].append(atom)
        elif atom.kind == "AttributeMatch":
            groups["attribute"].append(atom)

    report.groups = {
        name: [atom.atom_id for atom in atoms]
        for name, atoms in groups.items()
        if atoms
    }
    group_nodes: List[LogicNode] = []
    for name in ["existence", "count", "correspondence", "spatial"]:
        node = _group_node(groups[name], op="AND")
        if node is not None:
            group_nodes.append(node)
    attribute_node = _attribute_tree(groups["attribute"])
    if attribute_node is not None:
        group_nodes.append(attribute_node)

    if not group_nodes:
        compiled.formula = LogicNode(op="AND", children=[])
    elif len(group_nodes) == 1:
        compiled.formula = LogicNode(op="AND", children=[group_nodes[0]])
    else:
        compiled.formula = LogicNode(op="AND", children=group_nodes)
    report.tree_depth = _tree_depth(compiled.formula)

    # ------------------------------------------------------ coverage checks
    referenced = {
        component for atom in compiled.atoms for component in atom.components
    }
    for symbol in compiled.components:
        if symbol.name not in referenced:
            report.coverage_issues.append(
                f"component {symbol.name!r} is not referenced by any rule"
            )
    if len(report.groups) <= 1:
        report.coverage_issues.append(
            "compiled tree has only one semantic group (flat rule bank)"
        )

    compiled.metadata["compile_report"] = report.to_dict()
    compiled.metadata["compiled"] = True
    return compiled, report
