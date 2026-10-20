"""Safe, executable product-fuzzy logic programs.

This module deliberately implements a small data-only AST instead of parsing or
evaluating Python expressions.  It is intended for the inference-time part of a
neuro-symbolic detector: grounded rule truths and neural truths are supplied as
numbers in ``[0, 1]``, and every selected rule is visibly executed by the AST.

The product fuzzy operators are

``NOT(a) = 1 - a``
``AND(a_1, ..., a_n) = product(a_i)``
``OR(a_1, ..., a_n) = 1 - product(1 - a_i)``
``IMPLIES(a, b) = OR(NOT(a), b) = 1 - a * (1 - b)``
``XOR(a, b) = OR(AND(a, NOT(b)), AND(NOT(a), b)) = a + b - 2*a*b``.

AST nodes use one of the following forms (operator names are case-insensitive)::

    {"op": "ATOM", "name": "count_rule"}
    {"atom": "count_rule"}                 # shorthand
    {"op": "NOT", "arg": <node>}
    {"op": "AND", "args": [<node>, ...]}
    {"op": "OR", "args": [<node>, ...]}
    {"op": "IMPLIES", "args": [<antecedent>, <consequent>]}
    {"op": "XOR", "args": [<node>, <node>]}

``children`` is accepted as an alias for ``args``.  A node may optionally have
an ``id``; otherwise a deterministic path such as ``root.1.0`` is used.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


TruthValue = Union[float, np.ndarray]
DEFAULT_SPECIAL_ATOMS = frozenset({"neural", "grounding"})
_OPERATORS = frozenset({"ATOM", "NOT", "AND", "OR", "IMPLIES", "XOR"})


class LogicValidationError(ValueError):
    """Raised when a fuzzy-logic AST is not a valid executable rule program."""


def _truth_array(value: Any, *, name: str = "truth value") -> np.ndarray:
    """Convert a truth value to a finite, clipped float64 NumPy array."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    # Grounders and neural calibration can produce a few ulps outside the unit
    # interval.  Clipping also makes the public operators total on finite input.
    return np.clip(array, 0.0, 1.0)


def _public_truth(value: np.ndarray) -> TruthValue:
    """Return Python floats for scalar results and arrays for spatial results."""
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 0:
        return float(value)
    return value


def _broadcast(values: Sequence[Any], *, operator: str) -> List[np.ndarray]:
    if not values:
        raise ValueError(f"{operator} requires at least one operand")
    arrays = [_truth_array(value, name=f"{operator} operand") for value in values]
    try:
        return list(np.broadcast_arrays(*arrays))
    except ValueError as exc:
        shapes = [tuple(array.shape) for array in arrays]
        raise ValueError(
            f"{operator} operands cannot be broadcast together: {shapes}"
        ) from exc


def fuzzy_not(value: Any) -> TruthValue:
    """Product-fuzzy negation, with finite output in ``[0, 1]``."""
    array = _truth_array(value, name="NOT operand")
    return _public_truth(np.clip(1.0 - array, 0.0, 1.0))


def fuzzy_and(*values: Any) -> TruthValue:
    """Product t-norm over one or more broadcast-compatible operands."""
    arrays = _broadcast(values, operator="AND")
    result = np.ones_like(arrays[0], dtype=np.float64)
    for array in arrays:
        # Sequential multiplication avoids constructing an additional stacked
        # tensor for full-resolution rule maps.
        np.multiply(result, array, out=result)
    return _public_truth(np.clip(result, 0.0, 1.0))


def fuzzy_or(*values: Any) -> TruthValue:
    """Probabilistic-sum t-conorm over broadcast-compatible operands.

    ``-expm1(sum(log1p(-x)))`` avoids cancellation when all rule violations are
    very small.  Exact truth values of one are handled naturally as ``-inf`` in
    the log domain and produce an exact result of one.
    """
    arrays = _broadcast(values, operator="OR")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_complement = np.zeros_like(arrays[0], dtype=np.float64)
        for array in arrays:
            log_complement += np.log1p(-array)
        result = -np.expm1(log_complement)
    return _public_truth(np.clip(result, 0.0, 1.0))


def fuzzy_implies(antecedent: Any, consequent: Any) -> TruthValue:
    """Reichenbach implication induced by product NOT/OR semantics."""
    return fuzzy_or(fuzzy_not(antecedent), consequent)


def fuzzy_xor(*values: Any) -> TruthValue:
    """Product-fuzzy XOR over exactly two broadcast-compatible operands.

    XOR(a, b) = OR(AND(a, NOT(b)), AND(NOT(a), b))
             = a*(1-b) + (1-a)*b
             = a + b - 2*a*b
    """
    arrays = _broadcast(values, operator="XOR")
    if len(arrays) != 2:
        raise ValueError("XOR requires exactly two operands")
    a, b = arrays[0], arrays[1]
    result = a + b - 2.0 * a * b
    return _public_truth(np.clip(result, 0.0, 1.0))


def _operator(node: Any) -> str:
    if isinstance(node, str):
        return "ATOM"
    if not isinstance(node, Mapping):
        raise LogicValidationError("Every AST node must be a mapping or atom string")
    if "atom" in node and "op" not in node:
        return "ATOM"
    op = str(node.get("op", "")).strip().upper()
    if op not in _OPERATORS:
        raise LogicValidationError(f"Unknown fuzzy-logic operator: {op or '<missing>'}")
    return op


def _atom_name(node: Any) -> str:
    if isinstance(node, str):
        name = node
    else:
        name = node.get("name", node.get("atom", ""))
    name = str(name).strip()
    if not name:
        raise LogicValidationError("ATOM requires a non-empty string name")
    return name


def _children(node: Mapping[str, Any], op: str) -> List[Any]:
    if op == "NOT" and "arg" in node:
        raw: Any = [node["arg"]]
    else:
        raw = node.get("args", node.get("children"))
    if not isinstance(raw, (list, tuple)):
        raise LogicValidationError(f"{op} requires an args/children sequence")
    children = list(raw)
    if op == "NOT" and len(children) != 1:
        raise LogicValidationError("NOT requires exactly one operand")
    if op == "IMPLIES" and len(children) != 2:
        raise LogicValidationError("IMPLIES requires exactly two operands")
    if op == "XOR" and len(children) != 2:
        raise LogicValidationError("XOR requires exactly two operands")
    if op in {"AND", "OR"} and not children:
        raise LogicValidationError(f"{op} requires at least one operand")
    return children


def _node_id(node: Any, path: str) -> str:
    if isinstance(node, Mapping) and "id" in node:
        node_id = str(node["id"]).strip()
        if not node_id:
            raise LogicValidationError("AST node id must be a non-empty string")
        return node_id
    return path


@dataclass(frozen=True)
class LogicValidationReport:
    """Static reachability information for an executable logic program."""

    active_atom_counts: Dict[str, int]
    special_atom_counts: Dict[str, int]
    execution_atoms: Tuple[str, ...]
    node_count: int


def validate_logic_ast(
    ast: Any,
    active_atoms: Iterable[str],
    *,
    special_atoms: Iterable[str] = DEFAULT_SPECIAL_ATOMS,
) -> LogicValidationReport:
    """Validate AST shape and exact active-rule reachability.

    Every name in ``active_atoms`` must occur as a reachable leaf exactly once.
    Leaves outside that set are rejected, except for explicitly allowed neural
    or grounding atoms.  Special atoms are optional, but cannot be duplicated;
    this keeps the trace an unambiguous record of one execution per evidence
    source.
    """
    active = tuple(str(name).strip() for name in active_atoms)
    special = frozenset(str(name).strip() for name in special_atoms)
    if any(not name for name in active) or "" in special:
        raise LogicValidationError("Atom names must be non-empty strings")
    if len(set(active)) != len(active):
        duplicates = sorted(name for name, count in Counter(active).items() if count > 1)
        raise LogicValidationError(f"active_atoms contains duplicates: {duplicates}")
    overlap = set(active).intersection(special)
    if overlap:
        raise LogicValidationError(
            f"Atoms cannot be both active rules and special atoms: {sorted(overlap)}"
        )

    active_set = frozenset(active)
    atom_order: List[str] = []
    node_ids: set[str] = set()
    visited_objects: set[int] = set()

    def visit(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            identity = id(node)
            if identity in visited_objects:
                raise LogicValidationError(
                    "AST nodes cannot be shared or cyclic; use a tree with explicit leaves"
                )
            visited_objects.add(identity)

        node_id = _node_id(node, path)
        if node_id in node_ids:
            raise LogicValidationError(f"Duplicate AST node id: {node_id}")
        node_ids.add(node_id)
        op = _operator(node)
        if op == "ATOM":
            name = _atom_name(node)
            if name not in active_set and name not in special:
                raise LogicValidationError(
                    f"Inactive or unknown atom is reachable from the root: {name}"
                )
            atom_order.append(name)
            return
        assert isinstance(node, Mapping)
        for index, child in enumerate(_children(node, op)):
            visit(child, f"{path}.{index}")

    visit(ast, "root")
    counts = Counter(atom_order)
    missing = sorted(name for name in active if counts[name] == 0)
    duplicate = sorted(name for name in active if counts[name] > 1)
    duplicate_special = sorted(name for name in special if counts[name] > 1)
    if missing or duplicate or duplicate_special:
        details = []
        if missing:
            details.append(f"missing active atoms={missing}")
        if duplicate:
            details.append(f"duplicated active atoms={duplicate}")
        if duplicate_special:
            details.append(f"duplicated special atoms={duplicate_special}")
        raise LogicValidationError("Invalid atom reachability: " + "; ".join(details))

    return LogicValidationReport(
        active_atom_counts={name: counts[name] for name in active},
        special_atom_counts={name: counts[name] for name in sorted(special) if counts[name]},
        execution_atoms=tuple(atom_order),
        node_count=len(node_ids),
    )


def _trace_truth(value: Any) -> TruthValue:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return array.copy()


def execute_logic_ast(
    ast: Any,
    atom_values: Mapping[str, Any],
    *,
    active_atoms: Optional[Iterable[str]] = None,
    special_atoms: Iterable[str] = DEFAULT_SPECIAL_ATOMS,
) -> Tuple[TruthValue, List[Dict[str, Any]]]:
    """Execute a validated AST and return ``(root_truth, postorder_trace)``.

    If ``active_atoms`` is omitted, all non-special leaves in the AST are
    treated as active.  Production callers should normally pass the selected
    rule ids explicitly, so an inactive candidate cannot enter the program.
    """
    special = frozenset(str(name).strip() for name in special_atoms)

    if active_atoms is None:
        discovered: List[str] = []

        def discover(node: Any) -> None:
            op = _operator(node)
            if op == "ATOM":
                name = _atom_name(node)
                if name not in special:
                    discovered.append(name)
                return
            assert isinstance(node, Mapping)
            for child in _children(node, op):
                discover(child)

        discover(ast)
        active_atoms = discovered

    validate_logic_ast(ast, active_atoms, special_atoms=special)
    trace: List[Dict[str, Any]] = []

    def evaluate(node: Any, path: str) -> Tuple[TruthValue, str]:
        op = _operator(node)
        node_id = _node_id(node, path)
        if op == "ATOM":
            name = _atom_name(node)
            if name not in atom_values:
                raise ValueError(f"Missing truth value for atom: {name}")
            output = _public_truth(_truth_array(atom_values[name], name=f"atom {name}"))
            trace.append({
                "node_id": node_id,
                "op": "ATOM",
                "atom": name,
                "input_node_ids": [],
                "output": _trace_truth(output),
            })
            return output, node_id

        assert isinstance(node, Mapping)
        values: List[TruthValue] = []
        child_ids: List[str] = []
        for index, child in enumerate(_children(node, op)):
            value, child_id = evaluate(child, f"{path}.{index}")
            values.append(value)
            child_ids.append(child_id)
        try:
            if op == "NOT":
                output = fuzzy_not(values[0])
            elif op == "AND":
                output = fuzzy_and(*values)
            elif op == "OR":
                output = fuzzy_or(*values)
            elif op == "XOR":
                output = fuzzy_xor(*values)
            else:
                output = fuzzy_implies(values[0], values[1])
        except ValueError as exc:
            raise ValueError(f"Failed to execute {op} node {node_id}: {exc}") from exc
        trace.append({
            "node_id": node_id,
            "op": op,
            "input_node_ids": child_ids,
            "output": _trace_truth(output),
        })
        return output, node_id

    result, _ = evaluate(ast, "root")
    return result, trace


class FuzzyLogicProgram:
    """Validated reusable AST whose active rules cannot silently be skipped."""

    def __init__(
        self,
        ast: Any,
        active_atoms: Iterable[str],
        *,
        special_atoms: Iterable[str] = DEFAULT_SPECIAL_ATOMS,
    ) -> None:
        self.ast = ast
        self.active_atoms = tuple(str(name).strip() for name in active_atoms)
        self.special_atoms = frozenset(str(name).strip() for name in special_atoms)
        self.validation = validate_logic_ast(
            self.ast, self.active_atoms, special_atoms=self.special_atoms
        )

    def evaluate(
        self, atom_values: Mapping[str, Any]
    ) -> Tuple[TruthValue, List[Dict[str, Any]]]:
        return execute_logic_ast(
            self.ast,
            atom_values,
            active_atoms=self.active_atoms,
            special_atoms=self.special_atoms,
        )


__all__ = [
    "DEFAULT_SPECIAL_ATOMS",
    "FuzzyLogicProgram",
    "LogicValidationError",
    "LogicValidationReport",
    "TruthValue",
    "execute_logic_ast",
    "fuzzy_and",
    "fuzzy_implies",
    "fuzzy_not",
    "fuzzy_or",
    "fuzzy_xor",
    "validate_logic_ast",
]
