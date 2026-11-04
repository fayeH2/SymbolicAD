"""Fuzzy logic-tree execution over atom truth values."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import numpy as np

from .schema import LogicNode


@dataclass
class AtomicResult:
    atom_id: str
    expression: str
    kind: str
    truth: float
    violation: float
    responsibility_map: np.ndarray
    valid: bool
    weight: float = 1.0
    evidence: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "atom_id": self.atom_id,
            "expression": self.expression,
            "kind": self.kind,
            "truth": float(self.truth),
            "violation": float(self.violation),
            "weight": float(self.weight),
            "valid": bool(self.valid),
            "evidence": _jsonable(self.evidence),
        }


@dataclass
class LogicResult:
    root_truth: float
    logic_score: float
    failed_atoms: List[Dict[str, object]]
    atom_truths: Dict[str, float]
    active_failed_atom_ids: List[str] = field(default_factory=list)
    node_trace: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "root_truth": float(self.root_truth),
            "logic_score": float(self.logic_score),
            "failed_atoms": _jsonable(self.failed_atoms),
            "active_failed_atom_ids": list(self.active_failed_atom_ids),
            "atom_truths": {key: float(value) for key, value in self.atom_truths.items()},
            "node_trace": _jsonable(self.node_trace),
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(min(high, max(low, value)))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _leaf_truth(atom_id: str, atom_results: Mapping[str, AtomicResult]) -> float:
    result = atom_results.get(atom_id)
    if result is None:
        return 1.0
    if not result.valid or result.weight <= 0.0:
        # Disabled atoms are neutral so a single bad rule cannot veto the tree.
        return 1.0
    truth = _clamp(result.truth)
    weight = _clamp(result.weight)
    # Fractional weights are reliability scores: weight 1 keeps the raw truth,
    # weight 0 makes the leaf neutral. The weight is frozen by offline support
    # compilation and never changes from query-set statistics.
    return _clamp(1.0 - weight * (1.0 - truth))


def _evaluate(
    node: LogicNode,
    atom_results: Mapping[str, AtomicResult],
    epsilon: float,
    trace: List[Dict[str, object]],
) -> float:
    if node.op == "LEAF":
        value = _leaf_truth(node.atom_id, atom_results)
        trace.append({"op": "LEAF", "atom_id": node.atom_id, "truth": value})
        return value

    child_values = [
        _evaluate(child, atom_results, epsilon, trace) for child in node.children
    ]
    if node.op == "NOT":
        value = 1.0 - child_values[0]
    elif not child_values:
        value = 1.0
    elif node.op == "AND":
        mean_log = sum(math.log(_clamp(item, epsilon, 1.0)) for item in child_values)
        value = math.exp(mean_log / len(child_values))
    elif node.op == "OR":
        mean_log = sum(math.log(_clamp(1.0 - item, epsilon, 1.0)) for item in child_values)
        value = 1.0 - math.exp(mean_log / len(child_values))
    else:  # pragma: no cover - parser rejects other operators
        raise ValueError(f"unsupported logic operator: {node.op}")
    value = _clamp(value)
    trace.append({"op": node.op, "truth": value, "children": node.children})
    return value


def _active_failed_atom_ids(
    node: LogicNode,
    atom_results: Mapping[str, AtomicResult],
    fail_threshold: float,
    epsilon: float,
) -> List[str]:
    """Return failed leaves that can actually violate the formula.

    In particular, a false alternative below a satisfied OR is not a rule
    violation.  Treating every low-valued OR leaf as anomalous used to make
    mutually exclusive normal variants all paint the logic heat-map.
    """
    if node.op == "LEAF":
        return (
            [node.atom_id]
            if _leaf_truth(node.atom_id, atom_results) < fail_threshold
            else []
        )

    if node.op == "OR":
        value = _evaluate(node, atom_results, epsilon=epsilon, trace=[])
        if value >= fail_threshold:
            return []

    if node.op == "NOT":
        # Atomic responsibility maps describe how a positive predicate fails.
        # They cannot faithfully localise a NOT violation, so do not project
        # the child as if it were an ordinary failed positive predicate.
        return []

    active: List[str] = []
    for child in node.children:
        active.extend(
            _active_failed_atom_ids(
                child,
                atom_results,
                fail_threshold=fail_threshold,
                epsilon=epsilon,
            )
        )
    return list(dict.fromkeys(active))


def execute_formula(
    formula: LogicNode,
    atom_results: Mapping[str, AtomicResult],
    fail_threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> LogicResult:
    trace: List[Dict[str, object]] = []
    root_truth = _evaluate(formula, atom_results, epsilon=epsilon, trace=trace)
    active_failed_atom_ids = _active_failed_atom_ids(
        formula,
        atom_results,
        fail_threshold=float(fail_threshold),
        epsilon=epsilon,
    )
    active_failed = set(active_failed_atom_ids)

    failed = []
    for atom_id in formula.atom_ids():
        result = atom_results.get(atom_id)
        if result is None or not result.valid or result.weight <= 0.0:
            continue
        if atom_id in active_failed:
            failed.append(
                {
                    "id": atom_id,
                    "expression": result.expression,
                    "truth": float(result.truth),
                    "violation": float(result.violation),
                }
            )
    failed.sort(key=lambda item: float(item["truth"]))

    # Serialise node traces without embedding LogicNode objects.
    serial_trace = []
    for item in trace:
        if "children" in item:
            serial_trace.append(
                {
                    "op": item["op"],
                    "truth": item["truth"],
                    "children": [child.op for child in item["children"]],
                }
            )
        else:
            serial_trace.append(item)

    return LogicResult(
        root_truth=float(root_truth),
        logic_score=float(1.0 - root_truth),
        failed_atoms=failed,
        atom_truths={
            atom_id: float(_leaf_truth(atom_id, atom_results))
            for atom_id in formula.atom_ids()
        },
        active_failed_atom_ids=active_failed_atom_ids,
        node_trace={"nodes": serial_trace},
    )
