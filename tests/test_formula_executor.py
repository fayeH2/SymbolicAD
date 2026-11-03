import numpy as np

from symbolicAD.rules.executor import AtomicResult, execute_formula
from symbolicAD.rules.schema import LogicNode


def _result(atom_id, truth, weight=1.0, valid=True):
    return AtomicResult(
        atom_id=atom_id,
        expression=f"Test({atom_id})",
        kind="Exist",
        truth=float(truth),
        violation=float(1.0 - truth),
        responsibility_map=np.zeros((2, 2), dtype=np.float32),
        valid=valid,
        weight=weight,
        evidence={},
    )


def test_and_uses_geometric_mean():
    results = {"A": _result("A", 1.0), "B": _result("B", 0.25)}
    formula = LogicNode(op="AND", children=[LogicNode.leaf("A"), LogicNode.leaf("B")])
    out = execute_formula(formula, results)
    assert abs(out.root_truth - 0.5) < 1e-5
    assert out.failed_atoms[0]["id"] == "B"


def test_or_and_not():
    results = {"A": _result("A", 0.9), "B": _result("B", 0.2)}
    or_node = LogicNode(op="OR", children=[LogicNode.leaf("A"), LogicNode.leaf("B")])
    expected_or = 1.0 - ((1.0 - 0.9) * (1.0 - 0.2)) ** 0.5
    assert abs(execute_formula(or_node, results).root_truth - expected_or) < 1e-5
    not_node = LogicNode(op="NOT", children=[LogicNode.leaf("B")])
    assert abs(execute_formula(not_node, results).root_truth - 0.8) < 1e-5


def test_disabled_atom_is_neutral():
    results = {"A": _result("A", 0.0, weight=0.0, valid=False)}
    formula = LogicNode(op="AND", children=[LogicNode.leaf("A")])
    assert execute_formula(formula, results).root_truth == 1.0
