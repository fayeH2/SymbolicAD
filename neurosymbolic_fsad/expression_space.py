"""
Arithmetic-Logic Expression Space for SymbolicDet-style Neuro-Symbolic Reasoning.

This module defines the search space of symbolic expressions that can be evolved
to detect anomalies. Unlike pure fuzzy logic, this supports:
- Arithmetic operations: +, -, *, /, max, min
- Comparisons: >, <, >=, <=, ==, !=
- Logic: AND, OR, NOT, IMPLIES, XOR
- Variables: grounded rule truths, counts, ratios, raw features
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union
import numpy as np
import random


class ExprType(Enum):
    """Types of expression nodes."""
    VARIABLE = "variable"      # Leaf: grounded rule truth, count, ratio, feature
    CONSTANT = "constant"      # Leaf: numeric constant
    ARITHMETIC = "arithmetic"  # +, -, *, /, max, min
    COMPARISON = "comparison"  # >, <, >=, <=, ==, !=
    LOGIC = "logic"            # AND, OR, NOT, IMPLIES, XOR


class ArithmeticOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MAX = "max"
    MIN = "min"


class ComparisonOp(Enum):
    GT = ">"
    LT = "<"
    GE = ">="
    LE = "<="
    EQ = "=="
    NE = "!="


class LogicOp(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    XOR = "XOR"


@dataclass(frozen=True)
class ExprNode:
    """A node in the expression tree."""
    expr_type: ExprType
    op: Optional[Union[ArithmeticOp, ComparisonOp, LogicOp]] = None
    name: Optional[str] = None          # For VARIABLE
    value: Optional[float] = None       # For CONSTANT
    children: Tuple["ExprNode", ...] = field(default_factory=tuple)
    
    def __str__(self) -> str:
        if self.expr_type == ExprType.VARIABLE:
            return self.name or "?"
        elif self.expr_type == ExprType.CONSTANT:
            return f"{self.value:.3f}"
        elif self.expr_type in (ExprType.ARITHMETIC, ExprType.COMPARISON):
            if len(self.children) == 2:
                return f"({self.children[0]} {self.op.value} {self.children[1]})"
            elif len(self.children) == 1:
                return f"{self.op.value}({self.children[0]})"
        elif self.expr_type == ExprType.LOGIC:
            if self.op == LogicOp.NOT:
                return f"NOT({self.children[0]})"
            elif len(self.children) == 2:
                return f"({self.children[0]} {self.op.value} {self.children[1]})"
            else:
                args = " ".join(str(c) for c in self.children)
                return f"{self.op.value}({args})"
        return "?"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {"type": self.expr_type.value}
        if self.op:
            result["op"] = self.op.value
        if self.name:
            result["name"] = self.name
        if self.value is not None:
            result["value"] = self.value
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExprNode":
        """Deserialize from dictionary."""
        expr_type = ExprType(data["type"])
        op = None
        if "op" in data:
            op_str = data["op"]
            if op_str in [o.value for o in ArithmeticOp]:
                op = ArithmeticOp(op_str)
            elif op_str in [o.value for o in ComparisonOp]:
                op = ComparisonOp(op_str)
            elif op_str in [o.value for o in LogicOp]:
                op = LogicOp(op_str)
        children = tuple(cls.from_dict(c) for c in data.get("children", []))
        return cls(
            expr_type=expr_type,
            op=op,
            name=data.get("name"),
            value=data.get("value"),
            children=children,
        )

    def get_variables(self) -> Set[str]:
        """Get all variable names used in this expression."""
        vars_set = set()
        if self.expr_type == ExprType.VARIABLE and self.name:
            vars_set.add(self.name)
        for child in self.children:
            vars_set.update(child.get_variables())
        return vars_set

    def depth(self) -> int:
        """Get tree depth."""
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        """Get number of nodes."""
        return 1 + sum(c.size() for c in self.children)


# Factory functions for building expressions
def var(name: str) -> ExprNode:
    return ExprNode(ExprType.VARIABLE, name=name)

def const(value: float) -> ExprNode:
    return ExprNode(ExprType.CONSTANT, value=value)

def add(*args: ExprNode) -> ExprNode:
    if len(args) == 1:
        return args[0]
    result = args[0]
    for arg in args[1:]:
        result = ExprNode(ExprType.ARITHMETIC, op=ArithmeticOp.ADD, children=(result, arg))
    return result

def sub(a: ExprNode, b: ExprNode) -> ExprNode:
    return ExprNode(ExprType.ARITHMETIC, op=ArithmeticOp.SUB, children=(a, b))

def mul(*args: ExprNode) -> ExprNode:
    if len(args) == 1:
        return args[0]
    result = args[0]
    for arg in args[1:]:
        result = ExprNode(ExprType.ARITHMETIC, op=ArithmeticOp.MUL, children=(result, arg))
    return result

def div(a: ExprNode, b: ExprNode) -> ExprNode:
    return ExprNode(ExprType.ARITHMETIC, op=ArithmeticOp.DIV, children=(a, b))

def max_op(*args: ExprNode) -> ExprNode:
    if len(args) == 1:
        return args[0]
    result = args[0]
    for arg in args[1:]:
        result = ExprNode(ExprType.ARITHMETIC, op=ArithmeticOp.MAX, children=(result, arg))
    return result

def min_op(*args: ExprNode) -> ExprNode:
    if len(args) == 1:
        return args[0]
    result = args[0]
    for arg in args[1:]:
        result = ExprNode(ExprType.ARITHMETIC, op=ArithmeticOp.MIN, children=(result, arg))
    return result

def gt(a: ExprNode, b: ExprNode) -> ExprNode:
    return ExprNode(ExprType.COMPARISON, op=ComparisonOp.GT, children=(a, b))

def lt(a: ExprNode, b: ExprNode) -> ExprNode:
    return ExprNode(ExprType.COMPARISON, op=ComparisonOp.LT, children=(a, b))

def ge(a: ExprNode, b: ExprNode) -> ExprNode:
    return ExprNode(ExprType.COMPARISON, op=ComparisonOp.GE, children=(a, b))

def le(a: ExprNode, b: ExprNode) -> ExprNode:
    return ExprNode(ExprType.COMPARISON, op=ComparisonOp.LE, children=(a, b))

def eq(a: ExprNode, b: ExprNode) -> ExprNode:
    return ExprNode(ExprType.COMPARISON, op=ComparisonOp.EQ, children=(a, b))

def ne(a: ExprNode, b: ExprNode) -> ExprNode:
    return ExprNode(ExprType.COMPARISON, op=ComparisonOp.NE, children=(a, b))

def logic_and(*args: ExprNode) -> ExprNode:
    if len(args) == 1:
        return args[0]
    result = args[0]
    for arg in args[1:]:
        result = ExprNode(ExprType.LOGIC, op=LogicOp.AND, children=(result, arg))
    return result

def logic_or(*args: ExprNode) -> ExprNode:
    if len(args) == 1:
        return args[0]
    result = args[0]
    for arg in args[1:]:
        result = ExprNode(ExprType.LOGIC, op=LogicOp.OR, children=(result, arg))
    return result

def logic_not(a: ExprNode) -> ExprNode:
    return ExprNode(ExprType.LOGIC, op=LogicOp.NOT, children=(a,))

def logic_implies(a: ExprNode, b: ExprNode) -> ExprNode:
    return ExprNode(ExprType.LOGIC, op=LogicOp.IMPLIES, children=(a, b))

def logic_xor(a: ExprNode, b: ExprNode) -> ExprNode:
    return ExprNode(ExprType.LOGIC, op=LogicOp.XOR, children=(a, b))


# Expression Evaluator
class ExpressionEvaluator:
    """Evaluates arithmetic-logic expressions on grounded values."""
    
    def __init__(self, variable_values: Mapping[str, float]):
        self.variables = dict(variable_values)
        # Add safe constants
        self.variables.setdefault("pi", np.pi)
        self.variables.setdefault("e", np.e)
    
    def evaluate(self, expr: ExprNode) -> float:
        """Evaluate expression to a float value."""
        if expr.expr_type == ExprType.VARIABLE:
            if expr.name not in self.variables:
                raise ValueError(f"Variable not found: {expr.name}")
            return float(self.variables[expr.name])
        elif expr.expr_type == ExprType.CONSTANT:
            return float(expr.value or 0.0)
        elif expr.expr_type == ExprType.ARITHMETIC:
            return self._eval_arithmetic(expr)
        elif expr.expr_type == ExprType.COMPARISON:
            return self._eval_comparison(expr)
        elif expr.expr_type == ExprType.LOGIC:
            return self._eval_logic(expr)
        else:
            raise ValueError(f"Unknown expression type: {expr.expr_type}")
    
    def _eval_arithmetic(self, expr: ExprNode) -> float:
        vals = [self.evaluate(c) for c in expr.children]
        if expr.op == ArithmeticOp.ADD:
            return sum(vals)
        elif expr.op == ArithmeticOp.SUB:
            return vals[0] - vals[1]
        elif expr.op == ArithmeticOp.MUL:
            result = 1.0
            for v in vals:
                result *= v
            return result
        elif expr.op == ArithmeticOp.DIV:
            denom = vals[1]
            return vals[0] / denom if abs(denom) > 1e-8 else 1e8 * np.sign(vals[0])
        elif expr.op == ArithmeticOp.MAX:
            return max(vals)
        elif expr.op == ArithmeticOp.MIN:
            return min(vals)
        raise ValueError(f"Unknown arithmetic op: {expr.op}")
    
    def _eval_comparison(self, expr: ExprNode) -> float:
        a, b = self.evaluate(expr.children[0]), self.evaluate(expr.children[1])
        if expr.op == ComparisonOp.GT:
            return 1.0 if a > b else 0.0
        elif expr.op == ComparisonOp.LT:
            return 1.0 if a < b else 0.0
        elif expr.op == ComparisonOp.GE:
            return 1.0 if a >= b else 0.0
        elif expr.op == ComparisonOp.LE:
            return 1.0 if a <= b else 0.0
        elif expr.op == ComparisonOp.EQ:
            return 1.0 if abs(a - b) < 1e-6 else 0.0
        elif expr.op == ComparisonOp.NE:
            return 1.0 if abs(a - b) >= 1e-6 else 0.0
        raise ValueError(f"Unknown comparison op: {expr.op}")
    
    def _eval_logic(self, expr: ExprNode) -> float:
        vals = [self.evaluate(c) for c in expr.children]
        if expr.op == LogicOp.NOT:
            return 1.0 - vals[0]
        elif expr.op == LogicOp.AND:
            return min(vals)
        elif expr.op == LogicOp.OR:
            return max(vals)
        elif expr.op == LogicOp.IMPLIES:
            return 1.0 - vals[0] * (1.0 - vals[1])
        elif expr.op == LogicOp.XOR:
            return vals[0] + vals[1] - 2.0 * vals[0] * vals[1]
        raise ValueError(f"Unknown logic op: {expr.op}")


# Fuzzy Logic Evaluator (for soft logic)
class FuzzyExpressionEvaluator:
    """Evaluates expressions with fuzzy logic semantics (values in [0,1])."""
    
    def __init__(self, variable_values: Mapping[str, float]):
        self.variables = dict(variable_values)
    
    def evaluate(self, expr: ExprNode) -> float:
        """Evaluate with fuzzy semantics, returning value in [0,1]."""
        if expr.expr_type == ExprType.VARIABLE:
            if expr.name not in self.variables:
                raise ValueError(f"Variable not found: {expr.name}")
            return np.clip(float(self.variables[expr.name]), 0.0, 1.0)
        elif expr.expr_type == ExprType.CONSTANT:
            return np.clip(float(expr.value or 0.0), 0.0, 1.0)
        elif expr.expr_type == ExprType.ARITHMETIC:
            return self._eval_arithmetic_fuzzy(expr)
        elif expr.expr_type == ExprType.COMPARISON:
            return self._eval_comparison_fuzzy(expr)
        elif expr.expr_type == ExprType.LOGIC:
            return self._eval_logic_fuzzy(expr)
        else:
            raise ValueError(f"Unknown expression type: {expr.expr_type}")
    
    def _eval_arithmetic_fuzzy(self, expr: ExprNode) -> float:
        vals = [np.clip(self.evaluate(c), 0.0, 1.0) for c in expr.children]
        if expr.op == ArithmeticOp.ADD:
            return np.clip(sum(vals), 0.0, 1.0)
        elif expr.op == ArithmeticOp.SUB:
            return np.clip(vals[0] - vals[1], 0.0, 1.0)
        elif expr.op == ArithmeticOp.MUL:
            result = 1.0
            for v in vals:
                result *= v
            return result
        elif expr.op == ArithmeticOp.DIV:
            denom = vals[1]
            return np.clip(vals[0] / denom if denom > 1e-8 else 1.0, 0.0, 1.0)
        elif expr.op == ArithmeticOp.MAX:
            return max(vals)
        elif expr.op == ArithmeticOp.MIN:
            return min(vals)
        raise ValueError(f"Unknown arithmetic op: {expr.op}")
    
    def _eval_comparison_fuzzy(self, expr: ExprNode) -> float:
        """Soft comparison using sigmoid."""
        a, b = self.evaluate(expr.children[0]), self.evaluate(expr.children[1])
        diff = a - b
        # Soft comparison: sigmoid(k * diff) where k controls sharpness
        k = 20.0
        sigmoid = 1.0 / (1.0 + np.exp(-k * diff))
        if expr.op == ComparisonOp.GT:
            return sigmoid
        elif expr.op == ComparisonOp.LT:
            return 1.0 - sigmoid
        elif expr.op == ComparisonOp.GE:
            return sigmoid
        elif expr.op == ComparisonOp.LE:
            return 1.0 - sigmoid
        elif expr.op == ComparisonOp.EQ:
            return 1.0 - 2.0 * abs(sigmoid - 0.5)
        elif expr.op == ComparisonOp.NE:
            return 2.0 * abs(sigmoid - 0.5)
        raise ValueError(f"Unknown comparison op: {expr.op}")
    
    def _eval_logic_fuzzy(self, expr: ExprNode) -> float:
        vals = [np.clip(self.evaluate(c), 0.0, 1.0) for c in expr.children]
        if expr.op == LogicOp.NOT:
            return 1.0 - vals[0]
        elif expr.op == LogicOp.AND:
            return np.prod(vals)
        elif expr.op == LogicOp.OR:
            return 1.0 - np.prod([1.0 - v for v in vals])
        elif expr.op == LogicOp.IMPLIES:
            return 1.0 - vals[0] * (1.0 - vals[1])
        elif expr.op == LogicOp.XOR:
            return vals[0] + vals[1] - 2.0 * vals[0] * vals[1]
        raise ValueError(f"Unknown logic op: {expr.op}")


# Expression Generator for Evolutionary Search
class ExpressionGenerator:
    """Generates random expressions for evolutionary search initialization."""
    
    def __init__(
        self,
        variable_names: Sequence[str],
        constants: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 3.0),
        max_depth: int = 4,
        max_arity: int = 3,
    ):
        self.variable_names = list(variable_names)
        self.constants = list(constants)
        self.max_depth = max_depth
        self.max_arity = max_arity
    
    def random_expression(self, depth: int = 0) -> ExprNode:
        """Generate a random expression."""
        if depth >= self.max_depth or (depth > 0 and random.random() < 0.3):
            # Leaf node
            if random.random() < 0.7 and self.variable_names:
                return var(random.choice(self.variable_names))
            else:
                return const(random.choice(self.constants))
        
        # Internal node - choose operation type
        op_type = random.choice([ExprType.ARITHMETIC, ExprType.COMPARISON, ExprType.LOGIC])
        
        if op_type == ExprType.ARITHMETIC:
            op = random.choice(list(ArithmeticOp))
            arity = 2 if op in (ArithmeticOp.ADD, ArithmeticOp.MUL, ArithmeticOp.MAX, ArithmeticOp.MIN) else 2
            children = [self.random_expression(depth + 1) for _ in range(arity)]
            return ExprNode(ExprType.ARITHMETIC, op=op, children=tuple(children))
        
        elif op_type == ExprType.COMPARISON:
            op = random.choice(list(ComparisonOp))
            children = [self.random_expression(depth + 1) for _ in range(2)]
            return ExprNode(ExprType.COMPARISON, op=op, children=tuple(children))
        
        else:  # LOGIC
            op = random.choice(list(LogicOp))
            if op == LogicOp.NOT:
                children = [self.random_expression(depth + 1)]
            else:
                arity = random.randint(2, self.max_arity)
                children = [self.random_expression(depth + 1) for _ in range(arity)]
            return ExprNode(ExprType.LOGIC, op=op, children=tuple(children))
    
    def random_population(self, size: int) -> List[ExprNode]:
        """Generate initial population."""
        return [self.random_expression() for _ in range(size)]


# Expression Mutator
class ExpressionMutator:
    """Mutates expressions for evolutionary search."""
    
    def __init__(
        self,
        variable_names: Sequence[str],
        constants: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 3.0),
        max_depth: int = 4,
    ):
        self.variable_names = list(variable_names)
        self.constants = list(constants)
        self.max_depth = max_depth
        self.generator = ExpressionGenerator(variable_names, constants, max_depth)
    
    def mutate(self, expr: ExprNode, mutation_rate: float = 0.3) -> ExprNode:
        """Apply mutation to expression."""
        if random.random() > mutation_rate:
            return expr
        
        mutation_type = random.choice(["point", "subtree", "hoist", "shrink"])
        
        if mutation_type == "point":
            return self._point_mutate(expr)
        elif mutation_type == "subtree":
            return self._subtree_mutate(expr)
        elif mutation_type == "hoist":
            return self._hoist_mutate(expr)
        else:  # shrink
            return self._shrink_mutate(expr)
    
    def _point_mutate(self, expr: ExprNode) -> ExprNode:
        """Mutate a single node."""
        if expr.expr_type == ExprType.VARIABLE and self.variable_names:
            if random.random() < 0.5:
                return var(random.choice(self.variable_names))
        elif expr.expr_type == ExprType.CONSTANT:
            if random.random() < 0.5:
                return const(random.choice(self.constants))
        elif expr.op:
            # Change operator
            if expr.expr_type == ExprType.ARITHMETIC:
                new_op = random.choice([o for o in ArithmeticOp if o != expr.op])
                return ExprNode(expr.expr_type, op=new_op, children=expr.children)
            elif expr.expr_type == ExprType.COMPARISON:
                new_op = random.choice([o for o in ComparisonOp if o != expr.op])
                return ExprNode(expr.expr_type, op=new_op, children=expr.children)
            elif expr.expr_type == ExprType.LOGIC:
                new_op = random.choice([o for o in LogicOp if o != expr.op])
                return ExprNode(expr.expr_type, op=new_op, children=expr.children)
        return expr
    
    def _subtree_mutate(self, expr: ExprNode) -> ExprNode:
        """Replace a random subtree."""
        if not expr.children or random.random() < 0.1:
            return self.generator.random_expression()
        
        new_children = []
        for child in expr.children:
            if random.random() < 0.3:
                new_children.append(self.generator.random_expression())
            else:
                new_children.append(self._subtree_mutate(child))
        return ExprNode(expr.expr_type, op=expr.op, name=expr.name, value=expr.value, children=tuple(new_children))
    
    def _hoist_mutate(self, expr: ExprNode) -> ExprNode:
        """Hoist a random subtree to root."""
        all_nodes = self._collect_nodes(expr)
        if len(all_nodes) <= 1:
            return expr
        return random.choice(all_nodes[1:])  # Skip root
    
    def _shrink_mutate(self, expr: ExprNode) -> ExprNode:
        """Replace a random subtree with a terminal."""
        if not expr.children:
            return expr
        if random.random() < 0.3:
            if self.variable_names and random.random() < 0.7:
                return var(random.choice(self.variable_names))
            return const(random.choice(self.constants))
        new_children = [self._shrink_mutate(c) for c in expr.children]
        return ExprNode(expr.expr_type, op=expr.op, name=expr.name, value=expr.value, children=tuple(new_children))
    
    def _collect_nodes(self, expr: ExprNode) -> List[ExprNode]:
        nodes = [expr]
        for child in expr.children:
            nodes.extend(self._collect_nodes(child))
        return nodes


# Expression Crossover
class ExpressionCrossover:
    """Performs crossover between two expressions."""
    
    def crossover(self, parent1: ExprNode, parent2: ExprNode) -> Tuple[ExprNode, ExprNode]:
        """Swap random subtrees between parents."""
        nodes1 = self._collect_nodes_with_path(parent1)
        nodes2 = self._collect_nodes_with_path(parent2)
        
        if not nodes1[1:] or not nodes2[1:]:  # Need at least one non-root node
            return parent1, parent2
        
        # Pick random non-root nodes
        _, path1, node1 = random.choice(nodes1[1:])
        _, path2, node2 = random.choice(nodes2[1:])
        
        # Swap
        new1 = self._replace_at_path(parent1, path1, node2)
        new2 = self._replace_at_path(parent2, path2, node1)
        
        return new1, new2
    
    def _collect_nodes_with_path(self, expr: ExprNode) -> List[Tuple[ExprNode, Tuple[int, ...], ExprNode]]:
        """Collect (parent, path, node) for all nodes."""
        result = []
        def collect(node: ExprNode, path: Tuple[int, ...], parent: Optional[ExprNode]):
            result.append((parent, path, node))
            for i, child in enumerate(node.children):
                collect(child, path + (i,), node)
        collect(expr, (), None)
        return result
    
    def _replace_at_path(self, expr: ExprNode, path: Tuple[int, ...], new_node: ExprNode) -> ExprNode:
        if not path:
            return new_node
        idx = path[0]
        new_children = list(expr.children)
        new_children[idx] = self._replace_at_path(expr.children[idx], path[1:], new_node)
        return ExprNode(expr.expr_type, op=expr.op, name=expr.name, value=expr.value, children=tuple(new_children))


__all__ = [
    "ExprType", "ArithmeticOp", "ComparisonOp", "LogicOp",
    "ExprNode", "ExpressionEvaluator", "FuzzyExpressionEvaluator",
    "ExpressionGenerator", "ExpressionMutator", "ExpressionCrossover",
    "var", "const", "add", "sub", "mul", "div", "max_op", "min_op",
    "gt", "lt", "ge", "le", "eq", "ne",
    "logic_and", "logic_or", "logic_not", "logic_implies", "logic_xor",
]