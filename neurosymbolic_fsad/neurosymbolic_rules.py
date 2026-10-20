"""Explicit neuro-symbolic rule combination engine.

This module provides a way to define and execute explicit rule combinations
like "rule1 ^ rule2" (XOR), "rule1 & rule2" (AND), "rule1 | rule2" (OR)
on top of the baseline anomaly detector. The idea is that if a query image
violates certain rule combinations, the anomaly probability increases.

This is similar to the SymbolicDet approach where symbolic rules are explicitly
combined and executed for anomaly detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .fuzzy_logic import (
    TruthValue,
    fuzzy_and,
    fuzzy_not,
    fuzzy_or,
    fuzzy_xor,
    fuzzy_implies,
)


@dataclass(frozen=True)
class NeuroSymbolicRule:
    """A single neuro-symbolic rule with its ground truth value."""
    rule_id: str
    description: str
    truth_value: float  # 1.0 = normal (rule satisfied), 0.0 = violated
    spatial_map: Optional[np.ndarray] = None  # 2D map of rule satisfaction
    weight: float = 1.0


@dataclass(frozen=True)
class RuleCombination:
    """Explicit rule combination like 'rule1 ^ rule2' (XOR)."""
    name: str
    operator: str  # "XOR", "AND", "OR", "IMPLIES", "NOT"
    rule_ids: List[str]
    description: str = ""
    weight: float = 1.0


def _evaluate_combination(
    combo: RuleCombination,
    rule_truths: Mapping[str, float],
) -> float:
    """Evaluate a single rule combination."""
    values = [rule_truths.get(rid, 1.0) for rid in combo.rule_ids]
    
    if combo.operator == "NOT":
        if len(values) != 1:
            raise ValueError(f"NOT requires exactly one rule")
        return float(fuzzy_not(values[0]))
    elif combo.operator == "XOR":
        if len(values) != 2:
            raise ValueError(f"XOR requires exactly two rules")
        return float(fuzzy_xor(values[0], values[1]))
    elif combo.operator == "AND":
        if len(values) < 2:
            raise ValueError(f"AND requires at least two rules")
        return float(fuzzy_and(*values))
    elif combo.operator == "OR":
        if len(values) < 2:
            raise ValueError(f"OR requires at least two rules")
        return float(fuzzy_or(*values))
    elif combo.operator == "IMPLIES":
        if len(values) != 2:
            raise ValueError(f"IMPLIES requires exactly two rules")
        return float(fuzzy_implies(values[0], values[1]))
    else:
        raise ValueError(f"Unknown operator: {combo.operator}")


def _evaluate_combination_map(
    combo: RuleCombination,
    rule_maps: Mapping[str, np.ndarray],
    default_value: float = 1.0,
) -> np.ndarray:
    """Evaluate a single rule combination on spatial maps."""
    maps = [rule_maps.get(rid, np.full_like(next(iter(rule_maps.values())), default_value)) for rid in combo.rule_ids]
    
    if combo.operator == "NOT":
        return fuzzy_not(maps[0])
    elif combo.operator == "XOR":
        return fuzzy_xor(maps[0], maps[1])
    elif combo.operator == "AND":
        return fuzzy_and(*maps)
    elif combo.operator == "OR":
        return fuzzy_or(*maps)
    elif combo.operator == "IMPLIES":
        return fuzzy_implies(maps[0], maps[1])
    else:
        raise ValueError(f"Unknown operator: {combo.operator}")


class NeuroSymbolicEngine:
    """Engine for executing explicit neuro-symbolic rule combinations.

    This engine takes grounded rule truths (from visual grounders) and executes
    explicit fuzzy logic combinations like:
    - rule1 ^ rule2 (XOR): exactly one rule violated
    - rule1 & rule2 (AND): both rules violated
    - rule1 | rule2 (OR): at least one rule violated
    - rule1 -> rule2 (IMPLIES): if rule1 violated then rule2 violated
    """

    def __init__(
        self,
        rules: Sequence[NeuroSymbolicRule],
        combinations: Sequence[RuleCombination],
        *,
        baseline_weight: float = 0.5,
    ):
        self.rules = {rule.rule_id: rule for rule in rules}
        self.combinations = list(combinations)
        self.baseline_weight = baseline_weight
        self.used_rule_ids = self._collect_used_rules()

    def _collect_used_rules(self) -> List[str]:
        """Collect all rule IDs that are actually used in combinations."""
        used = set()
        for combo in self.combinations:
            used.update(combo.rule_ids)
        return sorted(used)

    def evaluate(
        self,
        rule_truths: Mapping[str, float],
        neural_anomaly: float,
        grounding_confidence: float = 1.0,
        rule_maps: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Evaluate the neuro-symbolic combinations.

        Args:
            rule_truths: Dict mapping rule_id -> truth value (1.0=normal, 0.0=violated)
            neural_anomaly: Baseline neural anomaly score [0,1]
            grounding_confidence: Confidence in visual grounding [0,1]
            rule_maps: Optional spatial maps for each rule

        Returns:
            Dict with final anomaly score, symbolic score, trace, etc.
        """
        if not self.combinations:
            return {
                "final_score": neural_anomaly,
                "symbolic_score": 0.0,
                "neural_anomaly": neural_anomaly,
                "rule_truths": dict(rule_truths),
                "violated_rules": [],
                "trace": [],
                "spatial_map": None,
                "combination_scores": {},
            }

        # Evaluate each combination
        combination_scores = {}
        for combo in self.combinations:
            score = _evaluate_combination(combo, rule_truths)
            combination_scores[combo.name] = score

        # Symbolic anomaly: OR of all combination VIOLATION scores
        # For XOR: score is already violation (high when exactly one violated)
        # For AND: score is violation (high when both violated)
        # For OR: score is violation (high when at least one violated)
        # For IMPLIES: score is violation (high when antecedent true but consequent false)
        # For NOT: score is violation (high when rule violated)
        # So we use the scores directly as violation measures
        symbolic_violations = list(combination_scores.values())
        if symbolic_violations:
            symbolic_anomaly = float(fuzzy_or(*symbolic_violations))
        else:
            symbolic_anomaly = 0.0

        # Final neuro-symbolic fusion: OR(neural, AND(grounding, symbolic_anomaly))
        grounded_symbolic = fuzzy_and(grounding_confidence, symbolic_anomaly)
        final_score = float(fuzzy_or(neural_anomaly, grounded_symbolic))

        # Spatial map evaluation
        spatial_map = None
        if rule_maps and self.used_rule_ids:
            first_map = next((rule_maps[rid] for rid in self.used_rule_ids if rid in rule_maps), None)
            if first_map is not None:
                shape = first_map.shape
                map_values = {
                    "neural": np.full(shape, neural_anomaly, dtype=np.float32),
                    "grounding": np.full(shape, grounding_confidence, dtype=np.float32),
                }
                for rid in self.used_rule_ids:
                    if rid in rule_maps:
                        map_values[rid] = rule_maps[rid]
                    else:
                        map_values[rid] = np.full(shape, rule_truths.get(rid, 1.0), dtype=np.float32)

                # Evaluate each combination on maps
                combo_maps = {}
                for combo in self.combinations:
                    combo_maps[combo.name] = _evaluate_combination_map(combo, map_values)

                # Symbolic anomaly map: OR of combination violation maps
                symbolic_violation_maps = list(combo_maps.values())
                if symbolic_violation_maps:
                    symbolic_anomaly_map = fuzzy_or(*symbolic_violation_maps)
                else:
                    symbolic_anomaly_map = np.zeros(shape, dtype=np.float32)

                # Final fusion map
                grounded_symbolic_map = fuzzy_and(map_values["grounding"], symbolic_anomaly_map)
                final_map = fuzzy_or(map_values["neural"], grounded_symbolic_map)
                spatial_map = np.asarray(final_map, dtype=np.float32)

        violated_rules = [
            rule_id for rule_id, truth in rule_truths.items()
            if truth < 1.0 - 1e-8
        ]

        # Build a simple trace for debugging
        trace = []
        for combo in self.combinations:
            trace.append({
                "node_id": f"combo.{combo.name}",
                "op": combo.operator,
                "rule_ids": combo.rule_ids,
                "output": combination_scores[combo.name],
            })
        trace.append({
            "node_id": "symbolic.anomaly",
            "op": "OR",
            "input_node_ids": [f"combo.{c.name}" for c in self.combinations],
            "output": symbolic_anomaly,
        })
        trace.append({
            "node_id": "final.grounded_symbolic",
            "op": "AND",
            "input_node_ids": ["final.atom.grounding", "symbolic.anomaly"],
            "output": float(grounded_symbolic),
        })
        trace.append({
            "node_id": "final.neuro_symbolic_anomaly",
            "op": "OR",
            "input_node_ids": ["final.atom.neural", "final.grounded_symbolic"],
            "output": final_score,
        })

        return {
            "final_score": final_score,
            "symbolic_score": symbolic_anomaly,
            "neural_anomaly": neural_anomaly,
            "rule_truths": dict(rule_truths),
            "violated_rules": violated_rules,
            "trace": trace,
            "spatial_map": spatial_map,
            "combination_scores": combination_scores,
        }


def create_xor_combinations(
    rule_ids: Sequence[str],
    descriptions: Optional[Mapping[str, str]] = None,
) -> List[RuleCombination]:
    """Create XOR combinations for all pairs of rules.

    This creates combinations like rule1 ^ rule2, rule1 ^ rule3, etc.
    Useful for detecting anomalies where exactly one rule is violated.
    """
    descriptions = descriptions or {}
    combinations = []
    for i, rule1 in enumerate(rule_ids):
        for rule2 in rule_ids[i+1:]:
            desc = descriptions.get(f"{rule1}^{rule2}", f"XOR({rule1}, {rule2})")
            combinations.append(RuleCombination(
                name=f"xor_{rule1}_{rule2}",
                operator="XOR",
                rule_ids=[rule1, rule2],
                description=desc,
            ))
    return combinations


def create_and_combinations(
    rule_ids: Sequence[str],
    descriptions: Optional[Mapping[str, str]] = None,
) -> List[RuleCombination]:
    """Create AND combinations for all pairs/triples of rules."""
    descriptions = descriptions or {}
    combinations = []
    for i, rule1 in enumerate(rule_ids):
        for rule2 in rule_ids[i+1:]:
            desc = descriptions.get(f"{rule1}&{rule2}", f"AND({rule1}, {rule2})")
            combinations.append(RuleCombination(
                name=f"and_{rule1}_{rule2}",
                operator="AND",
                rule_ids=[rule1, rule2],
                description=desc,
            ))
    return combinations


def create_implication_chain(
    rule_ids: Sequence[str],
    descriptions: Optional[Mapping[str, str]] = None,
) -> List[RuleCombination]:
    """Create implication chain: rule1 -> rule2 -> rule3 -> ..."""
    descriptions = descriptions or {}
    combinations = []
    for i in range(len(rule_ids) - 1):
        desc = descriptions.get(f"{rule_ids[i]}->{rule_ids[i+1]}",
                               f"IMPLIES({rule_ids[i]}, {rule_ids[i+1]})")
        combinations.append(RuleCombination(
            name=f"implies_{rule_ids[i]}_{rule_ids[i+1]}",
            operator="IMPLIES",
            rule_ids=[rule_ids[i], rule_ids[i+1]],
            description=desc,
        ))
    return combinations


@dataclass
class NeuroSymbolicResult:
    """Result of neuro-symbolic evaluation."""
    final_score: float
    symbolic_score: float
    neural_anomaly: float
    rule_truths: Dict[str, float]
    violated_rules: List[str]
    trace: List[Dict[str, Any]]
    spatial_map: Optional[np.ndarray]
    combination_scores: Dict[str, float] = field(default_factory=dict)


def execute_neuro_symbolic(
    rules: Sequence[NeuroSymbolicRule],
    combinations: Sequence[RuleCombination],
    neural_anomaly: float,
    grounding_confidence: float = 1.0,
    baseline_weight: float = 0.5,
    rule_maps: Optional[Mapping[str, np.ndarray]] = None,
) -> NeuroSymbolicResult:
    """Convenience function to execute neuro-symbolic rule combinations."""
    rule_dict = {r.rule_id: r for r in rules}
    engine = NeuroSymbolicEngine(rules, combinations, baseline_weight=baseline_weight)
    result = engine.evaluate(
        {r.rule_id: r.truth_value for r in rules},
        neural_anomaly,
        grounding_confidence,
        rule_maps={r.rule_id: r.spatial_map for r in rules if r.spatial_map is not None} if rule_maps else None,
    )

    combo_scores = {}
    for combo in combinations:
        if combo.operator == "XOR":
            v1 = 1.0 - rule_dict[combo.rule_ids[0]].truth_value
            v2 = 1.0 - rule_dict[combo.rule_ids[1]].truth_value
            combo_scores[combo.name] = float(v1 + v2 - 2.0 * v1 * v2)
        elif combo.operator == "AND":
            vals = [1.0 - rule_dict[rid].truth_value for rid in combo.rule_ids]
            prod = 1.0
            for v in vals:
                prod *= v
            combo_scores[combo.name] = float(prod)
        elif combo.operator == "OR":
            vals = [1.0 - rule_dict[rid].truth_value for rid in combo.rule_ids]
            prod = 1.0
            for v in vals:
                prod *= (1.0 - v)
            combo_scores[combo.name] = float(1.0 - prod)
        elif combo.operator == "IMPLIES":
            v1 = 1.0 - rule_dict[combo.rule_ids[0]].truth_value
            v2 = 1.0 - rule_dict[combo.rule_ids[1]].truth_value
            combo_scores[combo.name] = float(1.0 - v1 * (1.0 - v2))

    return NeuroSymbolicResult(
        final_score=result["final_score"],
        symbolic_score=result["symbolic_score"],
        neural_anomaly=result["neural_anomaly"],
        rule_truths=result["rule_truths"],
        violated_rules=result["violated_rules"],
        trace=result["trace"],
        spatial_map=result["spatial_map"],
        combination_scores=combo_scores,
    )


__all__ = [
    "NeuroSymbolicRule",
    "RuleCombination",
    "NeuroSymbolicEngine",
    "NeuroSymbolicResult",
    "execute_neuro_symbolic",
    "create_xor_combinations",
    "create_and_combinations",
    "create_implication_chain",
]