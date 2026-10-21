"""Validation-Driven Rule Search for SymbolicAD.

This replaces the NSGA-II search with a validation-driven approach:
1. Rules are validated on normal supports first (must hold)
2. Counterfactual anomalies are generated to test discriminative power
3. Rule dependencies are tracked to avoid redundancy
4. Greedy coverage selection with dependency awareness
5. No artificial rewards - only real validation metrics
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Set
import numpy as np

from neurosymbolic_fsad.rules.generation.dsl import NormalityRule, RuleSet, RuleType
from neurosymbolic_fsad.perception.scene_graph import SceneGraph
from neurosymbolic_fsad.perception.prototypes import SemanticPrototypeBank


@dataclass
class RuleCandidate:
    """A candidate rule in the search population."""
    rule: NormalityRule
    # Validation metrics on normal supports
    support_hold_rate: float = 0.0
    # Discriminative power on counterfactual anomalies
    anomaly_detection_auroc: float = 0.5
    anomaly_detection_aupr: float = 0.5
    # Dependency analysis
    depends_on: Set[str] = field(default_factory=set)
    implied_by: Set[str] = field(default_factory=set)
    # Redundancy with other rules
    redundancy_score: float = 0.0
    # Overall fitness
    fitness: float = 0.0
    generation: int = 0
    parent_ids: Tuple[str, ...] = ()
    
    def __lt__(self, other: "RuleCandidate") -> bool:
        return self.fitness < other.fitness


@dataclass
class SearchConfig:
    """Configuration for rule search."""
    max_candidates: int = 100
    max_iterations: int = 20
    min_support_hold_rate: float = 0.9  # Must hold on 90% of normal supports
    min_anomaly_auroc: float = 0.6      # Must discriminate anomalies
    max_redundancy: float = 0.7         # Max correlation with existing rules
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    elite_size: int = 5
    random_seed: int = 42
    
    # Weights for fitness computation
    w_support: float = 0.4
    w_anomaly: float = 0.4
    w_novelty: float = 0.2


class RuleValidator:
    """Validates rules on normal supports and counterfactual anomalies."""
    
    def __init__(
        self,
        support_graphs: List[SceneGraph],
        normal_masks: List[np.ndarray],
        anomaly_graphs: List[SceneGraph],
        anomaly_masks: List[np.ndarray],
        executor: 'RuleExecutor',
        prototype_bank: SemanticPrototypeBank,
    ):
        self.support_graphs = support_graphs
        self.normal_masks = normal_masks
        self.anomaly_graphs = anomaly_graphs
        self.anomaly_masks = anomaly_masks
        self.executor = executor
        self.prototype_bank = prototype_bank
        self.rng = np.random.RandomState(42)
    
    def validate_on_supports(self, rule: NormalityRule) -> float:
        """Check if rule holds on normal supports.
        
        Returns hold rate in [0, 1].
        """
        if not self.support_graphs:
            return 0.5
        
        hold_count = 0
        for graph in self.support_graphs:
            result = self.executor.execute(graph)
            rule_result = next((r for r in result.rule_results if r.rule_id == rule.rule_id), None)
            if rule_result and rule_result.status.name == "SATISFIED":
                hold_count += 1
        return hold_count / len(self.support_graphs)
    
    def validate_on_anomalies(self, rule: NormalityRule) -> Tuple[float, float]:
        """Test discriminative power on anomaly graphs.
        
        Returns (AUROC, AUPR) on anomaly set.
        """
        if not self.anomaly_graphs:
            return 0.5, 0.5
        
        labels = []
        scores = []
        
        for graph, mask in zip(self.anomaly_graphs, self.anomaly_masks):
            result = self.executor.execute(graph)
            rule_result = next((r for r in result.rule_results if r.rule_id == rule.rule_id), None)
            if rule_result:
                scores.append(rule_result.violation_score)
                labels.append(1 if mask.any() else 0)
        
        if not scores or len(set(labels)) < 2:
            return 0.5, 0.5
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            auroc = roc_auc_score(labels, scores)
            aupr = average_precision_score(labels, scores)
            return float(auroc), float(aupr)
        except:
            return 0.5, 0.5
    
    def check_dependencies(self, rule: NormalityRule, existing_rules: List[NormalityRule]) -> Tuple[Set[str], Set[str]]:
        """Check logical dependencies with existing rules.
        
        Returns (depends_on, implied_by).
        """
        depends_on = set()
        implied_by = set()
        
        for existing in existing_rules:
            if self._implies(rule, existing):
                depends_on.add(existing.rule_id)
            if self._implies(existing, rule):
                implied_by.add(existing.rule_id)
        
        return depends_on, implied_by
    
    def _implies(self, rule_a: NormalityRule, rule_b: NormalityRule) -> bool:
        """Check if rule_a logically implies rule_b.
        
        Simple heuristic: same predicate, tighter bounds.
        """
        if rule_a.predicate_name != rule_b.predicate_name:
            return False
        
        if rule_a.rule_type == RuleType.COUNT:
            if rule_a.predicate_args.get("entity_type") != rule_b.predicate_args.get("entity_type"):
                return False
            # a implies b if a's range is subset of b's
            a_exp = rule_a.predicate_params.get("expected_count", 0)
            a_tol = rule_a.predicate_params.get("tolerance", 0)
            b_exp = rule_b.predicate_params.get("expected_count", 0)
            b_tol = rule_b.predicate_params.get("tolerance", 0)
            a_min, a_max = a_exp - a_tol, a_exp + a_tol
            b_min, b_max = b_exp - b_tol, b_exp + b_tol
            return a_min >= b_min and a_max <= b_max
        
        return False
    
    def compute_redundancy(self, rule: NormalityRule, existing_rules: List[NormalityRule]) -> float:
        """Compute redundancy with existing rules.
        
        Returns max correlation of violation patterns.
        """
        if not existing_rules:
            return 0.0
        
        # Collect violation vectors
        rule_violations = []
        for graph in self.support_graphs + self.anomaly_graphs:
            result = self.executor.execute(graph)
            r = next((r for r in result.rule_results if r.rule_id == rule.rule_id), None)
            if r:
                rule_violations.append(r.violation_score)
        
        if not rule_violations:
            return 0.0
        
        max_corr = 0.0
        for existing in existing_rules:
            existing_violations = []
            for graph in self.support_graphs + self.anomaly_graphs:
                result = self.executor.execute(graph)
                r = next((r for r in result.rule_results if r.rule_id == existing.rule_id), None)
                if r:
                    existing_violations.append(r.violation_score)
            
            if existing_violations and len(existing_violations) == len(rule_violations):
                corr = np.corrcoef(rule_violations, existing_violations)[0, 1]
                if np.isfinite(corr):
                    max_corr = max(max_corr, abs(corr))
        
        return max_corr


class RuleDependencyTracker:
    """Tracks logical dependencies between rules."""
    
    def __init__(self):
        self.dependencies: Dict[str, Set[str]] = {}  # rule_id -> depends_on
        self.reverse_deps: Dict[str, Set[str]] = {}  # rule_id -> implied_by
        self.rule_objects: Dict[str, NormalityRule] = {}
    
    def add_rule(self, rule: NormalityRule, depends_on: Set[str], implied_by: Set[str]) -> None:
        self.rule_objects[rule.rule_id] = rule
        self.dependencies[rule.rule_id] = depends_on
        self.reverse_deps[rule.rule_id] = implied_by
        
        # Update reverse deps
        for dep in depends_on:
            self.reverse_deps.setdefault(dep, set()).add(rule.rule_id)
        for imp in implied_by:
            self.dependencies.setdefault(imp, set()).add(rule.rule_id)
    
    def get_dependencies(self, rule_id: str) -> Set[str]:
        return self.dependencies.get(rule_id, set())
    
    def get_dependents(self, rule_id: str) -> Set[str]:
        return self.reverse_deps.get(rule_id, set())
    
    def has_cycle(self, rule_id: str) -> bool:
        """Check if adding rule would create dependency cycle."""
        visited = set()
        stack = [rule_id]
        while stack:
            current = stack.pop()
            if current in visited:
                return True
            visited.add(current)
            stack.extend(self.dependencies.get(current, set()))
        return False


class ValidationDrivenSearch:
    """Validation-driven rule search with dependency awareness.
    
    Process:
    1. Generate candidates from Qwen + mutations
    2. Validate on normal supports (filter: must hold >= 90%)
    3. Test on counterfactual anomalies (filter: AUROC >= 0.6)
    4. Check dependencies and redundancy
    4. Greedy coverage selection with dependency awareness
    """
    
    def __init__(
        self,
        config: SearchConfig,
        validator: RuleValidator,
        executor: 'RuleExecutor',
    ):
        self.config = config
        self.validator = validator
        self.executor = executor
        self.dependency_tracker = RuleDependencyTracker()
        self.candidates: List[RuleCandidate] = []
        self.accepted_rules: List[NormalityRule] = []
        self.rng = np.random.RandomState(config.random_seed)
    
    def run(
        self,
        qwen_rules: List[NormalityRule],
        support_graphs: List[SceneGraph],
        anomaly_graphs: List[SceneGraph],
        support_masks: List[np.ndarray],
        anomaly_masks: List[np.ndarray],
    ) -> Tuple[RuleSet, List[Dict[str, Any]]]:
        """Run validation-driven search."""
        
        # Initialize candidates from Qwen rules
        self.candidates = [
            RuleCandidate(rule=rule, generation=0)
            for rule in qwen_rules
        ]
        
        # Add variations
        self._expand_candidates()
        
        history = []
        
        for iteration in range(self.config.max_iterations):
            # 1. Validate all candidates
            self._validate_candidates()
            
            # 2. Filter by support hold rate
            self.candidates = [
                c for c in self.candidates
                if c.support_hold_rate >= self.config.min_support_hold_rate
            ]
            
            # 3. Filter by anomaly detection
            self.candidates = [
                c for c in self.candidates
                if c.anomaly_detection_auroc >= self.config.min_anomaly_auroc
            ]
            
            # 4. Compute redundancy and dependencies
            self._compute_relations()
            
            # 5. Filter by redundancy
            self.candidates = [
                c for c in self.candidates
                if c.redundancy_score <= self.config.max_redundancy
            ]
            
            # 5. Compute fitness
            self._compute_fitness()
            
            # 6. Select best candidates
            self.candidates.sort(key=lambda c: c.fitness, reverse=True)
            self.candidates = self.candidates[:self.config.max_candidates]
            
            # 6. Greedy coverage selection for final rule set
            selected = self._greedy_coverage_selection()
            self.accepted_rules = selected
            
            # Mutation and crossover for next generation
            self._evolve_population()
            
            history.append({
                "iteration": iteration,
                "num_candidates": len(self.candidates),
                "num_accepted": len(self.accepted_rules),
                "best_fitness": max(c.fitness for c in self.candidates) if self.candidates else 0,
            })
            
            print(f"Iteration {iteration}: {len(self.candidates)} candidates, {len(self.accepted_rules)} accepted")
        
        # Final selection
        final_rules = self._greedy_coverage_selection()
        rule_set = RuleSet(category="searched", rules=final_rules)
        
        return rule_set, history
    
    def _expand_candidates(self) -> None:
        """Generate variations of initial rules."""
        new_candidates = []
        for candidate in self.candidates:
            new_candidates.append(candidate)
            # Mutate
            for _ in range(3):
                mutated = self._mutate_rule(candidate.rule)
                new_candidates.append(RuleCandidate(rule=mutated, generation=1))
        
        self.candidates = new_candidates[:self.config.max_candidates]
    
    def _mutate_rule(self, rule: NormalityRule) -> NormalityRule:
        """Create mutated copy of rule."""
        new_rule = copy.deepcopy(rule)
        new_rule.rule_id = f"{rule.rule_id}_mut_{random.randint(1000, 9999)}"
        
        mutation_type = random.choice(["parameter", "threshold", "tolerance", "part_type"])
        
        if mutation_type == "parameter" and rule.rule_type == RuleType.COUNT:
            if new_rule.expected_count is not None:
                new_rule.expected_count = max(0, new_rule.expected_count + self.rng.choice([-1, 1]))
        
        elif mutation_type == "threshold" and rule.rule_type == RuleType.SPATIAL:
            new_rule.spatial_threshold = np.clip(
                new_rule.spatial_threshold + self.rng.normal(0, 0.05),
                0.02, 0.5
            )
        
        elif mutation_type == "tolerance" and rule.rule_type == RuleType.ATTRIBUTE:
            if new_rule.tolerance is not None:
                new_rule.tolerance = max(0.01, new_rule.tolerance * self.rng.uniform(0.5, 2.0))
        
        elif mutation_type == "part_type":
            # Could add new part type - simplified for now
            pass
        
        return new_rule
    
    def _validate_candidates(self) -> None:
        """Validate all candidates on supports and anomalies."""
        for candidate in self.candidates:
            # Support validation
            candidate.support_hold_rate = self.validator.validate_on_supports(candidate.rule)
            
            # Anomaly discrimination
            auroc, aupr = self.validator.validate_on_anomalies(candidate.rule)
            candidate.anomaly_detection_auroc = auroc
            candidate.anomaly_detection_aupr = aupr
    
    def _compute_relations(self) -> None:
        """Compute dependencies and redundancy for all candidates."""
        for candidate in self.candidates:
            # Dependencies with accepted rules
            depends_on, implied_by = self.validator.check_dependencies(
                candidate.rule, self.accepted_rules
            )
            candidate.depends_on = depends_on
            candidate.implied_by = implied_by
            
            # Redundancy with accepted rules
            candidate.redundancy_score = self.validator.compute_redundancy(
                candidate.rule, self.accepted_rules
            )
    
    def _compute_fitness(self) -> None:
        """Compute fitness for each candidate."""
        for candidate in self.candidates:
            fitness = (
                self.config.w_support * candidate.support_hold_rate +
                self.config.w_anomaly * candidate.anomaly_detection_auroc +
                self.config.w_novelty * (1.0 - candidate.redundancy_score)
            )
            # Penalty for high redundancy
            if candidate.redundancy_score > 0.5:
                fitness *= (1.0 - candidate.redundancy_score)
            candidate.fitness = fitness
    
    def _greedy_coverage_selection(self) -> List[NormalityRule]:
        """Greedy selection for maximum anomaly coverage with dependency awareness."""
        # Sort by fitness
        sorted_candidates = sorted(self.candidates, key=lambda c: c.fitness, reverse=True)
        
        selected = []
        covered_anomaly_types = set()
        dependency_tracker = RuleDependencyTracker()
        
        for candidate in sorted_candidates:
            # Check if dependencies are satisfied
            deps_satisfied = all(
                dep in [r.rule_id for r in selected]
                for dep in candidate.depends_on
            )
            if not deps_satisfied:
                continue
            
            # Check if adds novel coverage
            executor = self.executor
            new_coverage = 0
            for graph in self.validator.anomaly_graphs:
                result = executor.execute(graph)
                r = next((r for r in result.rule_results if r.rule_id == candidate.rule.rule_id), None)
                if r and r.violation_score > 1e-6:
                    # This rule catches this anomaly type
                    anomaly_type = "unknown"  # Would need proper typing
                    if anomaly_type not in covered_anomaly_types:
                        new_coverage += 1
                        covered_anomaly_types.add(anomaly_type)
            
            if new_coverage > 0 or not selected:
                selected.append(candidate.rule)
                # Track dependencies
                for dep in candidate.depends_on:
                    dependency_tracker.add_rule(candidate.rule, candidate.depends_on, candidate.implied_by)
        
        return [c.rule for c in selected]
    
    def _evolve_population(self) -> None:
        """Evolve population for next generation."""
        # Elite preservation
        elite = self.candidates[:self.config.elite_size]
        new_population = [copy.deepcopy(c) for c in elite]
        
        # Generate offspring
        while len(new_population) < self.config.max_candidates:
            if self.rng.random() < self.config.crossover_rate and len(self.candidates) >= 2:
                parent1 = self.rng.choice(self.candidates[:10])
                parent2 = self.rng.choice(self.candidates[:10])
                child_rule = self._crossover(parent1.rule, parent2.rule)
            else:
                parent = self.rng.choice(self.candidates[:10])
                child_rule = self._mutate_rule(parent.rule)
            
            child = RuleCandidate(rule=child_rule, generation=0)
            new_population.append(child)
        
        self.candidates = new_population[:self.config.max_candidates]
    
    def _crossover(self, rule1: NormalityRule, rule2: NormalityRule) -> NormalityRule:
        """Crossover two rules of same type."""
        if rule1.predicate_name != rule2.predicate_name:
            return copy.deepcopy(rule1)
        
        new_rule = copy.deepcopy(rule1)
        new_rule.rule_id = f"cross_{rule1.rule_id}_{rule2.rule_id}"
        
        # Blend numeric parameters
        if rule1.rule_type == RuleType.COUNT:
            if rule1.expected_count is not None and rule2.expected_count is not None:
                new_rule.expected_count = int((rule1.expected_count + rule2.expected_count) / 2)
            new_rule.count_tolerance = max(rule1.count_tolerance, rule2.count_tolerance)
        
        elif rule1.rule_type == RuleType.ATTRIBUTE:
            if rule1.expected_value is not None and rule2.expected_value is not None:
                new_rule.expected_value = (rule1.expected_value + rule2.expected_value) / 2
            if rule1.tolerance is not None and rule2.tolerance is not None:
                new_rule.tolerance = max(rule1.tolerance, rule2.tolerance)
        
        elif rule1.rule_type == RuleType.SPATIAL:
            new_rule.spatial_threshold = max(rule1.spatial_threshold, rule2.spatial_threshold)
        
        new_rule.confidence = min(rule1.confidence, rule2.confidence)
        new_rule.reliability = min(rule1.reliability, rule2.reliability)
        new_rule.source = "evolution"
        
        return new_rule


def run_validation_driven_search(
    qwen_rules: List[NormalityRule],
    support_graphs: List[SceneGraph],
    anomaly_graphs: List[SceneGraph],
    support_masks: List[np.ndarray],
    anomaly_masks: List[np.ndarray],
    executor: 'RuleExecutor',
    prototype_bank: SemanticPrototypeBank,
    config: Optional[SearchConfig] = None,
) -> Tuple[RuleSet, List[Dict[str, Any]]]:
    """Run validation-driven rule search."""
    if config is None:
        config = SearchConfig()
    
    validator = RuleValidator(
        support_graphs, [], anomaly_graphs, anomaly_masks,
        executor, prototype_bank
    )
    
    search = ValidationDrivenSearch(config, validator, executor)
    rule_set, history = search.run(
        qwen_rules, support_graphs, anomaly_graphs, [], anomaly_masks
    )
    
    return rule_set, history


__all__ = [
    "RuleCandidate", "SearchConfig", "RuleValidator", "RuleDependencyTracker",
    "ValidationDrivenSearch", "run_validation_driven_search",
]