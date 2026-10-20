"""Rule search and optimization using NSGA-II multi-objective evolution."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from neurosymbolic_fsad.rules.generation.dsl import NormalityRule, RuleSet, RuleType
from neurosymbolic_fsad.perception.scene_graph import SceneGraph
from neurosymbolic_fsad.perception.prototypes import SemanticPrototypeBank


@dataclass
class RuleCandidate:
    """A candidate rule in the evolutionary population."""
    rule: NormalityRule
    fitness: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_ids: Tuple[str, ...] = ()
    _uid: int = field(default_factory=lambda: random.randint(0, 2**31-1))
    
    def __hash__(self):
        return hash(self._uid)
    
    def __eq__(self, other):
        if not isinstance(other, RuleCandidate):
            return False
        return self._uid == other._uid
    
    def dominates(self, other: "RuleCandidate") -> bool:
        """Check if this candidate dominates another (Pareto dominance)."""
        better_in_any = False
        for key in self.fitness:
            if key not in other.fitness:
                continue
            if self.fitness[key] < other.fitness[key]:  # Minimization
                return False
            if self.fitness[key] > other.fitness[key]:
                better_in_any = True
        return better_in_any


@dataclass
class SearchConfig:
    """Configuration for NSGA-II rule search."""
    population_size: int = 100
    max_generations: int = 50
    tournament_size: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.3
    elite_size: int = 10
    random_seed: int = 42
    
    # Objective weights (for scalarization if needed)
    obj_weights: Dict[str, float] = field(default_factory=lambda: {
        "support_consistency": 1.0,
        "semantic_coverage": 1.0,
        "complementarity": 1.0,
        "redundancy": -1.0,  # Minimize
        "reliability": 1.0,
        "spatial_groundability": 1.0,
        "complexity": -0.5,  # Minimize
    })


class RuleSearchEvaluator:
    """Evaluate rule candidates on multiple objectives."""
    
    def __init__(
        self,
        support_graphs: List[SceneGraph],
        prototype_bank: SemanticPrototypeBank,
        normal_masks: List[np.ndarray],
        config: SearchConfig,
    ):
        self.support_graphs = support_graphs
        self.prototype_bank = prototype_bank
        self.normal_masks = normal_masks
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
    
    def evaluate(self, candidate: RuleCandidate) -> Dict[str, float]:
        """Evaluate a rule candidate on all objectives."""
        rule = candidate.rule
        fitness = {}
        
        # 1. Support consistency: how well rule holds on normal supports
        fitness["support_consistency"] = self._eval_support_consistency(rule)
        
        # 2. Semantic coverage: does rule cover a new semantic dimension
        fitness["semantic_coverage"] = self._eval_semantic_coverage(rule)
        
        # 3. Complementarity: does rule detect different anomalies than existing
        fitness["complementarity"] = self._eval_complementarity(rule)
        
        # 4. Redundancy: overlap with existing rules (minimize)
        fitness["redundancy"] = self._eval_redundancy(rule)
        
        # 5. Reliability: consistency of rule evaluation
        fitness["reliability"] = self._eval_reliability(rule)
        
        # 6. Spatial groundability: can we produce a spatial anomaly map
        fitness["spatial_groundability"] = self._eval_spatial_groundability(rule)
        
        # 7. Complexity: rule complexity (minimize)
        fitness["complexity"] = self._eval_complexity(rule)
        
        candidate.fitness = fitness
        return fitness
    
    def _eval_support_consistency(self, rule: NormalityRule) -> float:
        """Evaluate how consistently rule holds on normal supports."""
        if not self.support_graphs:
            return 0.5
        
        hold_count = 0
        total = 0
        
        for graph in self.support_graphs:
            if self._check_rule_on_graph(rule, graph):
                hold_count += 1
            total += 1
        
        return hold_count / total if total > 0 else 0.5
    
    def _check_rule_on_graph(self, rule: NormalityRule, graph: SceneGraph) -> bool:
        """Check if a rule holds on a scene graph."""
        if rule.rule_type == RuleType.COUNT:
            target_type = rule.part_type
            count = sum(1 for c in graph.components 
                       if graph.component_types.get(id(c), "") == target_type)
            expected = rule.expected_count or 0
            return abs(count - expected) <= rule.count_tolerance
        
        elif rule.rule_type == RuleType.ATTRIBUTE:
            # Check attribute constraint on relevant components
            target_type = rule.part_type
            for comp in graph.components:
                if graph.component_types.get(id(comp), "") == target_type:
                    attr_val = self._get_attribute(comp, rule.attribute)
                    if attr_val is not None:
                        expected = rule.expected_value or 0
                        tol = rule.tolerance or 0.1
                        if rule.direction == "stable":
                            if abs(attr_val - expected) > tol:
                                return False
                        elif rule.direction == "low":
                            if attr_val > expected + tol:
                                return False
                        elif rule.direction == "high":
                            if attr_val < expected - tol:
                                return False
            return True
        
        elif rule.rule_type == RuleType.SPATIAL:
            # Check spatial relation
            return self._check_spatial_relation(rule, graph)
        
        elif rule.rule_type == RuleType.TOPOLOGICAL:
            return self._check_topological_relation(rule, graph)
        
        elif rule.rule_type == RuleType.COMPOSITION:
            return self._check_composition(rule, graph)
        
        return True
    
    def _get_attribute(self, comp: Any, attr_type: Optional[AttributeType]) -> Optional[float]:
        """Get attribute value from component."""
        if attr_type is None or not hasattr(comp, 'features'):
            return None
        return comp.features.get(attr_type.value)
    
    def _check_spatial_relation(self, rule: NormalityRule, graph: SceneGraph) -> bool:
        """Check spatial relation rule."""
        if not rule.spatial_relation or not rule.part_type or not rule.target_part_type:
            return True
        
        source_comps = [c for c in graph.components 
                       if graph.component_types.get(id(c), "") == rule.part_type]
        target_comps = [c for c in graph.components 
                       if graph.component_types.get(id(c), "") == rule.target_part_type]
        
        if not source_comps or not target_comps:
            return True  # Vacuously true if components missing
        
        # Check if relation holds for at least one pair
        for src in source_comps:
            for tgt in target_comps:
                if self._check_spatial_pair(src, tgt, rule.spatial_relation, rule.spatial_threshold):
                    return True
        return False
    
    def _check_spatial_pair(
        self, src: Any, tgt: Any, rel: Any, threshold: float
    ) -> bool:
        """Check spatial relation between two components."""
        cx1, cy1 = src.centroid
        cx2, cy2 = tgt.centroid
        dx = cx2 - cx1
        dy = cy2 - cy1
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist > threshold:
            return False
        
        if rel == SpatialRelationType.LEFT_OF:
            return dx < -0.02
        elif rel == SpatialRelationType.RIGHT_OF:
            return dx > 0.02
        elif rel == SpatialRelationType.ABOVE:
            return dy < -0.02
        elif rel == SpatialRelationType.BELOW:
            return dy > 0.02
        elif rel == SpatialRelationType.NEAR:
            return dist < threshold
        elif rel == SpatialRelationType.ALIGNED_HORIZONTAL:
            return abs(dy) < 0.03
        elif rel == SpatialRelationType.ALIGNED_VERTICAL:
            return abs(dx) < 0.03
        return True
    
    def _check_topological_relation(self, rule: NormalityRule, graph: SceneGraph) -> bool:
        """Check topological relation rule."""
        if not rule.topological_relation:
            return True
        
        for rel in graph.topological_relations:
            src_type = graph.component_types.get(rel.source_id, "")
            tgt_type = graph.component_types.get(rel.target_id, "")
            
            if src_type == rule.part_type and tgt_type == rule.target_part_type:
                if rel.relation_type == rule.topological_relation.value:
                    return True
        return False
    
    def _check_composition(self, rule: NormalityRule, graph: SceneGraph) -> bool:
        """Check composition rule."""
        present_types = set(graph.component_types.values())
        
        # Check required types
        for req_type in rule.required_part_types:
            if req_type not in present_types:
                return False
        
        # Check forbidden combinations
        for forbidden in rule.forbidden_combinations:
            if all(t in present_types for t in forbidden):
                return False
        
        return True
    
    def _eval_semantic_coverage(self, rule: NormalityRule) -> float:
        """Evaluate semantic dimension coverage."""
        # Each rule type covers a different dimension
        type_coverage = {
            RuleType.COUNT: 1.0,
            RuleType.ATTRIBUTE: 1.0,
            RuleType.SPATIAL: 1.0,
            RuleType.TOPOLOGICAL: 1.0,
            RuleType.COMPOSITION: 1.0,
            RuleType.CONDITIONAL: 1.0,
        }
        return type_coverage.get(rule.rule_type, 0.5)
    
    def _eval_complementarity(self, rule: NormalityRule) -> float:
        """Evaluate complementarity with existing rules (placeholder)."""
        # In practice, this would compare against current rule set
        return 0.5
    
    def _eval_redundancy(self, rule: NormalityRule) -> float:
        """Evaluate redundancy (lower is better)."""
        # Check if rule is logically implied by others
        return 0.2  # Placeholder
    
    def _eval_reliability(self, rule: NormalityRule) -> float:
        """Evaluate rule reliability."""
        return rule.reliability * rule.confidence
    
    def _eval_spatial_groundability(self, rule: NormalityRule) -> float:
        """Evaluate if rule can produce spatial anomaly map."""
        if not rule.spatial_groundable:
            return 0.0
        strategy = rule.get_grounding_strategy()
        strategy_scores = {
            "missing_slots": 1.0,
            "extra_objects": 1.0,
            "attribute_map": 0.9,
            "position_deviation": 0.9,
            "connection_map": 0.8,
            "composition_map": 0.7,
            "count_mismatch": 0.6,
        }
        return strategy_scores.get(strategy, 0.5)
    
    def _eval_complexity(self, rule: NormalityRule) -> float:
        """Evaluate rule complexity (lower is better)."""
        complexity = 1.0
        if rule.conditions:
            complexity += 0.5 * len(rule.conditions)
        if rule.then_rule or rule.else_rule:
            complexity += 1.0
        if rule.rule_type in [RuleType.COMPOSITION, RuleType.CONDITIONAL]:
            complexity += 0.5
        return complexity


class NSGA2RuleSearch:
    """NSGA-II multi-objective evolutionary search for rule sets."""
    
    def __init__(
        self,
        config: SearchConfig,
        evaluator: RuleSearchEvaluator,
        initial_rules: List[NormalityRule],
    ):
        self.config = config
        self.evaluator = evaluator
        self.initial_rules = initial_rules
        self.population: List[RuleCandidate] = []
        self.archive: List[RuleCandidate] = []  # Pareto front
        self.generation = 0
        self.rng = np.random.RandomState(config.random_seed)
    
    def initialize_population(self) -> List[RuleCandidate]:
        """Initialize population with mutations of initial rules."""
        population = []
        
        # Add initial rules
        for rule in self.initial_rules:
            candidate = RuleCandidate(rule=rule, generation=0)
            self.evaluator.evaluate(candidate)
            population.append(candidate)
        
        # Generate variations
        while len(population) < self.config.population_size:
            base_rule = self.rng.choice(self.initial_rules)
            mutated = self._mutate_rule(base_rule)
            candidate = RuleCandidate(rule=mutated, generation=0)
            self.evaluator.evaluate(candidate)
            population.append(candidate)
        
        self.population = population
        return population
    
    def _mutate_rule(self, rule: NormalityRule) -> NormalityRule:
        """Create a mutated copy of a rule."""
        new_rule = copy.deepcopy(rule)
        new_rule.rule_id = f"{rule.rule_id}_mut_{self.generation}"
        
        mutation_type = self.rng.choice([
            "parameter", "threshold", "tolerance", "part_type"
        ])
        
        if mutation_type == "parameter" and rule.rule_type == RuleType.COUNT:
            if new_rule.expected_count is not None:
                delta = self.rng.choice([-1, 1])
                new_rule.expected_count = max(0, new_rule.expected_count + delta)
        
        elif mutation_type == "threshold" and rule.rule_type == RuleType.SPATIAL:
            new_rule.spatial_threshold = np.clip(
                new_rule.spatial_threshold + self.rng.normal(0, 0.05),
                0.02, 0.5
            )
        
        elif mutation_type == "tolerance" and rule.rule_type == RuleType.ATTRIBUTE:
            if new_rule.tolerance is not None:
                new_rule.tolerance = max(0.01, new_rule.tolerance * self.rng.uniform(0.5, 2.0))
        
        elif mutation_type == "part_type":
            # Change target part type
            available_types = list(self.evaluator.prototype_bank.prototypes.keys())
            if available_types:
                new_rule.part_type = self.rng.choice(available_types)
        
        return new_rule
    
    def _crossover(self, parent1: RuleCandidate, parent2: RuleCandidate) -> RuleCandidate:
        """Create offspring by combining two parents."""
        # For rules, crossover means creating a composite rule
        rule1, rule2 = parent1.rule, parent2.rule
        
        if rule1.rule_type == rule2.rule_type:
            new_rule = copy.deepcopy(rule1)
            new_rule.rule_id = f"cross_{self.generation}_{id(new_rule)}"
            
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
            
            # Combine metadata
            new_rule.confidence = min(rule1.confidence, rule2.confidence)
            new_rule.reliability = min(rule1.reliability, rule2.reliability)
            new_rule.source = "evolution"
            
            return RuleCandidate(
                rule=new_rule,
                generation=self.generation,
                parent_ids=(parent1.rule.rule_id, parent2.rule.rule_id),
            )
        
        # Different types - return one parent with mutation
        return self._mutate_rule(parent1.rule)
    
    def _tournament_selection(self, population: List[RuleCandidate]) -> RuleCandidate:
        """Tournament selection based on Pareto dominance."""
        tournament = self.rng.choice(population, size=self.config.tournament_size, replace=False)
        
        # Find non-dominated in tournament
        non_dominated = []
        for cand in tournament:
            dominated = False
            for other in tournament:
                if other is cand:
                    continue
                if other.dominates(cand):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(cand)
        
        if non_dominated:
            return self.rng.choice(non_dominated)
        return self.rng.choice(tournament)
    
    def _fast_non_dominated_sort(
        self, population: List[RuleCandidate]
    ) -> List[List[RuleCandidate]]:
        """Fast non-dominated sorting (NSGA-II)."""
        n = len(population)
        domination_count = [0] * n
        dominated_sets = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if population[i].dominates(population[j]):
                    dominated_sets[i].append(j)
                    domination_count[j] += 1
                elif population[j].dominates(population[i]):
                    dominated_sets[j].append(i)
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(population[i])
        
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i_idx, cand_i in enumerate(fronts[front_idx]):
                # Find index in population
                pop_idx = population.index(cand_i)
                for j_idx in dominated_sets[pop_idx]:
                    domination_count[j_idx] -= 1
                    if domination_count[j_idx] == 0:
                        next_front.append(population[j_idx])
            front_idx += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def _crowding_distance(self, front: List[RuleCandidate]) -> Dict[RuleCandidate, float]:
        """Calculate crowding distance for a front."""
        if len(front) <= 2:
            return {cand: float('inf') for cand in front}
        
        distances = {cand: 0.0 for cand in front}
        objectives = list(front[0].fitness.keys())
        
        for obj in objectives:
            sorted_front = sorted(front, key=lambda c: c.fitness.get(obj, 0))
            obj_min = sorted_front[0].fitness.get(obj, 0)
            obj_max = sorted_front[-1].fitness.get(obj, 0)
            
            if obj_max == obj_min:
                continue
            
            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')
            
            for i in range(1, len(sorted_front) - 1):
                prev_val = sorted_front[i-1].fitness.get(obj, 0)
                next_val = sorted_front[i+1].fitness.get(obj, 0)
                distances[sorted_front[i]] += (next_val - prev_val) / (obj_max - obj_min)
        
        return distances
    
    def run(self) -> Tuple[List[NormalityRule], List[Dict[str, Any]]]:
        """Run NSGA-II search and return Pareto-optimal rule set."""
        self.population = self.initialize_population()
        history = []
        
        for gen in range(self.config.max_generations):
            self.generation = gen
            
            # Create offspring
            offspring = []
            while len(offspring) < self.config.population_size:
                parent1 = self._tournament_selection(self.population)
                parent2 = self._tournament_selection(self.population)
                
                if self.rng.random() < self.config.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = RuleCandidate(
                        rule=copy.deepcopy(parent1.rule),
                        generation=gen,
                        parent_ids=(parent1.rule.rule_id,),
                    )
                
                # Mutation
                if self.rng.random() < self.config.mutation_rate:
                    child.rule = self._mutate_rule(child.rule)
                
                self.evaluator.evaluate(child)
                offspring.append(child)
            
            # Combine and select
            combined = self.population + offspring
            fronts = self._fast_non_dominated_sort(combined)
            
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.config.population_size:
                    new_population.extend(front)
                else:
                    # Crowding distance selection
                    distances = self._crowding_distance(front)
                    front_sorted = sorted(front, key=lambda c: distances[c], reverse=True)
                    needed = self.config.population_size - len(new_population)
                    new_population.extend(front_sorted[:needed])
                    break
            
            self.population = new_population
            
            # Update archive (Pareto front)
            self._update_archive()
            
            # Record history
            history.append({
                "generation": gen,
                "population_size": len(self.population),
                "archive_size": len(self.archive),
                "best_fitness": self._get_best_fitness(),
            })
            
            print(f"Generation {gen}: pop={len(self.population)}, archive={len(self.archive)}")
        
        # Return archive rules
        final_rules = [cand.rule for cand in self.archive]
        return final_rules, history
    
    def _update_archive(self):
        """Update Pareto archive with current population."""
        # Combine archive and population
        combined = self.archive + self.population
        fronts = self._fast_non_dominated_sort(combined)
        
        # Keep only first front (Pareto optimal)
        self.archive = fronts[0] if fronts else []
    
    def _get_best_fitness(self) -> Dict[str, float]:
        """Get best fitness values across objectives."""
        if not self.archive:
            return {}
        
        best = {}
        for obj in self.archive[0].fitness:
            best[obj] = max(c.fitness.get(obj, 0) for c in self.archive)
        return best


def run_rule_search(
    qwen_rules: List[NormalityRule],
    support_graphs: List[SceneGraph],
    prototype_bank: SemanticPrototypeBank,
    normal_masks: List[np.ndarray],
    config: Optional[SearchConfig] = None,
) -> Tuple[RuleSet, List[Dict[str, Any]]]:
    """Run complete rule search pipeline."""
    if config is None:
        config = SearchConfig()
    
    evaluator = RuleSearchEvaluator(
        support_graphs, prototype_bank, normal_masks, config
    )
    
    search = NSGA2RuleSearch(config, evaluator, qwen_rules)
    final_rules, history = search.run()
    
    # Create final rule set
    rule_set = RuleSet(category="unknown", rules=final_rules)
    
    return rule_set, history