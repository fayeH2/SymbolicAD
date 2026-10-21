"""
Evolutionary Search Framework for Symbolic Rule Discovery.

Implements the SymbolicDet-style evolutionary search:
- Population initialization
- Fitness evaluation on labeled data
- Selection (tournament/elitism)
- Crossover and mutation
- LLM-guided feedback (contextual feedback)
- Hall of Fame / best rule tracking
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Set
import json
import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score

from .expression_space import (
    ExprNode, FuzzyExpressionEvaluator, ExpressionGenerator,
    ExpressionMutator, ExpressionCrossover,
    ArithmeticOp, ComparisonOp, LogicOp, ExprType,
)
from .fuzzy_logic import fuzzy_or, fuzzy_and, fuzzy_not, fuzzy_implies, fuzzy_xor


@dataclass
class SearchConfig:
    """Configuration for evolutionary search."""
    population_size: int = 100
    max_generations: int = 50
    tournament_size: int = 5
    elitism: int = 2
    crossover_rate: float = 0.5
    mutation_rate: float = 0.3
    max_depth: int = 5
    complexity_penalty: float = 0.01  # lambda in paper
    target_metric: str = "auroc"  # or "aupr", "f1"
    early_stopping_patience: int = 10
    random_seed: int = 42


@dataclass
class Individual:
    """An individual in the population."""
    expr: ExprNode
    fitness: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_ids: Tuple[int, ...] = field(default_factory=tuple)
    id: int = 0
    
    def __lt__(self, other: "Individual") -> bool:
        return self.fitness < other.fitness


@dataclass
class SearchResult:
    """Result of evolutionary search."""
    best_individual: Individual
    hall_of_fame: List[Individual]
    generation_stats: List[Dict[str, Any]]
    config: SearchConfig


class FitnessEvaluator(ABC):
    """Abstract fitness evaluator for expressions."""
    
    @abstractmethod
    def evaluate(self, expr: ExprNode, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Evaluate expression on data. Returns (fitness, metrics_dict)."""
        pass


class ClassificationFitnessEvaluator(FitnessEvaluator):
    """Fitness evaluator for binary classification (anomaly detection)."""
    
    def __init__(
        self,
        metric: str = "auroc",
        complexity_penalty: float = 0.01,
        use_fuzzy: bool = True,
    ):
        self.metric = metric
        self.complexity_penalty = complexity_penalty
        self.use_fuzzy = use_fuzzy
    
    def evaluate(self, expr: ExprNode, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Evaluate expression as anomaly scorer."""
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        
        # X columns correspond to variable names
        # We need variable names - assume they're passed via feature_names
        # For now, use generic names
        var_names = [f"x{i}" for i in range(X.shape[1])]
        
        evaluator_class = FuzzyExpressionEvaluator if self.use_fuzzy else ExpressionEvaluator
        
        for i in range(n_samples):
            var_values = {var_names[j]: float(X[i, j]) for j in range(X.shape[1])}
            evaluator = evaluator_class(var_values)
            try:
                scores[i] = np.clip(evaluator.evaluate(expr), 0.0, 1.0)
            except Exception:
                scores[i] = 0.0  # Invalid expression gets minimum score
        
        # Compute metrics
        if len(np.unique(y)) < 2:
            return 0.0, {"error": "single class"}
        
        try:
            auroc = roc_auc_score(y, scores)
            aupr = average_precision_score(y, scores)
        except Exception:
            auroc = 0.5
            aupr = 0.5
        
        # Complexity penalty
        complexity = expr.size() * self.complexity_penalty
        
        if self.metric == "auroc":
            fitness = auroc - complexity
        elif self.metric == "aupr":
            fitness = aupr - complexity
        else:
            fitness = (auroc + aupr) / 2 - complexity
        
        metrics = {
            "auroc": auroc,
            "aupr": aupr,
            "complexity": expr.size(),
            "penalty": complexity,
            "raw_fitness": fitness + complexity,
        }
        
        return fitness, metrics


class EvolutionarySearch:
    """Main evolutionary search engine."""
    
    def __init__(
        self,
        config: SearchConfig,
        fitness_evaluator: FitnessEvaluator,
        variable_names: Sequence[str],
        constants: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0),
        llm_feedback_fn: Optional[Callable[[List[Individual], int], List[str]]] = None,
    ):
        self.config = config
        self.fitness_evaluator = fitness_evaluator
        self.variable_names = list(variable_names)
        self.constants = list(constants)
        self.llm_feedback_fn = llm_feedback_fn
        
        # Initialize random state
        self.rng = np.random.RandomState(config.random_seed)
        random.seed(config.random_seed)
        
        # Components
        self.generator = ExpressionGenerator(variable_names, constants, config.max_depth)
        self.mutator = ExpressionMutator(variable_names, constants, config.max_depth)
        self.crossover = ExpressionCrossover()
        
        # State
        self.population: List[Individual] = []
        self.hall_of_fame: List[Individual] = []
        self.generation_stats: List[Dict[str, Any]] = []
        self.next_id = 0
        self.best_fitness_history: List[float] = []
    
    def _new_id(self) -> int:
        self.next_id += 1
        return self.next_id
    
    def initialize_population(self) -> List[Individual]:
        """Create initial random population."""
        exprs = self.generator.random_population(self.config.population_size)
        population = []
        for expr in exprs:
            population.append(Individual(
                expr=expr,
                fitness=0.0,
                generation=0,
                id=self._new_id(),
            ))
        return population
    
    def evaluate_population(self, population: List[Individual], X: np.ndarray, y: np.ndarray) -> None:
        """Evaluate fitness for all individuals."""
        for ind in population:
            fitness, metrics = self.fitness_evaluator.evaluate(ind.expr, X, y)
            ind.fitness = fitness
            ind.metrics = metrics
    
    def tournament_selection(self, population: List[Individual], k: int = None) -> Individual:
        """Tournament selection."""
        k = k or self.config.tournament_size
        candidates = self.rng.choice(population, size=min(k, len(population)), replace=False)
        return max(candidates, key=lambda ind: ind.fitness)
    
    def select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """Select two parents for crossover."""
        return self.tournament_selection(population), self.tournament_selection(population)
    
    def create_offspring(self, parent1: Individual, parent2: Individual, generation: int) -> List[Individual]:
        """Create offspring through crossover and mutation."""
        offspring = []
        
        # Crossover
        if self.rng.random() < self.config.crossover_rate:
            child1_expr, child2_expr = self.crossover.crossover(parent1.expr, parent2.expr)
        else:
            child1_expr, child2_expr = parent1.expr, parent2.expr
        
        # Mutation
        child1_expr = self.mutator.mutate(child1_expr, self.config.mutation_rate)
        child2_expr = self.mutator.mutate(child2_expr, self.config.mutation_rate)
        
        for child_expr in [child1_expr, child2_expr]:
            offspring.append(Individual(
                expr=child_expr,
                fitness=0.0,
                generation=generation,
                parent_ids=(parent1.id, parent2.id),
                id=self._new_id(),
            ))
        
        return offspring
    
    def update_hall_of_fame(self, population: List[Individual], hof_size: int = 10) -> None:
        """Update hall of fame with best individuals."""
        all_candidates = self.hall_of_fame + population
        all_candidates.sort(key=lambda ind: ind.fitness, reverse=True)
        # Deduplicate by expression structure
        seen = set()
        unique = []
        for ind in all_candidates:
            expr_str = str(ind.expr)
            if expr_str not in seen:
                seen.add(expr_str)
                unique.append(ind)
                if len(unique) >= hof_size:
                    break
        self.hall_of_fame = unique
    
    def get_llm_feedback(self, generation: int) -> List[str]:
        """Get LLM feedback on current best individuals."""
        if self.llm_feedback_fn is None:
            return []
        try:
            return self.llm_feedback_fn(self.hall_of_fame[:5], generation)
        except Exception:
            return []
    
    def inject_llm_suggestions(self, suggestions: List[str], population: List[Individual]) -> None:
        """Inject LLM-suggested expressions into population (replace worst)."""
        if not suggestions:
            return
        # This would parse LLM suggestions into ExprNodes
        # For now, placeholder
        pass
    
    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> SearchResult:
        """Run the evolutionary search."""
        # Initialize
        self.population = self.initialize_population()
        self.hall_of_fame = []
        self.generation_stats = []
        self.best_fitness_history = []
        
        # Evaluate initial population
        self.evaluate_population(self.population, X_train, y_train)
        self.update_hall_of_fame(self.population)
        
        best_fitness = max(ind.fitness for ind in self.population)
        self.best_fitness_history.append(best_fitness)
        
        patience_counter = 0
        
        for gen in range(1, self.config.max_generations + 1):
            # Selection and reproduction
            new_population = []
            
            # Elitism: keep best individuals
            elite = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)[:self.config.elitism]
            new_population.extend(elite)
            
            # Generate offspring
            while len(new_population) < self.config.population_size:
                parent1, parent2 = self.select_parents(self.population)
                offspring = self.create_offspring(parent1, parent2, gen)
                new_population.extend(offspring)
            
            # Trim to population size
            new_population = new_population[:self.config.population_size]
            
            # Evaluate new population
            self.evaluate_population(new_population, X_train, y_train)
            
            # Update hall of fame
            self.update_hall_of_fame(new_population)
            self.population = new_population
            
            # Track stats
            gen_fitness = [ind.fitness for ind in self.population]
            best_ind = max(self.population, key=lambda ind: ind.fitness)
            self.best_fitness_history.append(best_ind.fitness)
            
            stats = {
                "generation": gen,
                "best_fitness": best_ind.fitness,
                "mean_fitness": float(np.mean(gen_fitness)),
                "std_fitness": float(np.std(gen_fitness)),
                "best_expr": str(best_ind.expr),
                "best_metrics": best_ind.metrics,
                "population_size": len(self.population),
            }
            self.generation_stats.append(stats)
            
            # Validation check
            if X_val is not None and y_val is not None:
                val_fitness, val_metrics = self.fitness_evaluator.evaluate(best_ind.expr, X_val, y_val)
                stats["val_fitness"] = val_fitness
                stats["val_metrics"] = val_metrics
            
            # Early stopping
            if best_ind.fitness <= self.best_fitness_history[-2] + 1e-6:
                patience_counter += 1
            else:
                patience_counter = 0
            
            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at generation {gen}")
                break
            
            # LLM feedback
            if self.llm_feedback_fn and gen % 5 == 0:
                suggestions = self.get_llm_feedback(gen)
                if suggestions:
                    stats["llm_suggestions"] = suggestions
        
        # Final evaluation on validation set
        best_individual = max(self.hall_of_fame, key=lambda ind: ind.fitness)
        if X_val is not None and y_val is not None:
            val_fitness, val_metrics = self.fitness_evaluator.evaluate(best_individual.expr, X_val, y_val)
            best_individual.metrics["val"] = val_metrics
        
        return SearchResult(
            best_individual=best_individual,
            hall_of_fame=self.hall_of_fame,
            generation_stats=self.generation_stats,
            config=self.config,
        )


def create_default_search(
    variable_names: Sequence[str],
    metric: str = "auroc",
    **kwargs
) -> EvolutionarySearch:
    """Factory for default evolutionary search."""
    config = SearchConfig(
        target_metric=metric,
        **kwargs
    )
    evaluator = ClassificationFitnessEvaluator(
        metric=metric,
        complexity_penalty=config.complexity_penalty,
    )
    return EvolutionarySearch(config, evaluator, variable_names)


# LLM Feedback Integration (placeholder for Qwen integration)
def create_llm_feedback_fn(
    llm_client: Any,  # Qwen or other LLM client
    scene_description: str,
    variable_descriptions: Dict[str, str],
) -> Callable[[List[Individual], int], List[str]]:
    """Create LLM feedback function for evolutionary search."""
    
    def feedback_fn(hall_of_fame: List[Individual], generation: int) -> List[str]:
        # Format current best rules for LLM
        rules_text = "\n".join([
            f"{i+1}. {str(ind.expr)} (fitness: {ind.fitness:.4f}, AUROC: {ind.metrics.get('auroc', 0):.4f})"
            for i, ind in enumerate(hall_of_fame[:3])
        ])
        
        prompt = f"""You are guiding an evolutionary search for anomaly detection rules.

Scene: {scene_description}
Variables: {json.dumps(variable_descriptions, indent=2)}

Current best rules (generation {generation}):
{rules_text}

Please suggest 3-5 new rule ideas or modifications that might improve anomaly detection.
Focus on:
1. Arithmetic combinations (ratios, differences, products)
2. Logical combinations (implications, conjunctions)
3. Threshold comparisons

Return as JSON list of strings, each a rule expression idea."""
        
        # Call LLM (placeholder)
        # response = llm_client.generate(prompt)
        # Parse response and return suggestions
        return []  # Placeholder
    
    return feedback_fn


__all__ = [
    "SearchConfig", "Individual", "SearchResult",
    "FitnessEvaluator", "ClassificationFitnessEvaluator",
    "EvolutionarySearch", "create_default_search",
    "create_llm_feedback_fn",
]