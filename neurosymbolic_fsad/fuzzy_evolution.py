"""
Fuzzy Logic AST Evolutionary Search for Symbolic Rule Discovery.

This module implements evolutionary search over fuzzy logic ASTs (NOT/AND/OR/IMPLIES)
similar to the v7 executable logic structure, but with flexible atom combinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import random
import numpy as np
from copy import deepcopy

from neurosymbolic_fsad.fuzzy_logic import (
    FuzzyLogicProgram, validate_logic_ast, execute_logic_ast,
    fuzzy_and, fuzzy_or, fuzzy_not, fuzzy_implies,
    LogicValidationError,
)
from neurosymbolic_fsad.evolutionary_search import (
    SearchConfig, FitnessEvaluator, Individual, SearchResult,
)


class FuzzyLogicAST:
    """Represents a fuzzy logic AST for evolutionary search."""
    
    def __init__(self, ast: Dict[str, Any], active_atoms: Sequence[str]):
        self.ast = ast
        self.active_atoms = tuple(active_atoms)
        # Validate
        self.program = FuzzyLogicProgram(ast, self.active_atoms)
    
    def evaluate(self, atom_values: Dict[str, float]) -> float:
        """Evaluate AST with fuzzy logic."""
        try:
            result, _ = self.program.evaluate(atom_values)
            return float(result)
        except Exception:
            return 0.0
    
    def mutate(self, available_atoms: Sequence[str], mutation_rate: float = 0.3) -> "FuzzyLogicAST":
        """Create a mutated copy."""
        new_ast = self._mutate_node(self.ast, available_atoms, mutation_rate)
        return FuzzyLogicAST(new_ast, self.active_atoms)
    
    def _mutate_node(self, node: Dict[str, Any], available_atoms: Sequence[str], rate: float) -> Dict[str, Any]:
        if random.random() > rate:
            return node
        
        op = node.get("op", "").upper()
        
        if op == "ATOM":
            # Mutate to different atom
            if random.random() < 0.5:
                return {"op": "ATOM", "name": random.choice(list(available_atoms))}
            return node
        
        elif op == "NOT":
            new_arg = self._mutate_node(node["arg"], available_atoms, rate)
            return {"op": "NOT", "arg": new_arg}
        
        elif op in ("AND", "OR"):
            new_args = [self._mutate_node(arg, available_atoms, rate) for arg in node.get("args", [])]
            # Possibly add/remove args
            if random.random() < 0.1 and len(new_args) > 1:
                new_args.pop(random.randrange(len(new_args)))
            elif random.random() < 0.1 and len(new_args) < 4:
                new_args.append(self._random_atom(available_atoms))
            return {"op": op, "args": new_args}
        
        elif op == "IMPLIES":
            new_args = [self._mutate_node(arg, available_atoms, rate) for arg in node.get("args", [])]
            return {"op": "IMPLIES", "args": new_args}
        
        return node
    
    def _random_atom(self, available_atoms: Sequence[str]) -> Dict[str, Any]:
        return {"op": "ATOM", "name": random.choice(list(available_atoms))}
    
    @classmethod
    def crossover(cls, parent1: "FuzzyLogicAST", parent2: "FuzzyLogicAST", 
                  available_atoms: Sequence[str]) -> Tuple["FuzzyLogicAST", "FuzzyLogicAST"]:
        """Crossover two ASTs by swapping subtrees."""
        # Collect all nodes with paths
        nodes1 = cls._collect_nodes(parent1.ast)
        nodes2 = cls._collect_nodes(parent2.ast)
        
        if not nodes1[1:] or not nodes2[1:]:
            return parent1, parent2
        
        # Pick random non-root nodes
        _, path1, _ = random.choice(nodes1[1:])
        _, path2, _ = random.choice(nodes2[1:])
        
        # Swap subtrees
        new_ast1 = cls._replace_at_path(parent1.ast, path1, cls._get_at_path(parent2.ast, path2))
        new_ast2 = cls._replace_at_path(parent2.ast, path2, cls._get_at_path(parent1.ast, path1))
        
        child1 = FuzzyLogicAST(new_ast1, parent1.active_atoms)
        child2 = FuzzyLogicAST(new_ast2, parent2.active_atoms)
        return child1, child2
    
    @classmethod
    def _collect_nodes(cls, ast: Dict[str, Any]) -> List[Tuple[Dict, Tuple[int, ...], Dict]]:
        """Collect (parent, path, node) for all nodes."""
        result = []
        def collect(node: Dict, path: Tuple[int, ...], parent: Optional[Dict]):
            result.append((parent, path, node))
            if isinstance(node, dict):
                if node.get("op") == "NOT":
                    for i, child in enumerate([node.get("arg")]):
                        if child:
                            collect(child, path + (i,), node)
                else:
                    for i, child in enumerate(node.get("args", [])):
                        if child:
                            collect(child, path + (i,), node)
        collect(ast, (), None)
        return result
    
    @classmethod
    def _get_at_path(cls, ast: Dict, path: Tuple[int, ...]) -> Dict:
        node = ast
        for idx in path:
            op = node.get("op", "").upper()
            if op == "NOT":
                node = node["arg"]
            else:
                node = node["args"][idx]
        return node
    
    @classmethod
    def _replace_at_path(cls, ast: Dict, path: Tuple[int, ...], new_node: Dict) -> Dict:
        if not path:
            return new_node
        idx = path[0]
        op = ast.get("op", "").upper()
        new_ast = deepcopy(ast)
        if new_ast.get("op") == "NOT":
            new_ast["arg"] = cls._replace_at_path(new_ast["arg"], path[1:], new_node)
        else:
            new_ast["args"][idx] = cls._replace_at_path(new_ast["args"][idx], path[1:], new_node)
        return new_ast
    
    @classmethod
    def random_ast(cls, available_atoms: Sequence[str], special_atoms: Sequence[str],
                   max_depth: int = 4) -> "FuzzyLogicAST":
        """Generate a random fuzzy logic AST."""
        all_atoms = list(available_atoms) + list(special_atoms)
        
        def random_node(depth: int = 0) -> Dict[str, Any]:
            if depth >= 3 or (depth > 0 and random.random() < 0.3):
                return {"op": "ATOM", "name": random.choice(all_atoms)}
            
            op = random.choice(["NOT", "AND", "OR", "IMPLIES"])
            
            if op == "NOT":
                return {"op": "NOT", "arg": random_node(depth + 1)}
            elif op == "IMPLIES":
                return {"op": "IMPLIES", "args": [random_node(depth + 1), random_node(depth + 1)]}
            else:  # AND, OR
                arity = random.randint(2, 3)
                return {"op": op, "args": [random_node(depth + 1) for _ in range(arity)]}
        
        # Build symbolic anomaly AST
        sym_anom = random_node()
        
        # Build full AST: OR(neural, AND(grounding, sym_anom))
        neural_atom = {"op": "ATOM", "name": "neural_anomaly"}
        grounding_atom = {"op": "ATOM", "name": "grounding"}
        sym_atom = {"op": "ATOM", "name": "symbolic_anomaly"}
        
        full_ast = {
            "op": "OR",
            "id": "final",
            "args": [
                neural_atom,
                {
                    "op": "AND",
                    "id": "grounded_symbolic",
                    "args": [
                        grounding_atom,
                        sym_atom
                    ]
                }
            ]
        }
        
        # Add symbolic_anomaly definition
        # We'll use a wrapper that evaluates sym_anom separately
        return FuzzyLogicAST(sym_anom, tuple())  # Simplified for now


class FuzzyLogicFitnessEvaluator(FitnessEvaluator):
    """Fitness evaluator for fuzzy logic ASTs."""
    
    def __init__(self, special_atoms: Sequence[str] = ("neural_anomaly", "grounding")):
        self.special_atoms = tuple(special_atoms)
    
    def evaluate(self, expr: Union[FuzzyLogicAST, Dict], X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Evaluate fuzzy logic AST on data."""
        if isinstance(expr, FuzzyLogicAST):
            ast_obj = expr
        else:
            # Assume it's a symbolic anomaly AST, wrap in full program
            ast_obj = self._wrap_symbolic_anomaly(expr)
        
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        
        # Variable names from X columns
        var_names = [f"x{i}" for i in range(X.shape[1])]
        
        for i in range(n_samples):
            var_values = {var_names[j]: float(X[i, j]) for j in range(X.shape[1])}
            # Add special atoms
            var_values["neural_anomaly"] = var_values.get("x5", 0.5)  # neural
            var_values["grounding"] = var_values.get("x5", 0.5)  # grounding
            
            try:
                result, _ = execute_logic_ast(ast_obj.ast if isinstance(ast_obj, FuzzyLogicAST) else ast_obj, 
                                              var_values,
                                              active_atoms=ast_obj.active_atoms if isinstance(ast_obj, FuzzyLogicAST) else [],
                                              special_atoms=self.special_atoms)
                scores[i] = float(np.clip(result, 0.0, 1.0))
            except Exception:
                scores[i] = 0.0
        
        if len(np.unique(y)) < 2:
            return 0.0, {"error": "single class"}
        
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auroc = roc_auc_score(y, scores)
            aupr = average_precision_score(y, scores)
        except Exception:
            auroc = 0.5
            aupr = 0.5
        
        # Complexity penalty
        complexity = ast_obj.ast.get("size", 10) * 0.005 if isinstance(ast_obj, FuzzyLogicAST) else 10 * 0.005
        
        fitness = auroc - complexity
        metrics = {"auroc": auroc, "aupr": aupr, "complexity": complexity, "raw_fitness": auroc}
        
        return fitness, metrics
    
    def _wrap_symbolic_anomaly(self, sym_anom_ast: Dict) -> FuzzyLogicProgram:
        """Wrap symbolic anomaly AST into full neuro-symbolic program."""
        # This is a simplified version - in practice, we'd build the full AST
        # OR(neural, AND(grounding, sym_anom))
        neural_atom = {"op": "ATOM", "name": "neural_anomaly", "id": "neural"}
        grounding_atom = {"op": "ATOM", "name": "grounding", "id": "grounding"}
        sym_atom = {"op": "ATOM", "name": "symbolic_anomaly", "id": "symbolic"}
        
        full_ast = {
            "op": "OR",
            "id": "final",
            "args": [
                neural_atom,
                {
                    "op": "AND",
                    "id": "grounded_symbolic",
                    "args": [
                        grounding_atom,
                        sym_atom
                    ]
                }
            ]
        }
        
        # For evaluation, we'd need to bind symbolic_anomaly to the actual sym_anom_ast
        # This is complex - we'll use a simpler approach in practice
        return FuzzyLogicProgram(full_ast, [], special_atoms=self.special_atoms)


class FuzzyLogicEvolutionarySearch:
    """Evolutionary search over fuzzy logic ASTs."""
    
    def __init__(
        self,
        config: SearchConfig,
        fitness_evaluator: FuzzyLogicFitnessEvaluator,
        available_atoms: Sequence[str],
        special_atoms: Sequence[str] = ("neural_anomaly", "grounding"),
    ):
        self.config = config
        self.fitness_evaluator = fitness_evaluator
        self.available_atoms = list(available_atoms)
        self.special_atoms = list(special_atoms)
        self.all_atoms = list(available_atoms) + list(special_atoms)
        self.rng = np.random.RandomState(config.random_seed)
        
        self.population: List[FuzzyLogicAST] = []
        self.hall_of_fame: List[FuzzyLogicAST] = []
        self.generation_stats: List[Dict] = []
    
    def initialize_population(self) -> List[FuzzyLogicAST]:
        """Create initial population of random fuzzy logic ASTs."""
        population = []
        for _ in range(self.config.population_size):
            # Generate random symbolic anomaly AST
            sym_anom = self._random_symbolic_anomaly()
            ast = self._build_full_ast(sym_anom)
            population.append(FuzzyLogicAST(ast, self.available_atoms))
        return population
    
    def _random_symbolic_anomaly(self) -> Dict:
        """Generate random symbolic anomaly AST."""
        all_atoms = self.available_atoms + self.special_atoms
        
        def random_node(depth=0):
            if depth >= 3 or (depth > 0 and random.random() < 0.3):
                return {"op": "ATOM", "name": random.choice(self.available_atoms + self.special_atoms)}
            
            op = random.choice(["NOT", "AND", "OR", "IMPLIES"])
            
            if op == "NOT":
                return {"op": "NOT", "arg": self._random_node(depth + 1)}
            elif op == "IMPLIES":
                return {"op": "IMPLIES", "args": [self._random_node(depth + 1), self._random_node(depth + 1)]}
            else:  # AND, OR
                arity = random.randint(2, 3)
                return {"op": op, "args": [self._random_node(depth + 1) for _ in range(arity)]}
        
        return random_node()
    
    def _random_node(self, depth):
        return self._random_node(depth)
    
    def _build_full_ast(self, sym_anom: Dict) -> Dict:
        """Build full neuro-symbolic AST with symbolic anomaly."""
        return {
            "op": "OR",
            "id": "final",
            "args": [
                {"op": "ATOM", "name": "neural_anomaly", "id": "neural"},
                {
                    "op": "AND",
                    "id": "grounded_symbolic",
                    "args": [
                        {"op": "ATOM", "name": "grounding", "id": "grounding"},
                        {"op": "ATOM", "name": "symbolic_anomaly", "id": "symbolic"}
                    ]
                }
            ]
        }
    
    def evaluate_population(self, population: List[FuzzyLogicAST], X: np.ndarray, y: np.ndarray):
        """Evaluate fitness for all individuals."""
        for ind in population:
            fitness, metrics = self.fitness_evaluator.evaluate(ind, X, y)
            ind.fitness = fitness
            ind.metrics = metrics
    
    def run(self, X_train, y_train, X_val, y_val, feature_names) -> SearchResult:
        """Run evolutionary search."""
        self.population = self.initialize_population()
        self.hall_of_fame = []
        self.generation_stats = []
        
        # Evaluate initial population
        self.evaluate_population(self.population, X_train, y_train)
        
        for gen in range(1, self.config.max_generations + 1):
            # Sort by fitness
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            
            # Update hall of fame
            all_candidates = self.hall_of_fame + self.population
            all_candidates.sort(key=lambda ind: ind.fitness, reverse=True)
            self.hall_of_fame = all_candidates[:10]
            
            best_fitness = self.population[0].fitness
            
            # Selection and reproduction
            new_population = self.population[:self.config.elitism]
            
            while len(new_population) < self.config.population_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                if random.random() < self.config.crossover_rate:
                    child1, child2 = FuzzyLogicAST.crossover(parent1, parent2, self.available_atoms)
                else:
                    child1, child2 = parent1, parent2
                
                child1 = child1.mutate(self.available_atoms, self.config.mutation_rate)
                child2 = child2.mutate(self.available_atoms, self.config.mutation_rate)
                
                new_population.extend([child1, child2])
            
            new_population = new_population[:self.config.population_size]
            self.evaluate_population(new_population, X_train, y_train)
            self.population = new_population
            
            # Track stats
            best_ind = self.population[0]
            self.generation_stats.append({
                "generation": gen,
                "best_fitness": best_ind.fitness,
                "best_ast": str(best_ind.ast),
            })
            
            # Early stopping
            if gen > 10 and best_ind.fitness <= self.generation_stats[-10]["best_fitness"] + 1e-4:
                break
        
        return SearchResult(
            best_individual=self.hall_of_fame[0],
            hall_of_fame=self.hall_of_fame,
            generation_stats=self.generation_stats,
            config=self.config,
        )
    
    def _tournament_selection(self) -> FuzzyLogicAST:
        candidates = random.sample(self.population, min(self.config.tournament_size, len(self.population)))
        return max(candidates, key=lambda ind: ind.fitness)


__all__ = [
    "FuzzyLogicAST",
    "FuzzyLogicFitnessEvaluator",
    "FuzzyLogicEvolutionarySearch",
]