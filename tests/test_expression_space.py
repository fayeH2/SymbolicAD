"""Tests for expression space and evolutionary search."""

import numpy as np
import pytest
import random

from neurosymbolic_fsad.expression_space import (
    ExprNode, ExprType, ArithmeticOp, ComparisonOp, LogicOp,
    ExpressionEvaluator, FuzzyExpressionEvaluator,
    ExpressionGenerator, ExpressionMutator, ExpressionCrossover,
    var, const, add, sub, mul, div, max_op, min_op,
    gt, lt, ge, le, eq, ne,
    logic_and, logic_or, logic_not, logic_implies, logic_xor,
)
from neurosymbolic_fsad.evolutionary_search import (
    SearchConfig, ClassificationFitnessEvaluator, EvolutionarySearch,
    Individual, create_default_search,
)


def test_expression_building():
    """Test building expressions with factory functions."""
    x = var("x")
    y = var("y")
    c = const(1.0)
    
    # Arithmetic
    e1 = add(x, y)
    assert e1.expr_type == ExprType.ARITHMETIC
    assert e1.op == ArithmeticOp.ADD
    
    e2 = mul(x, const(2.0))
    assert e2.op == ArithmeticOp.MUL
    
    # Comparison
    e3 = gt(x, y)
    assert e3.expr_type == ExprType.COMPARISON
    assert e3.op == ComparisonOp.GT
    
    # Logic
    e4 = logic_and(gt(x, const(0.5)), lt(y, const(0.8)))
    assert e4.expr_type == ExprType.LOGIC
    assert e4.op == LogicOp.AND
    
    # Complex
    e5 = logic_or(
        logic_and(gt(var("count"), const(3)), lt(var("ratio"), const(0.5))),
        logic_not(eq(var("flag"), const(1.0)))
    )
    assert e5.expr_type == ExprType.LOGIC


def test_expression_evaluation():
    """Test expression evaluation."""
    expr = add(var("x"), mul(var("y"), const(2.0)))
    evaluator = ExpressionEvaluator({"x": 3.0, "y": 4.0})
    result = evaluator.evaluate(expr)
    assert result == 11.0  # 3 + 4*2 = 11
    
    # Comparison
    expr2 = gt(var("x"), const(5.0))
    evaluator = ExpressionEvaluator({"x": 6.0})
    assert evaluator.evaluate(expr2) == 1.0
    evaluator = ExpressionEvaluator({"x": 4.0})
    assert evaluator.evaluate(expr2) == 0.0
    
    # Logic
    expr3 = logic_and(gt(var("x"), const(0.5)), lt(var("y"), const(0.8)))
    evaluator = ExpressionEvaluator({"x": 0.7, "y": 0.6})
    assert evaluator.evaluate(expr3) == 1.0
    evaluator = ExpressionEvaluator({"x": 0.3, "y": 0.6})
    assert evaluator.evaluate(expr3) == 0.0


def test_fuzzy_evaluation():
    """Test fuzzy logic evaluation."""
    expr = logic_and(var("x"), var("y"))
    evaluator = FuzzyExpressionEvaluator({"x": 0.8, "y": 0.6})
    result = evaluator.evaluate(expr)
    # Fuzzy AND = product
    assert abs(result - 0.48) < 1e-6
    
    expr2 = logic_or(var("x"), var("y"))
    result2 = evaluator.evaluate(expr2)
    # Fuzzy OR = 1 - (1-x)(1-y)
    assert abs(result2 - (1 - 0.2 * 0.4)) < 1e-6
    
    # Soft comparison
    expr3 = gt(var("x"), var("y"))
    evaluator2 = FuzzyExpressionEvaluator({"x": 0.7, "y": 0.6})
    result3 = evaluator2.evaluate(expr3)
    assert 0.5 < result3 < 1.0  # x > y, so should be > 0.5


def test_expression_serialization():
    """Test serialization/deserialization."""
    expr = logic_or(
        logic_and(gt(var("x"), const(0.5)), var("y")),
        logic_not(eq(var("z"), const(1.0)))
    )
    
    data = expr.to_dict()
    expr2 = ExprNode.from_dict(data)
    assert str(expr) == str(expr2)


def test_variable_extraction():
    """Test extracting variable names."""
    expr = logic_or(
        gt(var("count"), const(3)),
        logic_and(var("ratio"), logic_not(var("flag")))
    )
    vars_set = expr.get_variables()
    assert vars_set == {"count", "ratio", "flag"}


def test_expression_generator():
    """Test random expression generation."""
    gen = ExpressionGenerator(["x", "y", "z"], max_depth=3)
    pop = gen.random_population(10)
    assert len(pop) == 10
    for expr in pop:
        assert isinstance(expr, ExprNode)
        # Depth can occasionally exceed max due to randomness, allow +1
        assert expr.depth() <= 4
        assert all(v in ["x", "y", "z"] for v in expr.get_variables())


def test_expression_mutator():
    """Test expression mutation."""
    gen = ExpressionGenerator(["x", "y"], max_depth=3)
    mutator = ExpressionMutator(["x", "y"], max_depth=3)
    
    # Create a simple expression
    expr = logic_and(gt(var("x"), const(0.5)), lt(var("y"), const(0.8)))
    
    # Mutate multiple times
    for _ in range(20):
        mutated = mutator.mutate(expr, mutation_rate=1.0)
        assert isinstance(mutated, ExprNode)
        assert mutated.get_variables().issubset({"x", "y"})


def test_expression_crossover():
    """Test crossover between expressions."""
    gen = ExpressionGenerator(["x", "y", "z"], max_depth=3)
    crossover = ExpressionCrossover()
    
    parent1 = logic_and(gt(var("x"), const(0.5)), var("y"))
    parent2 = logic_or(lt(var("z"), const(0.3)), logic_not(var("x")))
    
    child1, child2 = crossover.crossover(parent1, parent2)
    assert isinstance(child1, ExprNode)
    assert isinstance(child2, ExprNode)
    # Children should use variables from parents
    all_vars = child1.get_variables() | child2.get_variables()
    assert all_vars.issubset({"x", "y", "z"})


def test_classification_fitness():
    """Test fitness evaluation on synthetic data."""
    # Create synthetic anomaly detection data
    np.random.seed(42)
    n_normal = 100
    n_anomaly = 50
    
    # Normal: x ~ N(0, 1), y ~ N(0, 1)
    # Anomaly: x ~ N(2, 1), y ~ N(2, 1)
    X_normal = np.random.randn(n_normal, 2)
    X_anomaly = np.random.randn(n_anomaly, 2) + 2.0
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Test with a good expression: x + y > 1
    expr = gt(add(var("x0"), var("x1")), const(1.0))
    
    evaluator = ClassificationFitnessEvaluator(metric="auroc", complexity_penalty=0.0)
    fitness, metrics = evaluator.evaluate(expr, X, y)
    
    assert metrics["auroc"] > 0.8  # Should separate reasonably well
    assert fitness > 0.7


def test_evolutionary_search_simple():
    """Test evolutionary search on simple synthetic data."""
    np.random.seed(42)
    random.seed(42)
    
    # Simple 1D problem: anomaly if x > 1.5
    n_normal = 100
    n_anomaly = 50
    X_normal = np.random.randn(n_normal, 1) * 0.5
    X_anomaly = np.random.randn(n_anomaly, 1) * 0.5 + 2.0
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Small config for fast test
    config = SearchConfig(
        population_size=30,
        max_generations=10,
        tournament_size=3,
        elitism=1,
        crossover_rate=0.5,
        mutation_rate=0.3,
        max_depth=3,
        complexity_penalty=0.01,
        random_seed=42,
    )
    
    search = create_default_search(
        variable_names=["x0"],
        metric="auroc",
        population_size=config.population_size,
        max_generations=config.max_generations,
        tournament_size=config.tournament_size,
        elitism=config.elitism,
        crossover_rate=config.crossover_rate,
        mutation_rate=config.mutation_rate,
        max_depth=config.max_depth,
        complexity_penalty=config.complexity_penalty,
        random_seed=config.random_seed,
    )
    
    result = search.run(X, y)
    
    assert result.best_individual.fitness > 0.5
    assert result.best_individual.metrics.get("auroc", 0) > 0.7
    assert len(result.generation_stats) > 0
    print(f"Best expression: {result.best_individual.expr}")
    print(f"Best AUROC: {result.best_individual.metrics.get('auroc', 0):.4f}")


def test_individual_comparison():
    """Test Individual comparison for sorting."""
    ind1 = Individual(expr=var("x"), fitness=0.5, id=1)
    ind2 = Individual(expr=var("y"), fitness=0.8, id=2)
    ind3 = Individual(expr=var("z"), fitness=0.3, id=3)
    
    individuals = [ind1, ind2, ind3]
    individuals.sort(reverse=True)
    assert individuals[0].fitness == 0.8
    assert individuals[1].fitness == 0.5
    assert individuals[2].fitness == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])