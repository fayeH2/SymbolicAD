"""Leakage-free SymbolicDet-style evolution for static FSAD rules.

The visual vocabulary and calibrated atom truths are learned from known-normal
support images.  Evolution never sees a query image or query label: positive
search examples are deterministic counterfactual truth assignments generated
from normal support rows.  A local Qwen3-VL advisor may inject closed-schema
normality ASTs during evolution.  The selected AST is frozen and is executable
without the LLM at inference time.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .fuzzy_logic import FuzzyLogicProgram, LogicValidationError


_BINARY_OPERATORS = ("AND", "OR", "IMPLIES", "XOR")


def normalize_normality_ast(ast: Mapping[str, Any]) -> Dict[str, Any]:
    """Strictly normalise the closed Qwen/GA AST schema.

    Qwen sometimes emits ``NOT`` with a one-element ``args`` list.  The fuzzy
    executor accepts that spelling, but the stored deployment schema uses the
    unambiguous singular ``arg`` field.
    """
    if not isinstance(ast, Mapping):
        raise LogicValidationError("AST node must be an object")
    op = str(ast.get("op", "")).upper()
    if op == "ATOM":
        if set(ast) != {"op", "name"} or not str(ast.get("name", "")).strip():
            raise LogicValidationError("ATOM must contain exactly op and name")
        return {"op": "ATOM", "name": str(ast["name"]).strip()}
    if op == "NOT":
        if set(ast) == {"op", "arg"}:
            child = ast["arg"]
        elif set(ast) == {"op", "args"} and isinstance(ast["args"], list) and len(ast["args"]) == 1:
            child = ast["args"][0]
        else:
            raise LogicValidationError("NOT must contain exactly one arg")
        return {"op": "NOT", "arg": normalize_normality_ast(child)}
    if op not in _BINARY_OPERATORS or set(ast) != {"op", "args"}:
        raise LogicValidationError(f"Unknown or malformed AST operator: {op}")
    args = ast["args"]
    if not isinstance(args, list):
        raise LogicValidationError(f"{op}.args must be a list")
    if op in {"IMPLIES", "XOR"} and len(args) != 2:
        raise LogicValidationError(f"{op} requires exactly two operands")
    if op in {"AND", "OR"} and len(args) < 2:
        raise LogicValidationError(f"{op} requires at least two operands")
    return {"op": op, "args": [normalize_normality_ast(child) for child in args]}


def atom_names(ast: Mapping[str, Any]) -> Tuple[str, ...]:
    """Return reachable atom names in left-to-right order."""
    op = str(ast.get("op", "")).upper()
    if op == "ATOM":
        return (str(ast["name"]),)
    if op == "NOT":
        return atom_names(ast["arg"])
    result: List[str] = []
    for child in ast.get("args", []):
        result.extend(atom_names(child))
    return tuple(result)


def node_count(ast: Mapping[str, Any]) -> int:
    op = str(ast.get("op", "")).upper()
    if op == "ATOM":
        return 1
    if op == "NOT":
        return 1 + node_count(ast["arg"])
    return 1 + sum(node_count(child) for child in ast.get("args", []))


def canonical_ast(ast: Mapping[str, Any]) -> str:
    """Canonical JSON identity; commutative children are order-normalised."""
    op = str(ast.get("op", "")).upper()
    if op == "ATOM":
        payload: Dict[str, Any] = {"op": "ATOM", "name": str(ast["name"])}
    elif op == "NOT":
        payload = {"op": "NOT", "arg": json.loads(canonical_ast(ast["arg"]))}
    else:
        children = [json.loads(canonical_ast(child)) for child in ast.get("args", [])]
        if op in {"AND", "OR", "XOR"}:
            children.sort(key=lambda item: json.dumps(item, sort_keys=True))
        payload = {"op": op, "args": children}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def anomaly_ast(normality_ast: Mapping[str, Any]) -> Dict[str, Any]:
    return {"op": "NOT", "arg": copy.deepcopy(dict(normality_ast))}


def validate_normality_ast(
    ast: Mapping[str, Any], available_atoms: Sequence[str]
) -> Tuple[str, ...]:
    """Validate a normality tree and require every selected atom exactly once."""
    normalized = normalize_normality_ast(ast)
    names = atom_names(normalized)
    if not names:
        raise LogicValidationError("A candidate AST must contain at least one atom")
    if len(names) != len(set(names)):
        raise LogicValidationError("A candidate AST cannot repeat an atom")
    unknown = sorted(set(names) - set(available_atoms))
    if unknown:
        raise LogicValidationError(f"Unknown candidate atoms: {unknown}")
    FuzzyLogicProgram(anomaly_ast(normalized), names)
    return names


def evaluate_ast_matrix(
    ast: Mapping[str, Any], atom_truths: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Vectorised product-fuzzy evaluation of a data-only AST."""
    op = str(ast.get("op", "")).upper()
    if op == "ATOM":
        return np.clip(np.asarray(atom_truths[str(ast["name"])], dtype=float), 0.0, 1.0)
    if op == "NOT":
        return 1.0 - evaluate_ast_matrix(ast["arg"], atom_truths)
    values = [evaluate_ast_matrix(child, atom_truths) for child in ast.get("args", [])]
    if not values:
        raise ValueError(f"{op} has no operands")
    if op == "AND":
        result = np.ones_like(values[0])
        for value in values:
            result *= value
        return np.clip(result, 0.0, 1.0)
    if op == "OR":
        result = np.ones_like(values[0])
        for value in values:
            result *= 1.0 - value
        return np.clip(1.0 - result, 0.0, 1.0)
    if op == "IMPLIES":
        if len(values) != 2:
            raise ValueError("IMPLIES requires two operands")
        return np.clip(1.0 - values[0] * (1.0 - values[1]), 0.0, 1.0)
    if op == "XOR":
        if len(values) != 2:
            raise ValueError("XOR requires two operands")
        return np.clip(values[0] + values[1] - 2.0 * values[0] * values[1], 0.0, 1.0)
    raise ValueError(f"Unknown operator: {op}")


def support_truth_matrix(
    candidate_audit: Sequence[Mapping[str, Any]],
    eligible_atoms: Sequence[str],
) -> np.ndarray:
    """Convert leave-one-support-out facts into calibrated normality truths."""
    by_id = {str(row["id"]): row for row in candidate_audit}
    lengths = {len(by_id[name]["leave_one_out_values"]) for name in eligible_atoms}
    if len(lengths) != 1:
        raise ValueError("LOO atom rows have inconsistent lengths")
    columns = []
    for name in eligible_atoms:
        row = by_id[name]
        raw = np.asarray(row["leave_one_out_values"], dtype=float)
        excess = np.maximum(0.0, raw - float(row["upper"]))
        violation = np.tanh(excess / max(float(row["scale"]), 1e-8))
        columns.append(1.0 - violation)
    return np.column_stack(columns)


def counterfactual_search_data(
    normal_truths: np.ndarray,
    atom_names_order: Sequence[str],
    *,
    seed: int,
    variants_per_atom: int = 6,
    pair_variants: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Create support-only pseudo events by violating one or two atom truths.

    Each pseudo-event begins from a real LOO normal row.  Only symbolic truth
    values are altered; no query image, defect name, or query label is read.
    """
    normal = np.asarray(normal_truths, dtype=float)
    if normal.ndim != 2 or normal.shape[1] != len(atom_names_order):
        raise ValueError("normal_truths shape does not match atom names")
    rng = np.random.default_rng(seed)
    rows = [row.copy() for row in normal]
    labels = [0] * len(normal)
    provenance = [
        {"kind": "real_normal_loo", "normal_index": index, "violated_atoms": []}
        for index in range(len(normal))
    ]
    severities = np.asarray([0.15, 0.30, 0.45, 0.60, 0.78, 0.95], dtype=float)
    for atom_index, atom in enumerate(atom_names_order):
        for normal_index in range(len(normal)):
            for variant in range(variants_per_atom):
                row = normal[normal_index].copy()
                severity = float(severities[variant % len(severities)])
                row[atom_index] = min(row[atom_index], 1.0 - severity)
                rows.append(row)
                labels.append(1)
                provenance.append({
                    "kind": "single_atom_counterfactual",
                    "normal_index": normal_index,
                    "violated_atoms": [atom],
                    "severity": severity,
                })
    if len(atom_names_order) > 1:
        for left in range(len(atom_names_order)):
            for right in range(left + 1, len(atom_names_order)):
                for normal_index in range(len(normal)):
                    for _ in range(pair_variants):
                        row = normal[normal_index].copy()
                        severity = float(rng.choice(severities[2:]))
                        row[left] = min(row[left], 1.0 - severity)
                        row[right] = min(row[right], 1.0 - severity)
                        rows.append(row)
                        labels.append(1)
                        provenance.append({
                            "kind": "two_atom_counterfactual",
                            "normal_index": normal_index,
                            "violated_atoms": [
                                atom_names_order[left], atom_names_order[right]
                            ],
                            "severity": severity,
                        })
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=np.uint8), provenance


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 64
    generations: int = 40
    tournament_size: int = 5
    elite_count: int = 6
    crossover_rate: float = 0.5
    mutation_rate: float = 0.3
    complexity_penalty: float = 0.004
    normal_false_positive_penalty: float = 0.20
    early_stop_patience: int = 12
    random_seed: int = 668
    llm_feedback_interval: int = 5
    seed_all_conjunction: bool = True


@dataclass
class Individual:
    normality_ast: Dict[str, Any]
    fitness: float = -math.inf
    metrics: Optional[Dict[str, float]] = None
    origin: str = "random"

    def key(self) -> str:
        return canonical_ast(self.normality_ast)


def _atom(name: str) -> Dict[str, Any]:
    return {"op": "ATOM", "name": name}


def _random_tree(atoms: Sequence[str], rng: random.Random) -> Dict[str, Any]:
    nodes = [_atom(name) for name in atoms]
    rng.shuffle(nodes)
    while len(nodes) > 1:
        left = nodes.pop(rng.randrange(len(nodes)))
        right = nodes.pop(rng.randrange(len(nodes)))
        op = rng.choice(_BINARY_OPERATORS)
        node: Dict[str, Any] = {"op": op, "args": [left, right]}
        if rng.random() < 0.12:
            node = {"op": "NOT", "arg": node}
        nodes.append(node)
    return nodes[0]


def initial_population(
    atoms: Sequence[str], size: int, rng: random.Random,
    llm_seed_asts: Sequence[Mapping[str, Any]] = (),
    seed_all_conjunction: bool = True,
) -> List[Individual]:
    result: List[Individual] = []
    for ast in llm_seed_asts:
        try:
            validate_normality_ast(ast, atoms)
        except (LogicValidationError, KeyError, TypeError, ValueError):
            continue
        result.append(Individual(copy.deepcopy(dict(ast)), origin="llm_seed"))
    for name in atoms:
        result.append(Individual(_atom(name), origin="single_atom_seed"))
    if len(atoms) > 1:
        for op in _BINARY_OPERATORS:
            result.append(Individual({
                "op": op, "args": [_atom(atoms[0]), _atom(atoms[1])]
            }, origin="semantic_seed"))
        if seed_all_conjunction:
            result.append(Individual({
                "op": "AND", "args": [_atom(name) for name in atoms]
            }, origin="all_normality_conjunction"))
    while len(result) < size:
        subset_size = rng.randint(1, len(atoms))
        subset = rng.sample(list(atoms), subset_size)
        result.append(Individual(_random_tree(subset, rng), origin="random"))
    deduplicated: Dict[str, Individual] = {}
    for individual in result:
        deduplicated.setdefault(individual.key(), individual)
    result = list(deduplicated.values())
    while len(result) < size:
        subset = rng.sample(list(atoms), rng.randint(1, len(atoms)))
        candidate = Individual(_random_tree(subset, rng), origin="random")
        # With one or two eligible atoms the closed tree space can contain
        # fewer unique shallow programs than the requested population.  GA
        # populations may legitimately contain duplicate genotypes.
        result.append(candidate)
    return result[:size]


def _paths(ast: Mapping[str, Any], path: Tuple[Any, ...] = ()) -> List[Tuple[Any, ...]]:
    result = [path]
    op = str(ast.get("op", "")).upper()
    if op == "NOT":
        result.extend(_paths(ast["arg"], path + ("arg",)))
    elif op != "ATOM":
        for index, child in enumerate(ast.get("args", [])):
            result.extend(_paths(child, path + ("args", index)))
    return result


def _get(ast: Mapping[str, Any], path: Tuple[Any, ...]) -> Mapping[str, Any]:
    node: Any = ast
    for item in path:
        node = node[item]
    return node


def _replace(ast: Mapping[str, Any], path: Tuple[Any, ...], value: Mapping[str, Any]):
    if not path:
        return copy.deepcopy(dict(value))
    result: Any = copy.deepcopy(dict(ast))
    target = result
    for item in path[:-1]:
        target = target[item]
    target[path[-1]] = copy.deepcopy(dict(value))
    return result


def crossover(
    left: Mapping[str, Any], right: Mapping[str, Any], atoms: Sequence[str],
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    left_paths, right_paths = _paths(left), _paths(right)
    for _ in range(24):
        lp, rp = rng.choice(left_paths), rng.choice(right_paths)
        lchild, rchild = _get(left, lp), _get(right, rp)
        first = _replace(left, lp, rchild)
        second = _replace(right, rp, lchild)
        try:
            validate_normality_ast(first, atoms)
            validate_normality_ast(second, atoms)
        except (LogicValidationError, KeyError, TypeError, ValueError):
            continue
        return first, second
    return copy.deepcopy(dict(left)), copy.deepcopy(dict(right))


def mutate(ast: Mapping[str, Any], atoms: Sequence[str], rng: random.Random) -> Dict[str, Any]:
    result = copy.deepcopy(dict(ast))
    names = set(atom_names(result))
    action = rng.choice(("operator", "swap", "replace", "add", "drop", "negate"))
    if action == "operator":
        candidates = [path for path in _paths(result) if str(_get(result, path).get("op", "")).upper() in _BINARY_OPERATORS]
        if candidates:
            path = rng.choice(candidates)
            node = copy.deepcopy(dict(_get(result, path)))
            node["op"] = rng.choice([op for op in _BINARY_OPERATORS if op != node["op"]])
            result = _replace(result, path, node)
    elif action == "swap":
        candidates = [path for path in _paths(result) if str(_get(result, path).get("op", "")).upper() in _BINARY_OPERATORS]
        if candidates:
            path = rng.choice(candidates)
            node = copy.deepcopy(dict(_get(result, path)))
            node["args"] = list(reversed(node["args"]))
            result = _replace(result, path, node)
    elif action == "replace":
        missing = list(set(atoms) - names)
        leaves = [path for path in _paths(result) if str(_get(result, path).get("op", "")).upper() == "ATOM"]
        if missing and leaves:
            result = _replace(result, rng.choice(leaves), _atom(rng.choice(missing)))
    elif action == "add":
        missing = list(set(atoms) - names)
        if missing:
            result = {
                "op": rng.choice(_BINARY_OPERATORS),
                "args": [result, _atom(rng.choice(missing))],
            }
    elif action == "drop":
        candidates = [path for path in _paths(result) if str(_get(result, path).get("op", "")).upper() in _BINARY_OPERATORS]
        if candidates:
            node = _get(result, (path := rng.choice(candidates)))
            result = _replace(result, path, rng.choice(list(node["args"])))
    else:
        paths = _paths(result)
        path = rng.choice(paths)
        node = _get(result, path)
        if str(node.get("op", "")).upper() == "NOT":
            result = _replace(result, path, node["arg"])
        else:
            result = _replace(result, path, {"op": "NOT", "arg": copy.deepcopy(dict(node))})
    try:
        validate_normality_ast(result, atoms)
    except (LogicValidationError, KeyError, TypeError, ValueError):
        return copy.deepcopy(dict(ast))
    return result


class SymbolicEvolutionSearch:
    """Genetic search over executable product-fuzzy normality programs."""

    def __init__(
        self,
        atom_order: Sequence[str],
        config: EvolutionConfig,
        llm_advisor: Optional[
            Callable[[int, Sequence[Mapping[str, Any]]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]
        ] = None,
        llm_seed_asts: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if not atom_order:
            raise ValueError("Evolution needs at least one eligible atom")
        self.atoms = tuple(atom_order)
        self.config = config
        self.llm_advisor = llm_advisor
        self.llm_seed_asts = tuple(llm_seed_asts)
        self.rng = random.Random(config.random_seed)

    def _fitness(
        self, individual: Individual, X: np.ndarray, y: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        values = {name: X[:, index] for index, name in enumerate(self.atoms)}
        normal_truth = evaluate_ast_matrix(individual.normality_ast, values)
        anomaly = np.clip(1.0 - normal_truth, 0.0, 1.0)
        if np.ptp(anomaly) <= 1e-12:
            auroc = 0.5
        else:
            auroc = float(roc_auc_score(y, anomaly))
        aupr = float(average_precision_score(y, anomaly))
        normal_scores = anomaly[y == 0]
        normal_fpr = float(np.mean(normal_scores > 1e-8))
        complexity = node_count(individual.normality_ast) + 1
        raw = 0.5 * auroc + 0.5 * aupr
        fitness = (
            raw
            - self.config.complexity_penalty * complexity
            - self.config.normal_false_positive_penalty * normal_fpr
        )
        metrics = {
            "auroc": auroc,
            "aupr": aupr,
            "raw_objective": raw,
            "normal_trigger_rate": normal_fpr,
            "normal_mean_anomaly": float(np.mean(normal_scores)),
            "node_count": complexity,
            "selected_atom_count": len(atom_names(individual.normality_ast)),
            "fitness": float(fitness),
        }
        return float(fitness), metrics

    def _evaluate(self, population: Iterable[Individual], X: np.ndarray, y: np.ndarray) -> None:
        for individual in population:
            individual.fitness, individual.metrics = self._fitness(individual, X, y)

    def _tournament(self, population: Sequence[Individual]) -> Individual:
        rows = self.rng.sample(
            list(population), min(self.config.tournament_size, len(population))
        )
        return max(rows, key=lambda row: row.fitness)

    def run(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        population = initial_population(
            self.atoms, self.config.population_size, self.rng,
            self.llm_seed_asts,
            seed_all_conjunction=self.config.seed_all_conjunction,
        )
        self._evaluate(population, X, y)
        history: List[Dict[str, Any]] = []
        llm_feedback: List[Dict[str, Any]] = []
        lineage: List[Dict[str, Any]] = []
        best_seen = -math.inf
        stale = 0
        hall: Dict[str, Individual] = {}

        for generation in range(self.config.generations + 1):
            population.sort(key=lambda row: row.fitness, reverse=True)
            for row in population[:16]:
                previous = hall.get(row.key())
                if previous is None or row.fitness > previous.fitness:
                    hall[row.key()] = copy.deepcopy(row)
            best = population[0]
            history.append({
                "generation": generation,
                "best_fitness": float(best.fitness),
                "mean_fitness": float(np.mean([row.fitness for row in population])),
                "best_normality_ast": best.normality_ast,
                "best_anomaly_ast": anomaly_ast(best.normality_ast),
                "best_metrics": best.metrics,
                "unique_population": len({row.key() for row in population}),
            })
            if best.fitness > best_seen + 1e-12:
                best_seen, stale = best.fitness, 0
            else:
                stale += 1
            if generation >= self.config.generations or stale >= self.config.early_stop_patience:
                break

            injected: List[Individual] = []
            if (
                self.llm_advisor is not None
                and generation > 0
                and generation % self.config.llm_feedback_interval == 0
            ):
                top = [
                    {
                        "normality_ast": row.normality_ast,
                        "fitness": row.fitness,
                        "metrics": row.metrics,
                    }
                    for row in population[:5]
                ]
                suggestions, audit = self.llm_advisor(generation, top)
                accepted = []
                for suggestion in suggestions:
                    try:
                        validate_normality_ast(suggestion, self.atoms)
                    except (LogicValidationError, KeyError, TypeError, ValueError):
                        continue
                    candidate = Individual(suggestion, origin=f"qwen_feedback_g{generation}")
                    self._evaluate([candidate], X, y)
                    injected.append(candidate)
                    accepted.append(candidate.normality_ast)
                llm_feedback.append({
                    **audit,
                    "generation": generation,
                    "accepted_suggestions": accepted,
                })

            next_population = [copy.deepcopy(row) for row in population[: self.config.elite_count]]
            next_population.extend(injected)
            while len(next_population) < self.config.population_size:
                first, second = self._tournament(population), self._tournament(population)
                if self.rng.random() < self.config.crossover_rate:
                    child_asts = crossover(
                        first.normality_ast, second.normality_ast, self.atoms, self.rng
                    )
                    operation = "crossover"
                else:
                    child_asts = (
                        copy.deepcopy(first.normality_ast),
                        copy.deepcopy(second.normality_ast),
                    )
                    operation = "clone"
                for child_ast in child_asts:
                    if self.rng.random() < self.config.mutation_rate:
                        child_ast = mutate(child_ast, self.atoms, self.rng)
                        operation += "+mutation"
                    child = Individual(child_ast, origin=operation)
                    next_population.append(child)
                    lineage.append({
                        "generation": generation + 1,
                        "origin": operation,
                        "parent_keys": [first.key(), second.key()],
                        "child_key": child.key(),
                    })
                    if len(next_population) >= self.config.population_size:
                        break
            dedup: Dict[str, Individual] = {}
            for row in next_population:
                dedup.setdefault(row.key(), row)
            next_population = list(dedup.values())
            while len(next_population) < self.config.population_size:
                subset = self.rng.sample(
                    list(self.atoms), self.rng.randint(1, len(self.atoms))
                )
                candidate = Individual(_random_tree(subset, self.rng), origin="immigrant")
                next_population.append(candidate)
            population = next_population[: self.config.population_size]
            self._evaluate(population, X, y)

        hall_rows = sorted(hall.values(), key=lambda row: row.fitness, reverse=True)
        best = hall_rows[0]
        return {
            "schema_version": "symbolicdet-fsad-evolution-v8",
            "best_normality_ast": best.normality_ast,
            "best_anomaly_ast": anomaly_ast(best.normality_ast),
            "best_active_atoms": list(atom_names(best.normality_ast)),
            "best_fitness": float(best.fitness),
            "best_origin": best.origin,
            "best_metrics": best.metrics,
            "config": asdict(self.config),
            "generation_history": history,
            "llm_feedback": llm_feedback,
            "hall_of_fame": [
                {
                    "normality_ast": row.normality_ast,
                    "anomaly_ast": anomaly_ast(row.normality_ast),
                    "fitness": row.fitness,
                    "metrics": row.metrics,
                    "origin": row.origin,
                }
                for row in hall_rows[:20]
            ],
            "lineage_event_count": len(lineage),
            "lineage_sample": lineage[:200],
        }


def qwen_seed_asts(eligible_atoms: Sequence[str]) -> List[Dict[str, Any]]:
    """Deterministic semantic seeds derived from the Qwen-admitted atom set."""
    atoms = list(eligible_atoms)
    seeds: List[Dict[str, Any]] = [_atom(name) for name in atoms]
    if len(atoms) >= 2:
        seeds.extend([
            {"op": "AND", "args": [_atom(name) for name in atoms]},
            {"op": "OR", "args": [_atom(name) for name in atoms]},
        ])
        if "lead_count" in atoms and "lead_geometry" in atoms:
            seeds.append({
                "op": "IMPLIES",
                "args": [_atom("lead_count"), _atom("lead_geometry")],
            })
    return seeds


__all__ = [
    "EvolutionConfig", "Individual", "SymbolicEvolutionSearch",
    "anomaly_ast", "atom_names", "canonical_ast", "counterfactual_search_data",
    "evaluate_ast_matrix", "node_count", "qwen_seed_asts",
    "normalize_normality_ast", "support_truth_matrix", "validate_normality_ast",
]
