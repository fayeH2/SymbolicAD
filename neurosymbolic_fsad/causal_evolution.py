"""Group-robust evolutionary objective for grounded counterfactuals."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .symbolicdet_evolution import (
    EvolutionConfig,
    Individual,
    SymbolicEvolutionSearch,
    atom_names,
    evaluate_ast_matrix,
    node_count,
)


class GroupRobustEvolutionSearch(SymbolicEvolutionSearch):
    """Optimise real grounded groups instead of atom-indexed pseudo labels.

    The objective combines global ranking, worst intervention-family ranking,
    normal invariance and minimum-description-length costs.  No term rewards a
    particular operator or the inclusion of all atoms.
    """

    def __init__(
        self,
        atom_order: Sequence[str],
        config: EvolutionConfig,
        groups: Sequence[str],
        *,
        atom_cost: float = 0.025,
        worst_group_weight: float = 0.30,
        normal_invariance_weight: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(atom_order, config, **kwargs)
        self.groups = np.asarray(groups, dtype=str)
        self.atom_cost = float(atom_cost)
        self.worst_group_weight = float(worst_group_weight)
        self.normal_invariance_weight = float(normal_invariance_weight)

    def _fitness(
        self, individual: Individual, X: np.ndarray, y: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        if len(self.groups) != len(y):
            raise ValueError("groups length does not match evolutionary samples")
        values = {name: X[:, index] for index, name in enumerate(self.atoms)}
        normality = evaluate_ast_matrix(individual.normality_ast, values)
        anomaly = np.clip(1.0 - normality, 0.0, 1.0)
        auroc = float(roc_auc_score(y, anomaly)) if np.ptp(anomaly) else 0.5
        aupr = float(average_precision_score(y, anomaly))
        normal = anomaly[y == 0]
        normal_mean = float(np.mean(normal))
        normal_fpr = float(np.mean(normal > 0.05))

        group_aurocs: Dict[str, float] = {}
        for group in sorted(set(self.groups[y == 1].tolist())):
            selected = (y == 0) | ((y == 1) & (self.groups == group))
            scores = anomaly[selected]
            labels = y[selected]
            group_aurocs[group] = (
                float(roc_auc_score(labels, scores)) if np.ptp(scores) else 0.5
            )
        mean_group = float(np.mean(list(group_aurocs.values())))
        worst_group = float(np.min(list(group_aurocs.values())))
        selected_count = len(atom_names(individual.normality_ast))
        complexity = node_count(individual.normality_ast) + 1

        # Weights sum to one before explicit risk penalties.  Atom and node
        # costs implement MDL; they do not single out AND or any other operator.
        ranking = (
            0.25 * auroc
            + 0.15 * aupr
            + (0.60 - self.worst_group_weight) * mean_group
            + self.worst_group_weight * worst_group
        )
        fitness = (
            ranking
            - self.normal_invariance_weight * normal_mean
            - 0.10 * normal_fpr
            - self.atom_cost * selected_count
            - self.config.complexity_penalty * complexity
        )
        metrics: Dict[str, float] = {
            "auroc": auroc,
            "aupr": aupr,
            "mean_group_auroc": mean_group,
            "worst_group_auroc": worst_group,
            "normal_trigger_rate_at_0.05": normal_fpr,
            "normal_mean_anomaly": normal_mean,
            "selected_atom_count": float(selected_count),
            "node_count": float(complexity),
            "fitness": float(fitness),
        }
        metrics.update({f"group_auroc/{key}": value for key, value in group_aurocs.items()})
        return float(fitness), metrics


__all__ = ["GroupRobustEvolutionSearch"]
