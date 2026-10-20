"""Category-agnostic orthogonal rules and rule-conditioned INP fusion.

This module contains no dataset class names and no category branches.  Qwen's
closed-schema proposal is compiled into generic component predicates, redundant
predicates are removed by intervention-response orthogonality, and an evolved
fuzzy-logic AST is executed on spatial atom maps.  The resulting logic map
modulates INP encoder/decoder residual maps *before* their layer aggregation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .causal_evolution import GroupRobustEvolutionSearch
from .features import prepare_inp_aligned_rgb
from .fuzzy_logic import FuzzyLogicProgram
from .generic_relations import (
    FACT_BY_PREDICATE,
    GenericComponentGrounder,
    compile_generic_rules,
)
from .symbolicdet_evolution import EvolutionConfig, atom_names


PREDICATE_FAMILY = {
    "count_eq": "cardinality",
    "arrangement_eq": "spatial_configuration",
    "ordered_spacing": "spatial_configuration",
    "similar_size": "scale_geometry",
    "similar_orientation": "orientation_geometry",
    "appearance_consistency": "appearance",
}


def _symbolic_candidate_pool(proposal: Mapping[str, Any]) -> Dict[str, Any]:
    """Expand an LLM seed into the closed executable operator space.

    These rows are hypotheses for cross-fitted symbolic search, not accepted
    rules.  This mirrors SymbolicDet's separation between semantic guidance
    and expression search and avoids treating an LLM's omitted property as
    proof that the property is irrelevant.
    """
    expanded = dict(proposal)
    entity = proposal.get("primary_entity")
    if not isinstance(entity, Mapping):
        expanded["rules"] = []
        return expanded
    expected = int(entity.get("expected_count", 1))
    arrangement = str(entity.get("arrangement", "free"))
    rows = []
    seen = set()
    for row in proposal.get("rules", []):
        predicate = str(row.get("predicate", ""))
        if predicate not in PREDICATE_FAMILY or predicate in seen:
            continue
        rows.append({**dict(row), "candidate_origin": "llm_seed"})
        seen.add(predicate)
    searchable = ["count_eq", "arrangement_eq", "appearance_consistency"]
    if expected > 1:
        searchable.extend(["similar_size", "similar_orientation"])
    if expected > 1 and arrangement not in {"single", "free"}:
        searchable.append("ordered_spacing")
    for predicate in searchable:
        if predicate in seen:
            continue
        rows.append({
            "id": f"search_{predicate}",
            "predicate": predicate,
            "importance": 0.75,
            "confidence": 0.70,
            "family": PREDICATE_FAMILY[predicate],
            "reason": (
                "Closed-DSL symbolic-search candidate; must pass support "
                "cross-fitting and grounded counterfactual validation."
            ),
            "candidate_origin": "symbolic_operator_enumeration",
        })
        seen.add(predicate)
    expanded["rules"] = rows
    return expanded


def _violation(value: float, rule: Mapping[str, Any]) -> float:
    excess = max(0.0, float(value) - float(rule["upper"]))
    return float(math.tanh(excess / max(float(rule["soft_scale"]), 1e-8)))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if np.std(left) < 1e-10 or np.std(right) < 1e-10:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(abs(np.corrcoef(left, right)[0, 1]))


def _inp_view(image: Any) -> np.ndarray:
    return prepare_inp_aligned_rgb(image, input_size=448, crop_size=392)


def compile_orthogonal_program(
    proposal: Mapping[str, Any],
    support_images: Sequence[Any],
    *,
    image_size: int = 256,
    max_rules: int = 4,
    maximum_response_correlation: float = 0.85,
    search_seed: int = 668,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compile a generic grounder, orthogonal atoms and evolved logic AST."""
    prepared = [_inp_view(item) for item in support_images]
    search_proposal = _symbolic_candidate_pool(proposal)
    grounder, rules, crossfit = compile_generic_rules(
        search_proposal, prepared, image_size=image_size,
        minimum_cv_auc=0.60, maximum_normal_fpr=0.25,
    )
    if grounder is None or not grounder.prototype or not rules:
        library = {
            "schema_version": "11.0-orthogonal-generic-symbolic",
            "grounder": grounder.state_dict() if grounder is not None else None,
            "rules": [], "logic": None, "abstained": True,
            "abstention_reason": "no support-stable generic predicates",
        }
        return library, {
            "candidate_pool": search_proposal.get("rules", []),
            "crossfit": crossfit, "orthogonality": [], "evolution": None,
        }

    # Every candidate is tested on every candidate's grounded intervention.
    interventions = []
    for source_index, image in enumerate(prepared):
        for source_rule in rules:
            changed = grounder.counterfactual(image, str(source_rule["predicate"]))
            facts, _ = grounder.extract(changed)
            interventions.append((source_index, str(source_rule["predicate"]), facts))
    signatures: Dict[str, np.ndarray] = {}
    utilities: Dict[str, Dict[str, float]] = {}
    for rule in rules:
        predicate = str(rule["predicate"])
        signature = np.asarray([
            _violation(facts[str(rule["fact"])], rule)
            for _, _, facts in interventions
        ], dtype=float)
        own = np.asarray([
            value for value, (_, source_predicate, _) in zip(signature, interventions)
            if source_predicate == predicate
        ], dtype=float)
        cross = np.asarray([
            value for value, (_, source_predicate, _) in zip(signature, interventions)
            if source_predicate != predicate
        ], dtype=float)
        own_mean = float(np.mean(own)) if len(own) else 0.0
        cross_mean = float(np.mean(cross)) if len(cross) else 0.0
        specificity = own_mean - cross_mean
        identifier = str(rule["id"])
        signatures[identifier] = signature
        utilities[identifier] = {
            "own_intervention_response": own_mean,
            "cross_intervention_response": cross_mean,
            "specificity": specificity,
            "utility": own_mean + 0.35 * max(specificity, 0.0),
        }

    ranked = sorted(
        rules,
        key=lambda row: (
            utilities[str(row["id"])]["utility"],
            float(row.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    selected, used_families, audit = [], set(), []
    for rule in ranked:
        identifier = str(rule["id"])
        predicate = str(rule["predicate"])
        family = PREDICATE_FAMILY.get(predicate, predicate)
        correlations = {
            str(other["id"]): _correlation(
                signatures[identifier], signatures[str(other["id"])]
            ) for other in selected
        }
        maximum_correlation = max(correlations.values(), default=0.0)
        reason = None
        if family in used_families:
            reason = "duplicate_semantic_family"
        elif maximum_correlation > maximum_response_correlation:
            reason = "redundant_intervention_response"
        elif utilities[identifier]["own_intervention_response"] <= 1e-6:
            reason = "not_grounded_by_own_intervention"
        elif len(selected) >= max_rules:
            reason = "orthogonal_rule_budget"
        accepted = reason is None
        if accepted:
            selected.append(dict(rule))
            used_families.add(family)
        audit.append({
            "id": identifier, "predicate": predicate, "family": family,
            **utilities[identifier], "correlations_to_selected": correlations,
            "maximum_abs_response_correlation": maximum_correlation,
            "accepted": accepted,
            "reason": "accepted_orthogonal" if accepted else reason,
        })

    if not selected:
        library = {
            "schema_version": "11.0-orthogonal-generic-symbolic",
            "grounder": grounder.state_dict(), "rules": [], "logic": None,
            "abstained": True, "abstention_reason": "no orthogonal grounded predicates",
        }
        return library, {
            "candidate_pool": search_proposal.get("rules", []),
            "crossfit": crossfit, "orthogonality": audit, "evolution": None,
        }

    atom_order = [str(rule["id"]) for rule in selected]
    rows, labels, groups = [], [], []
    for image in prepared:
        facts, _ = grounder.extract(image)
        rows.append([1.0 - _violation(facts[str(rule["fact"])], rule) for rule in selected])
        labels.append(0); groups.append("normal_support")
    for source_index, source_predicate, facts in interventions:
        if source_predicate not in {str(rule["predicate"]) for rule in selected}:
            continue
        rows.append([1.0 - _violation(facts[str(rule["fact"])], rule) for rule in selected])
        labels.append(1); groups.append(f"counterfactual/{source_predicate}")
    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.uint8)
    search = GroupRobustEvolutionSearch(
        atom_order,
        EvolutionConfig(
            population_size=48, generations=35, tournament_size=5,
            elite_count=6, crossover_rate=0.5, mutation_rate=0.3,
            complexity_penalty=0.004, normal_false_positive_penalty=0.2,
            early_stop_patience=10, random_seed=search_seed,
        ),
        groups,
        atom_cost=0.015,
        worst_group_weight=0.30,
        normal_invariance_weight=0.25,
    ).run(X, y)
    active = list(search["best_active_atoms"])
    selected = [rule for rule in selected if str(rule["id"]) in active]
    library = {
        "schema_version": "11.0-orthogonal-generic-symbolic",
        "grounder": grounder.state_dict(),
        "rules": selected,
        "logic": {
            "normality_ast": search["best_normality_ast"],
            "anomaly_ast": search["best_anomaly_ast"],
            "semantics": "product_fuzzy_logic",
        },
        "abstained": False,
        "runtime_requires_llm": False,
        "category_specific_python_branching": False,
    }
    return library, {
        "candidate_pool": search_proposal.get("rules", []),
        "crossfit": crossfit,
        "orthogonality": audit,
        "evolution": search,
        "search_matrix": X.tolist(),
        "search_labels": y.tolist(),
        "search_groups": groups,
    }


@dataclass
class GenericLogicInference:
    symbolic_anomaly: float
    symbolic_map: np.ndarray
    facts: Dict[str, float]
    atoms: Dict[str, Dict[str, Any]]
    localization_maps: Dict[str, np.ndarray]
    trace: list
    diagnostics: Dict[str, Any]
    calibration_localizers: Dict[str, np.ndarray] | None = None


class OrthogonalSymbolicEngine:
    def __init__(self, library: Mapping[str, Any]):
        self.library = dict(library)
        self.rules = list(self.library.get("rules", []))
        state = self.library.get("grounder")
        self.grounder = None
        if isinstance(state, Mapping):
            self.grounder = GenericComponentGrounder(
                int(state["expected_count"]), str(state["arrangement"]),
                int(state["image_size"]), state=state,
            )
        self.active_ids = [str(row["id"]) for row in self.rules]
        self.program = None
        if self.active_ids:
            self.program = FuzzyLogicProgram(
                self.library["logic"]["anomaly_ast"], self.active_ids
            )

    def evaluate(self, image: Any) -> GenericLogicInference:
        size = int(self.grounder.image_size) if self.grounder is not None else 256
        if self.grounder is None or self.program is None:
            zero = np.zeros((size, size), dtype=np.float32)
            return GenericLogicInference(0.0, zero, {}, {}, {}, [], {"abstained": True})
        facts, fact_maps, diagnostics = self.grounder.extract_spatial(_inp_view(image))
        scalar_truths, map_truths, atoms, localizers = {}, {}, {}, {}
        for rule in self.rules:
            identifier = str(rule["id"])
            fact = str(rule["fact"])
            violation = _violation(facts[fact], rule)
            localizer = np.asarray(fact_maps[fact], dtype=np.float32)
            scalar_truths[identifier] = 1.0 - violation
            map_truths[identifier] = 1.0 - violation * localizer
            localizers[identifier] = localizer
            atoms[identifier] = {
                "predicate": str(rule["predicate"]), "fact": fact,
                "fact_value": float(facts[fact]), "upper": float(rule["upper"]),
                "scale": float(rule["soft_scale"]), "violation": violation,
                "family": PREDICATE_FAMILY.get(str(rule["predicate"]), str(rule["predicate"])),
            }
        symbolic, trace = self.program.evaluate(scalar_truths)
        symbolic_map, _ = self.program.evaluate(map_truths)
        return GenericLogicInference(
            float(symbolic), np.asarray(symbolic_map, dtype=np.float32),
            {key: float(value) for key, value in facts.items()}, atoms,
            localizers, trace, diagnostics,
        )


def _localized_top_mean(layer_map: np.ndarray, localizer: np.ndarray, fraction=0.10) -> float:
    layer_map = np.asarray(layer_map, dtype=np.float64)
    localizer = np.asarray(localizer, dtype=np.float64)
    selected = localizer > 0.05
    if not np.any(selected):
        return 0.0
    values = layer_map[selected] * localizer[selected]
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(np.partition(values, len(values) - count)[-count:].mean())


class RuleConditionedINPFusion:
    """Fuse at per-layer residual features, before INP layer aggregation."""

    def __init__(self, engine: OrthogonalSymbolicEngine, calibration: Mapping[str, Any]):
        self.engine = engine
        self.calibration = dict(calibration)

    @classmethod
    def calibrate(
        cls,
        engine: OrthogonalSymbolicEngine,
        support_images: Sequence[Any],
        support_layer_maps: np.ndarray,
    ) -> "RuleConditionedINPFusion":
        layers = np.asarray(support_layer_maps, dtype=np.float32)
        if layers.ndim != 4 or len(layers) != len(support_images):
            raise ValueError("support layer maps must be [N,L,H,W] and aligned")
        values = {name: [[] for _ in range(layers.shape[1])] for name in engine.active_ids}
        for image, layer_stack in zip(support_images, layers):
            result = engine.evaluate(image)
            for name in engine.active_ids:
                calibration_localizer = result.localization_maps[name]
                if result.calibration_localizers is not None:
                    calibration_localizer = result.calibration_localizers.get(
                        name, calibration_localizer
                    )
                for layer_index, layer_map in enumerate(layer_stack):
                    values[name][layer_index].append(
                        _localized_top_mean(layer_map, calibration_localizer)
                    )
        rules = {}
        for name, layer_rows in values.items():
            thresholds, scales = [], []
            for row in layer_rows:
                array = np.asarray(row, dtype=float)
                thresholds.append(float(np.max(array)))
                median = float(np.median(array))
                mad = float(np.median(np.abs(array - median)))
                scales.append(max(1.4826 * mad, float(np.std(array)) * 0.25, 0.002))
            rules[name] = {"thresholds": thresholds, "scales": scales}
        return cls(engine, {
            "schema_version": "11.0-rule-conditioned-inp-feature-fusion",
            "support_count": len(support_images),
            "layer_count": int(layers.shape[1]),
            "query_relative_evidence_quantile": 0.995,
            "rules": rules,
            "query_or_test_used": False,
        })

    def infer(
        self, image: Any, layer_maps: np.ndarray,
        baseline_map: np.ndarray | None = None,
        baseline_score: float | None = None,
    ) -> Tuple[float, np.ndarray, GenericLogicInference, Dict[str, Any]]:
        layers = np.clip(np.asarray(layer_maps, dtype=np.float64), 0.0, 1.0)
        if layers.ndim != 3:
            raise ValueError("query layer maps must be [L,H,W]")
        result = self.engine.evaluate(image)
        scalar_truths, map_truths = {}, {}
        rule_rows = {}
        layer_logits = np.zeros(layers.shape[0], dtype=np.float64)
        for name in self.engine.active_ids:
            atom = result.atoms[name]
            violation = float(atom["violation"])
            state = self.calibration["rules"][name]
            evidences, gates = [], []
            for layer_index, layer_map in enumerate(layers):
                evidence = _localized_top_mean(
                    layer_map, result.localization_maps[name]
                )
                # A detector miss is not anomaly evidence by itself.  Require
                # the rule-localized residual to be extreme both relative to
                # normal supports and relative to the current query.  The
                # latter protects against detector instability and global
                # illumination/texture shifts without using query labels.
                quantile = float(self.calibration.get(
                    "query_relative_evidence_quantile", 0.995
                ))
                query_threshold = float(np.quantile(layer_map, quantile))
                effective_threshold = max(
                    float(state["thresholds"][layer_index]), query_threshold
                )
                query_scale = max(
                    query_threshold - float(np.quantile(layer_map, 0.99)),
                    float(state["scales"][layer_index]), 0.002,
                )
                excess = max(0.0, evidence - effective_threshold)
                gate = math.tanh(excess / max(query_scale, 1e-8))
                evidences.append(evidence); gates.append(gate)
            gates_array = np.asarray(gates, dtype=np.float64)
            aggregate_gate = float(np.max(gates_array))
            supported = violation * aggregate_gate
            scalar_truths[name] = 1.0 - supported
            # Per-layer agreement is collapsed only after it has modulated the
            # individual residual feature maps.
            agreement_map = np.max(
                gates_array[:, None, None] * layers, axis=0
            )
            supported_map = np.clip(
                violation * result.localization_maps[name] * agreement_map,
                0.0, 1.0,
            )
            map_truths[name] = 1.0 - supported_map
            layer_logits += violation * gates_array
            rule_rows[name] = {
                **atom, "layer_evidence": evidences, "layer_gates": gates,
                "aggregate_gate": aggregate_gate,
                "supported_violation": supported,
            }
        if not self.engine.active_ids:
            base = (
                np.asarray(baseline_map, dtype=np.float64)
                if baseline_map is not None else np.mean(layers, axis=0)
            )
            flat = base.reshape(-1)
            count = max(1, flat.size // 100)
            score = (
                float(baseline_score) if baseline_score is not None
                else float(np.partition(flat, flat.size - count)[-count:].mean())
            )
            return score, base.astype(np.float32), result, {}
        confirmed_scalar, logic_trace = self.engine.program.evaluate(scalar_truths)
        confirmed_map, _ = self.engine.program.evaluate(map_truths)
        confirmed_map = np.asarray(confirmed_map, dtype=np.float64)
        logits = layer_logits - np.max(layer_logits)
        weights = np.exp(logits)
        weights /= max(np.sum(weights), 1e-12)
        # High-order logic modulates every residual feature layer before those
        # layers are aggregated into the final anomaly map.
        modulated = layers * (1.0 + confirmed_map[None])
        feature_fused = np.sum(weights[:, None, None] * modulated, axis=0)
        if baseline_map is not None:
            # Residual adapter: preserve the exact baseline prediction unless
            # executable symbolic logic and INP features jointly add evidence.
            neutral = np.mean(layers, axis=0)
            enhancement = np.maximum(feature_fused - neutral, 0.0)
            fused = np.clip(
                np.asarray(baseline_map, dtype=np.float64) + enhancement,
                0.0, 1.0,
            )
        else:
            fused = np.clip(feature_fused, 0.0, 1.0)
        flat = fused.reshape(-1)
        count = max(1, flat.size // 100)
        feature_score = float(
            np.partition(flat, flat.size - count)[-count:].mean()
        )
        score = (
            max(float(baseline_score), feature_score)
            if baseline_score is not None else feature_score
        )
        trace = {
            "confirmed_symbolic_anomaly": float(confirmed_scalar),
            "logic_trace": logic_trace,
            "layer_attention_weights": weights.tolist(),
            "rules": rule_rows,
        }
        return score, fused.astype(np.float32), result, trace


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "GenericLogicInference", "OrthogonalSymbolicEngine", "PREDICATE_FAMILY",
    "RuleConditionedINPFusion", "compile_orthogonal_program", "save_json",
]
