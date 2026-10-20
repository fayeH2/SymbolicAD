"""Support-validated static normality rules (v2).

Qwen is a semantic candidate generator, never the final judge.  Candidate
directions, thresholds, reliability and weights are estimated from normal
nuisance views and held-out synthetic local corruptions.  A rule can be
rejected, and the whole symbolic branch can abstain back to INP-Former.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .features import FEATURE_CATALOG, extract_features, prepare_inp_aligned_rgb


RAW_MAP_MAGNITUDE_FEATURES = {
    "map_mean", "map_std", "map_p95", "map_p99", "map_top1_mean",
}

RELIABLE_FEATURE_CATALOG = deepcopy(FEATURE_CATALOG)
RELIABLE_FEATURE_CATALOG.update({
    "map_shape_l1": {
        "description": "L1 distance between normalized map topology and the normal prototype",
        "default_direction": "low", "tolerance": 0.015,
    },
    "map_shape_cosine": {
        "description": "cosine distance between normalized map topology and the normal prototype",
        "default_direction": "low", "tolerance": 0.005,
    },
    "map_shape_z": {
        "description": "RMS distance between per-image standardized map topology and normal",
        "default_direction": "low", "tolerance": 0.08,
    },
    "map_shape_nearest_l1": {
        "description": "L1 distance to the closest normal map-topology prototype",
        "default_direction": "low", "tolerance": 0.015,
    },
})


def _grid(array: np.ndarray, size: int) -> np.ndarray:
    score = np.asarray(array, dtype=np.float32).squeeze()
    if score.ndim != 2:
        raise ValueError(f"Expected 2D map, got {score.shape}")
    return cv2.resize(score, (size, size), interpolation=cv2.INTER_AREA)


def _mass_vector(array: np.ndarray) -> np.ndarray:
    vector = np.maximum(np.asarray(array, dtype=np.float64).reshape(-1), 0.0)
    return vector / max(float(vector.sum()), 1e-12)


def _z_vector(array: np.ndarray) -> np.ndarray:
    vector = np.asarray(array, dtype=np.float64).reshape(-1)
    return (vector - vector.mean()) / max(float(vector.std()), 1e-8)


def build_shape_context(
    normal_maps: Sequence[np.ndarray], grid_size: int = 8, max_prototypes: int = 32,
) -> Dict[str, Any]:
    if not len(normal_maps):
        raise ValueError("normal_maps must not be empty")
    grids = np.stack([_grid(item, grid_size) for item in normal_maps])
    mass = np.stack([_mass_vector(item) for item in grids])
    z_values = np.stack([_z_vector(item) for item in grids])
    if len(grids) > max_prototypes:
        indices = np.linspace(0, len(grids) - 1, max_prototypes).round().astype(int)
        mass_prototypes = mass[indices]
    else:
        mass_prototypes = mass
    mass_mean = mass.mean(axis=0)
    mass_mean /= max(float(mass_mean.sum()), 1e-12)
    return {
        "grid_size": int(grid_size),
        "mass_mean": mass_mean.astype(float).tolist(),
        "z_mean": z_values.mean(axis=0).astype(float).tolist(),
        "mass_prototypes": mass_prototypes.astype(float).tolist(),
    }


def extract_shape_features(anomaly_map: np.ndarray, context: Mapping[str, Any]) -> Dict[str, float]:
    grid_size = int(context["grid_size"])
    grid = _grid(anomaly_map, grid_size)
    mass = _mass_vector(grid)
    z_value = _z_vector(grid)
    mean_mass = np.asarray(context["mass_mean"], dtype=np.float64)
    mean_z = np.asarray(context["z_mean"], dtype=np.float64)
    prototypes = np.asarray(context["mass_prototypes"], dtype=np.float64)
    cosine = 1.0 - float(np.dot(mass, mean_mass)) / max(
        float(np.linalg.norm(mass) * np.linalg.norm(mean_mass)), 1e-12
    )
    return {
        "map_shape_l1": float(np.abs(mass - mean_mass).sum()),
        "map_shape_cosine": float(max(cosine, 0.0)),
        "map_shape_z": float(np.sqrt(np.mean((z_value - mean_z) ** 2))),
        "map_shape_nearest_l1": float(np.min(np.abs(prototypes - mass[None]).sum(axis=1))),
    }


def extract_reliable_features(
    image: Any,
    anomaly_map: np.ndarray,
    shape_context: Mapping[str, Any],
    input_size: int = 448,
    crop_size: int = 392,
) -> Dict[str, float]:
    aligned = prepare_inp_aligned_rgb(image, input_size=input_size, crop_size=crop_size)
    facts = extract_features(aligned, anomaly_map)
    facts.update(extract_shape_features(anomaly_map, shape_context))
    return facts


def _rank(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # Average ranks for ties; the arrays here are small, so clarity wins.
    for value in np.unique(values):
        mask = values == value
        ranks[mask] = ranks[mask].mean()
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank, right_rank = _rank(left), _rank(right)
    if left_rank.std() < 1e-12 or right_rank.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _fit_envelope(values: Sequence[float], direction: str, tolerance: float) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    robust_scale = max(
        float(tolerance), 1.4826 * mad,
        float(np.subtract(*np.quantile(values, [0.75, 0.25]))) / 1.349,
        1e-8,
    )
    low_q, high_q = (float(item) for item in np.quantile(values, [0.025, 0.975]))
    lower = low_q - float(tolerance) if direction in {"stable", "high"} else None
    upper = high_q + float(tolerance) if direction in {"stable", "low"} else None
    return {
        "center": center,
        "lower": lower,
        "upper": upper,
        "soft_scale": robust_scale,
        "normal_q025": low_q,
        "normal_q975": high_q,
        "normal_mad": mad,
    }


def violation(rule: Mapping[str, Any], value: float) -> float:
    distance = 0.0
    if rule.get("lower") is not None and value < float(rule["lower"]):
        distance = float(rule["lower"]) - value
    elif rule.get("upper") is not None and value > float(rule["upper"]):
        distance = value - float(rule["upper"])
    return float(math.tanh(distance / max(float(rule["soft_scale"]), 1e-8)))


def _score_rule(rule: Mapping[str, Any], rows: Sequence[Mapping[str, float]]) -> np.ndarray:
    feature = str(rule["feature"])
    return np.asarray([violation(rule, float(row[feature])) for row in rows])


def _safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(np.unique(labels)) < 2 or float(scores.max() - scores.min()) < 1e-12:
        return 0.5
    return float(roc_auc_score(labels, scores))


def mine_reliable_rules(
    normal_rows: Sequence[Mapping[str, float]],
    pseudo_rows: Sequence[Mapping[str, float]],
    normal_baseline: Sequence[float],
    pseudo_baseline: Sequence[float],
    normal_groups: Sequence[int],
    pseudo_groups: Sequence[int],
    qwen_proposal: Optional[Mapping[str, Any]] = None,
    max_rules: int = 8,
    minimum_cv_auc: float = 0.58,
    maximum_normal_fpr: float = 0.25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Retrieve rules with leave-one-support-out empirical verification."""
    if not normal_rows or not pseudo_rows:
        raise ValueError("Both normal and pseudo rows are required")
    normal_groups = np.asarray(normal_groups, dtype=int)
    pseudo_groups = np.asarray(pseudo_groups, dtype=int)
    all_rows = list(normal_rows) + list(pseudo_rows)
    all_labels = np.r_[np.zeros(len(normal_rows), dtype=int), np.ones(len(pseudo_rows), dtype=int)]
    all_baseline = np.r_[np.asarray(normal_baseline), np.asarray(pseudo_baseline)]
    semantic = {
        str(item.get("feature")): item
        for item in (qwen_proposal or {}).get("rules", [])
        if str(item.get("feature")) in RELIABLE_FEATURE_CATALOG
    }
    groups = sorted(set(normal_groups.tolist()) & set(pseudo_groups.tolist()))
    audits: List[Dict[str, Any]] = []

    for feature, metadata in RELIABLE_FEATURE_CATALOG.items():
        if feature in RAW_MAP_MAGNITUDE_FEATURES:
            audits.append({
                "feature": feature, "accepted": False,
                "rejection": "absolute map magnitude duplicates baseline and is in-sample biased",
            })
            continue
        best = None
        for direction in ("stable", "low", "high"):
            cv_labels, cv_scores = [], []
            for held_out in groups:
                train_values = [
                    row[feature] for row, group in zip(normal_rows, normal_groups)
                    if group != held_out
                ]
                if not train_values:
                    continue
                fold_rule = {
                    "feature": feature, "direction": direction,
                    **_fit_envelope(train_values, direction, float(metadata["tolerance"])),
                }
                held_normal = [row for row, group in zip(normal_rows, normal_groups)
                               if group == held_out]
                held_pseudo = [row for row, group in zip(pseudo_rows, pseudo_groups)
                               if group == held_out]
                cv_scores.extend(_score_rule(fold_rule, held_normal).tolist())
                cv_labels.extend([0] * len(held_normal))
                cv_scores.extend(_score_rule(fold_rule, held_pseudo).tolist())
                cv_labels.extend([1] * len(held_pseudo))
            cv_scores_array = np.asarray(cv_scores, dtype=float)
            cv_labels_array = np.asarray(cv_labels, dtype=int)
            auc = _safe_auc(cv_labels_array, cv_scores_array)
            normal_values = cv_scores_array[cv_labels_array == 0]
            pseudo_values = cv_scores_array[cv_labels_array == 1]
            candidate = {
                "direction": direction,
                "cv_auc": auc,
                "cv_normal_fpr": float(np.mean(normal_values > 0.0)),
                "cv_pseudo_trigger_rate": float(np.mean(pseudo_values > 0.0)),
                "cv_normal_mean": float(normal_values.mean()),
                "cv_pseudo_mean": float(pseudo_values.mean()),
            }
            if best is None or (candidate["cv_auc"], -candidate["cv_normal_fpr"]) > (
                best["cv_auc"], -best["cv_normal_fpr"]
            ):
                best = candidate

        final_rule = {
            "id": f"normal_{feature}",
            "feature": feature,
            "direction": best["direction"],
            **_fit_envelope(
                [row[feature] for row in normal_rows], best["direction"],
                float(metadata["tolerance"]),
            ),
        }
        scores = _score_rule(final_rule, all_rows)
        correlation = abs(_spearman(scores, all_baseline))
        semantic_prior = 1.0 if feature in semantic else 0.85
        separation = max(best["cv_auc"] - 0.5, 0.0) * 2.0
        stability = max(1.0 - best["cv_normal_fpr"], 0.0)
        novelty = max(1.0 - 0.65 * correlation, 0.1)
        quality = separation * stability * novelty * semantic_prior
        accepted = bool(
            best["cv_auc"] >= minimum_cv_auc
            and best["cv_normal_fpr"] <= maximum_normal_fpr
            and best["cv_pseudo_mean"] > best["cv_normal_mean"] + 0.015
        )
        audit = {
            "feature": feature,
            "accepted": accepted,
            "quality": float(quality),
            "baseline_abs_spearman": float(correlation),
            "qwen_candidate": feature in semantic,
            **best,
        }
        if not accepted:
            audit["rejection"] = "failed CV AUC/FPR/separation gate"
        audits.append(audit)
        if accepted:
            final_rule.update({
                "description": str(semantic.get(feature, {}).get(
                    "reason", metadata["description"]
                )),
                "weight": float(max(quality, 1e-6)),
                "validation": {key: value for key, value in audit.items()
                               if key not in {"feature", "accepted"}},
            })
            audit["_rule"] = final_rule
            audit["_scores"] = scores

    selected: List[Dict[str, Any]] = []
    selected_scores: List[np.ndarray] = []
    for audit in sorted(audits, key=lambda item: float(item.get("quality", 0.0)), reverse=True):
        if not audit.get("accepted"):
            continue
        candidate_scores = audit["_scores"]
        redundancy = max(
            (abs(_spearman(candidate_scores, prior)) for prior in selected_scores),
            default=0.0,
        )
        if redundancy > 0.90:
            audit["accepted"] = False
            audit["rejection"] = f"redundant with selected rule (|rho|={redundancy:.3f})"
            continue
        selected.append(audit["_rule"])
        selected_scores.append(candidate_scores)
        if len(selected) >= max_rules:
            break

    for audit in audits:
        audit.pop("_rule", None)
        audit.pop("_scores", None)
    return selected, audits


def _baseline_calibration(normal_scores: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(normal_scores, dtype=float)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = max(
        1.4826 * mad,
        float(np.subtract(*np.quantile(values, [0.75, 0.25]))) / 1.349,
        float(np.ptp(values)) / 6.0,
        1e-6,
    )
    return {"center": center, "scale": scale}


def _baseline_z(scores: Sequence[float], calibration: Mapping[str, float]) -> np.ndarray:
    result = (np.asarray(scores, dtype=float) - float(calibration["center"])) / max(
        float(calibration["scale"]), 1e-8
    )
    # Do not clip: clipping creates ties and can change AUROC even when the
    # symbolic branch abstains.  Robust scaling alone remains monotonic.
    return result


class _FusionHead(nn.Module):
    def __init__(self, rule_count: int):
        super().__init__()
        self.rule_logits = nn.Parameter(torch.zeros(rule_count))
        self.strength_raw = nn.Parameter(torch.tensor(-1.1))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def coefficients(self):
        # Preserve the baseline slope exactly and cap the entire symbolic
        # residual.  The earlier unconstrained head reduced baseline weight and
        # assigned >1 to every rule, so synthetic counterexamples overwhelmed
        # real-test ordering.
        beta = torch.ones((), device=self.bias.device)
        strength = 4.0 * torch.sigmoid(self.strength_raw)
        rules = torch.softmax(self.rule_logits, dim=0) * strength
        return beta, rules

    def forward(self, baseline, rule_values):
        beta, rules = self.coefficients()
        return self.bias + beta * baseline + (rule_values * rules).sum(dim=1)


def _fit_head(
    baseline: np.ndarray,
    rule_matrix: np.ndarray,
    labels: np.ndarray,
    device: str,
    epochs: int,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    head = _FusionHead(rule_matrix.shape[1]).to(device)
    x_base = torch.as_tensor(baseline, dtype=torch.float32, device=device)
    x_rule = torch.as_tensor(rule_matrix, dtype=torch.float32, device=device)
    target = torch.as_tensor(labels, dtype=torch.float32, device=device)
    positive = max(float((labels == 1).sum()), 1.0)
    negative = max(float((labels == 0).sum()), 1.0)
    weights = torch.where(target > 0.5, negative / positive, torch.ones_like(target))
    optimizer = torch.optim.Adam(head.parameters(), lr=0.03)
    for _ in range(epochs):
        logits = head(x_base, x_rule)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target, weight=weights)
        beta, rule_weights = head.coefficients()
        loss = loss + 0.02 * rule_weights.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        beta, rule_weights = head.coefficients()
    return {
        "baseline_weight": float(beta.cpu()),
        "rule_weights": rule_weights.cpu().numpy().astype(float),
        "bias": float(head.bias.detach().cpu()),
        "training_device": device,
        "epochs": int(epochs),
    }


def train_reliable_fusion(
    rules: Sequence[Mapping[str, Any]],
    normal_rows: Sequence[Mapping[str, float]],
    pseudo_rows: Sequence[Mapping[str, float]],
    normal_scores: Sequence[float],
    pseudo_scores: Sequence[float],
    normal_groups: Sequence[int],
    pseudo_groups: Sequence[int],
    device: str = "cuda:0",
    epochs: int = 350,
    seed: int = 20261013,
    minimum_cv_gain: float = 0.005,
    require_cuda: bool = False,
) -> Dict[str, Any]:
    """Train a non-negative sparse fusion head and a CV abstention gate."""
    normal_groups, pseudo_groups = np.asarray(normal_groups), np.asarray(pseudo_groups)
    if require_cuda and device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA head training was required, but {device} is unavailable")
    rows = list(normal_rows) + list(pseudo_rows)
    raw_scores = np.r_[np.asarray(normal_scores), np.asarray(pseudo_scores)]
    labels = np.r_[np.zeros(len(normal_rows), dtype=int), np.ones(len(pseudo_rows), dtype=int)]
    groups = np.r_[normal_groups, pseudo_groups]
    rule_matrix = np.stack([_score_rule(rule, rows) for rule in rules], axis=1) \
        if rules else np.zeros((len(rows), 0), dtype=float)
    calibration_normal = np.r_[
        np.ones(len(normal_rows), dtype=bool), np.zeros(len(pseudo_rows), dtype=bool)
    ]
    counterfactual_mask = np.zeros(len(rows), dtype=bool)

    # Train the head on hard symbolic counterexamples as well as pixel-space
    # corruptions.  Each counterexample keeps a normal baseline score but
    # violates one empirically validated rule; without these, INP perfectly
    # separates the simple CutPaste set and the residual branch receives no
    # learning signal at all.
    counterfactual_count = 0
    if rules:
        counterfactual = rule_matrix[:len(normal_rows)].copy()
        for index in range(len(counterfactual)):
            rule_index = index % len(rules)
            counterfactual[index, rule_index] = max(counterfactual[index, rule_index], 0.85)
        rule_matrix = np.concatenate([rule_matrix, counterfactual], axis=0)
        raw_scores = np.r_[raw_scores, np.asarray(normal_scores)]
        labels = np.r_[labels, np.ones(len(normal_rows), dtype=int)]
        groups = np.r_[groups, normal_groups]
        calibration_normal = np.r_[calibration_normal, np.zeros(len(normal_rows), dtype=bool)]
        counterfactual_mask = np.r_[counterfactual_mask, np.ones(len(normal_rows), dtype=bool)]
        counterfactual_count = len(normal_rows)
    unique_groups = sorted(set(normal_groups.tolist()) & set(pseudo_groups.tolist()))
    fold_results = []
    for fold, held_out in enumerate(unique_groups):
        train = groups != held_out
        validation = groups == held_out
        calibration = _baseline_calibration(raw_scores[train & calibration_normal])
        train_z = _baseline_z(raw_scores[train], calibration)
        validation_z = _baseline_z(raw_scores[validation], calibration)
        if rules:
            fitted = _fit_head(
                train_z, rule_matrix[train], labels[train], device, epochs, seed + fold
            )
            fused = (
                fitted["baseline_weight"] * validation_z
                + rule_matrix[validation] @ fitted["rule_weights"]
                + fitted["bias"]
            )
        else:
            fitted = {"baseline_weight": 1.0, "rule_weights": np.zeros(0), "bias": 0.0}
            fused = validation_z
        baseline_auc = _safe_auc(labels[validation], validation_z)
        fused_auc = _safe_auc(labels[validation], fused)
        pixel_validation = validation & ~counterfactual_mask
        pixel_baseline_auc = _safe_auc(
            labels[pixel_validation], _baseline_z(raw_scores[pixel_validation], calibration)
        )
        if rules:
            pixel_fused = (
                fitted["baseline_weight"]
                * _baseline_z(raw_scores[pixel_validation], calibration)
                + rule_matrix[pixel_validation] @ fitted["rule_weights"]
                + fitted["bias"]
            )
        else:
            pixel_fused = _baseline_z(raw_scores[pixel_validation], calibration)
        pixel_fused_auc = _safe_auc(labels[pixel_validation], pixel_fused)
        fold_results.append({
            "held_out_support": int(held_out),
            "baseline_auc": baseline_auc,
            "fused_auc": fused_auc,
            "gain": fused_auc - baseline_auc,
            "pixel_baseline_auc": pixel_baseline_auc,
            "pixel_fused_auc": pixel_fused_auc,
            "pixel_gain": pixel_fused_auc - pixel_baseline_auc,
        })

    mean_gain = float(np.mean([item["gain"] for item in fold_results])) if fold_results else 0.0
    worst_gain = float(min((item["gain"] for item in fold_results), default=0.0))
    pixel_mean_gain = float(np.mean([item["pixel_gain"] for item in fold_results])) \
        if fold_results else 0.0
    pixel_worst_gain = float(min((item["pixel_gain"] for item in fold_results), default=0.0))
    enabled = bool(
        rules and mean_gain >= minimum_cv_gain and worst_gain >= -0.05
        and pixel_worst_gain >= -0.01
    )
    final_calibration = _baseline_calibration(normal_scores)
    final_z = _baseline_z(raw_scores, final_calibration)
    final = _fit_head(final_z, rule_matrix, labels, device, epochs, seed + 101) if rules else {
        "baseline_weight": 1.0, "rule_weights": np.zeros(0), "bias": 0.0,
        "training_device": "none", "epochs": 0,
    }
    if not enabled:
        final["baseline_weight"] = 1.0
        final["rule_weights"] = np.zeros(len(rules), dtype=float)
        final["bias"] = 0.0
    return {
        "enabled": enabled,
        "policy": "nonnegative_sparse_residual_with_cv_abstention",
        "baseline_calibration": final_calibration,
        "baseline_weight": float(final["baseline_weight"]),
        "rule_weights": [float(value) for value in final["rule_weights"]],
        "bias": float(final["bias"]),
        "training_device": final["training_device"],
        "epochs": int(final["epochs"]),
        "cv_mean_gain": mean_gain,
        "cv_worst_gain": worst_gain,
        "pixel_cv_mean_gain": pixel_mean_gain,
        "pixel_cv_worst_gain": pixel_worst_gain,
        "cv_folds": fold_results,
        "symbolic_counterfactual_count": counterfactual_count,
        "abstention_reason": None if enabled else (
            "no reliable rule" if not rules else "synthetic leave-one-support-out gain gate failed"
        ),
    }


class ReliableRuleEngine:
    def __init__(self, library: Mapping[str, Any]):
        self.library = dict(library)
        if self.library.get("schema_version") != "2.0":
            raise ValueError("ReliableRuleEngine requires schema_version=2.0")
        self.rules = list(self.library.get("rules", []))
        self.fusion = dict(self.library["fusion"])

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ReliableRuleEngine":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def facts(self, image: Any, anomaly_map: np.ndarray) -> Dict[str, float]:
        preprocessing = self.library.get("preprocessing", {})
        return extract_reliable_features(
            image, anomaly_map, self.library["shape_context"],
            int(preprocessing.get("input_size", 448)),
            int(preprocessing.get("crop_size", 392)),
        )

    def rule_values(self, facts: Mapping[str, float]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        values, evidence = [], []
        for rule in self.rules:
            value = float(facts[rule["feature"]])
            score = violation(rule, value)
            values.append(score)
            evidence.append({
                "rule_id": rule["id"], "feature": rule["feature"], "value": value,
                "lower": rule.get("lower"), "upper": rule.get("upper"),
                "violation": score, "validation": rule.get("validation", {}),
            })
        evidence.sort(key=lambda item: item["violation"], reverse=True)
        return np.asarray(values, dtype=float), evidence

    def baseline_z(self, raw_score: float) -> float:
        calibration = self.fusion["baseline_calibration"]
        value = (float(raw_score) - float(calibration["center"])) / max(
            float(calibration["scale"]), 1e-8
        )
        return float(value)

    def decision_score(self, raw_score: float, facts: Mapping[str, float]):
        values, evidence = self.rule_values(facts)
        if not self.fusion.get("enabled", False):
            # Exact safety fallback: same values, ordering and metrics as the
            # baseline—not merely another monotonic-looking calibration.
            decision = float(raw_score)
        else:
            decision = (
                float(self.fusion["bias"])
                + float(self.fusion["baseline_weight"]) * self.baseline_z(raw_score)
                + float(np.dot(values, np.asarray(self.fusion["rule_weights"], dtype=float)))
            )
        rule_score = float(np.average(values, weights=[rule.get("weight", 1.0) for rule in self.rules])) \
            if len(values) else 0.0
        return float(decision), rule_score, evidence


def build_reliable_library(
    *,
    dataset: str,
    class_name: str,
    support_paths: Sequence[str],
    shape_context: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]],
    audit: Sequence[Mapping[str, Any]],
    fusion: Mapping[str, Any],
    qwen_proposal: Optional[Mapping[str, Any]],
    checkpoint: str,
    checkpoint_sha256: str,
    input_size: int = 448,
    crop_size: int = 392,
) -> Dict[str, Any]:
    return {
        "schema_version": "2.0",
        "dataset": dataset,
        "class_name": class_name,
        "support_paths": list(support_paths),
        "support_only_calibration": True,
        "preprocessing": {"input_size": input_size, "crop_size": crop_size},
        "baseline": {
            "name": "INP-Former", "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
        },
        "shape_context": dict(shape_context),
        "rules": [dict(rule) for rule in rules],
        "candidate_audit": [dict(item) for item in audit],
        "fusion": dict(fusion),
        "qwen": {
            "candidate_generator_only": True,
            "normality_summary": str((qwen_proposal or {}).get("normality_summary", "")),
            "candidate_features": [
                item.get("feature") for item in (qwen_proposal or {}).get("rules", [])
            ],
            "self_reported_weights_used": False,
        },
    }
