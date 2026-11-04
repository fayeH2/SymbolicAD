"""Normal support memory: component prototypes, slots and relation statistics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from symbolicAD.config import Config
from symbolicAD.models.clip_backbone import CLIPBackbone, IMAGE_PREPROCESS_VERSION
from symbolicAD.models.text_embeddings import encode_prompts_cached
from symbolicAD.rules.schema import RuleKB

from .component_maps import (
    normalize_with_support,
    prototype_similarity_map,
    text_similarity_map,
)
from .geometry import containment, fit_gaussian, pair_feature, pair_geometry
from .instance_extractor import (
    component_spatial_prior,
    extract_instances,
    image_to_lab_grid,
    lab_color_similarity,
    named_color_lab,
    select_instances,
)
from .scene_graph import ComponentInstance


SUPPORT_MEMORY_VERSION = "v8_baseline_signature"
LOGIC_CALIBRATION_VERSION = "v4_loo_failure_rate"


def baseline_signature(config: Config) -> str:
    """Stable hash of the baseline configuration that produced cached stats."""
    payload = {
        "baseline": config.model.baseline,
        "feature_source": config.model.feature_source,
        "feature_layers": list(config.model.feature_layers),
        "projected_weight": config.model.projected_weight,
        "knn": config.model.knn,
        "spatial_radius": config.model.spatial_radius,
        "spatial_weight": config.model.spatial_weight,
        "local_radius": config.model.local_radius,
        "local_weight": config.model.local_weight,
        "whiten": config.model.whiten,
        "whiten_components": config.model.whiten_components,
        "image_score_percent": config.model.image_score_percent,
        "image_text_weight": config.model.image_text_weight,
        "foreground_masking": config.model.foreground_masking,
        "foreground_soft": config.model.foreground_soft,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def support_image_signature(images: Sequence) -> str:
    """Stable content hash for the ordered normal support set."""
    digest = hashlib.sha256()
    for image in images:
        array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(np.asarray(array, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def rule_grounding_signature(rule_kb: RuleKB) -> str:
    """Hash the rule fields that determine visual grounding memory."""
    payload = {
        "components": [
            {
                "name": item.name,
                "prompts": item.all_prompts(),
                "role": item.role,
                "part_of": item.part_of,
            }
            for item in rule_kb.components
        ],
        "atoms": [
            {
                "id": atom.atom_id,
                "expression": atom.expression,
                "weight": atom.weight,
                "params": atom.params,
            }
            for atom in rule_kb.atoms
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_seed_prototypes(
    patches: np.ndarray,
    text_maps: np.ndarray,
    max_prototypes: int,
    candidate_masks: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Select high-response support patches without thresholding raw cosine.

    Raw CLIP text/patch cosine is commonly around 0.2--0.4, so applying the
    post-normalisation component threshold (0.55 by default) at this bootstrap
    stage yields no instances and collapses every concept to global features.
    """
    patches = np.asarray(patches, dtype=np.float32)
    text_maps = np.asarray(text_maps, dtype=np.float32)
    if patches.ndim != 4 or text_maps.shape != (
        patches.shape[0],
        patches.shape[2],
        patches.shape[3],
    ):
        raise ValueError(
            f"patches {patches.shape} and text maps {text_maps.shape} are incompatible"
        )
    if candidate_masks is not None:
        candidate_masks = np.asarray(candidate_masks)
        if candidate_masks.shape != text_maps.shape:
            raise ValueError(
                f"candidate masks {candidate_masks.shape} and text maps "
                f"{text_maps.shape} are incompatible"
            )
    limit = max(1, int(max_prototypes))
    per_image = max(1, int(np.ceil(limit / max(patches.shape[0], 1))))
    candidates: List[tuple[float, np.ndarray]] = []
    for image_index in range(patches.shape[0]):
        scores = text_maps[image_index].reshape(-1)
        if candidate_masks is not None:
            valid = np.flatnonzero(candidate_masks[image_index].reshape(-1) > 0)
        else:
            valid = np.arange(scores.size)
        if valid.size == 0:
            valid = np.arange(scores.size)
        count = min(per_image, valid.size)
        local = np.argpartition(scores[valid], valid.size - count)[-count:]
        indices = valid[local]
        features = patches[image_index].reshape(patches.shape[1], -1)
        for index in indices:
            candidates.append((float(scores[index]), features[:, index].copy()))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [feature for _, feature in candidates[:limit]]


@dataclass
class ComponentMemory:
    name: str
    prototypes: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    global_prototype: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    score_q_low: float = 0.0
    score_q_high: float = 1.0
    expected_count: float = 0.0
    count_std: float = 0.0
    exist_q05: float = 0.0
    exist_q50: float = 1.0
    expected_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((1, 1), dtype=np.float32)
    )
    slot_masks: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 1, 1), dtype=np.float32)
    )
    color_prototypes: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )


@dataclass
class RelationMemory:
    relation: str
    component_a: str
    component_b: str
    mean_geometry: np.ndarray
    inverse_covariance: np.ndarray
    pair_prototype: Optional[np.ndarray] = None
    sample_count: int = 0


@dataclass
class SupportMemory:
    category: str
    grid_size: tuple[int, int]
    components: Dict[str, ComponentMemory] = field(default_factory=dict)
    relations: Dict[str, RelationMemory] = field(default_factory=dict)
    baseline_stats: Dict[str, float] = field(default_factory=dict)
    attribute_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    atom_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    logic_stats: Dict[str, float] = field(default_factory=dict)
    feature_dim: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)

    def has_component(self, name: str) -> bool:
        return name in self.components

    def relation(
        self, relation: str, component_a: str, component_b: str
    ) -> Optional[RelationMemory]:
        return self.relations.get(_relation_key(relation, component_a, component_b))


def _relation_key(relation: str, component_a: str, component_b: str) -> str:
    return f"{relation}:{component_a}:{component_b}"


def _best_partner(
    source: ComponentInstance,
    candidates: Sequence[ComponentInstance],
    relation: str,
) -> Optional[ComponentInstance]:
    if not candidates:
        return None
    if relation == "Inside":
        return max(candidates, key=lambda item: containment(source, item))
    # Nearest centre is a robust default for direction / alignment relations.
    return min(
        candidates,
        key=lambda item: (source.center[0] - item.center[0]) ** 2
        + (source.center[1] - item.center[1]) ** 2,
    )


def _slot_masks_from_instances(
    instances_per_image: List[List[ComponentInstance]],
    grid_size: tuple[int, int],
) -> np.ndarray:
    max_slots = max((len(items) for items in instances_per_image), default=0)
    if max_slots == 0:
        return np.zeros((0, grid_size[0], grid_size[1]), dtype=np.float32)
    slots = []
    for slot in range(max_slots):
        accumulated, count = None, 0
        for instances in instances_per_image:
            ordered = sorted(instances, key=lambda item: (item.center[0], item.center[1]))
            if slot >= len(ordered):
                continue
            mask = ordered[slot].mask.astype(np.float32)
            accumulated = mask if accumulated is None else accumulated + mask
            count += 1
        slots.append(accumulated / max(count, 1))
    return np.stack(slots, axis=0).astype(np.float32)


def build_support_memory(
    backbone: CLIPBackbone,
    support_images: Sequence,
    rule_kb: RuleKB,
    config: Config,
    baseline=None,
) -> SupportMemory:
    if not support_images:
        raise ValueError("support set is empty")
    encoded = backbone.encode_images(support_images)
    patches = encoded["patches"]  # [K, D, h, w]
    global_features = encoded["global"]
    count, _, height, width = patches.shape
    grid_size = (height, width)
    color_grids = [image_to_lab_grid(image, grid_size) for image in support_images]
    grounding = config.grounding

    spatial_priors = {
        symbol.name: np.stack(
            [
                component_spatial_prior(
                    image,
                    grid_size,
                    position=rule_kb.position_for(symbol.name),
                    role=symbol.role,
                )
                for image in support_images
            ],
            axis=0,
        ).astype(np.float32)
        for symbol in rule_kb.components
    }

    text_maps: Dict[str, np.ndarray] = {}
    prototypes: Dict[str, List[np.ndarray]] = {}
    for symbol in rule_kb.components:
        prompts = rule_kb.prompts_for(symbol.name)
        maps = text_similarity_map(
            backbone,
            patches,
            prompts,
            cache_dir=config.text_embedding_dir,
            enabled=grounding.text_embedding_cache,
        )
        text_maps[symbol.name] = maps
        collected = _text_seed_prototypes(
            patches,
            maps,
            grounding.max_prototypes_per_component,
            candidate_masks=spatial_priors[symbol.name],
        )
        if not collected:
            collected = [global_features[index] for index in range(count)]
        prototypes[symbol.name] = collected[: grounding.max_prototypes_per_component]

    components: Dict[str, ComponentMemory] = {}
    instances_per_component: Dict[str, List[List[ComponentInstance]]] = {}
    for symbol in rule_kb.components:
        name = symbol.name
        proto = np.stack(prototypes[name], axis=0).astype(np.float32)
        proto_norms = np.linalg.norm(proto, axis=1, keepdims=True)
        proto = proto / np.maximum(proto_norms, 1e-8)
        fused = (
            grounding.alpha * text_maps[name]
            + (1.0 - grounding.alpha) * prototype_similarity_map(patches, proto)
        )
        prior_values = fused[spatial_priors[name] > 0]
        if prior_values.size == 0:
            prior_values = fused.reshape(-1)
        q_low = float(np.quantile(prior_values, grounding.support_quantile_low))
        q_high = float(np.quantile(prior_values, grounding.support_quantile_high))

        image_instances: List[List[ComponentInstance]] = []
        expected = np.zeros(grid_size, dtype=np.float32)
        for image_index in range(count):
            normalized = normalize_with_support(fused[image_index], q_low, q_high)
            prior = spatial_priors[name][image_index]
            normalized *= prior
            if "container" in symbol.role.lower():
                normalized = np.maximum(normalized, 0.6 * prior)
            instances = extract_instances(
                component_map=normalized,
                patch_features=patches[image_index],
                concept=name,
                threshold=grounding.component_threshold,
                min_area=grounding.min_instance_area,
                morphology_kernel=grounding.morphology_kernel,
                color_grid=color_grids[image_index],
            )
            instances = select_instances(instances, rule_kb.count_for(name))
            image_instances.append(instances)
            union = np.zeros(grid_size, dtype=np.float32)
            for instance in instances:
                union = np.maximum(union, instance.mask)
            expected += union
        expected /= float(count)

        counts = np.asarray([len(items) for items in image_instances], dtype=np.float64)
        max_confidences = np.asarray(
            [max((item.confidence for item in items), default=0.0) for items in image_instances],
            dtype=np.float64,
        )
        components[name] = ComponentMemory(
            name=name,
            prototypes=proto,
            global_prototype=proto.mean(axis=0).astype(np.float32),
            score_q_low=q_low,
            score_q_high=q_high,
            expected_count=float(counts.mean()) if counts.size else 0.0,
            count_std=float(counts.std()) if counts.size else 0.0,
            exist_q05=float(np.quantile(max_confidences, 0.05)) if max_confidences.size else 0.0,
            exist_q50=float(np.quantile(max_confidences, 0.50)) if max_confidences.size else 1.0,
            expected_mask=expected.astype(np.float32),
            slot_masks=_slot_masks_from_instances(image_instances, grid_size),
            color_prototypes=np.stack(
                [item.color_feature for items in image_instances for item in items],
                axis=0,
            ).astype(np.float32)
            if any(image_instances)
            else np.zeros((0, 3), dtype=np.float32),
        )
        instances_per_component[name] = image_instances

    relations: Dict[str, RelationMemory] = {}
    for atom in rule_kb.atoms:
        if atom.weight <= 0 or len(atom.components) < 2:
            continue
        component_a, component_b = atom.components[0], atom.components[1]
        if component_a not in instances_per_component:
            continue
        if component_b not in instances_per_component:
            continue
        if atom.kind == "SpatialRelation":
            relation_name = atom.spatial_relation or "Inside"
            samples = []
            pair_vectors = []
            for image_index in range(count):
                list_a = instances_per_component[component_a][image_index]
                list_b = instances_per_component[component_b][image_index]
                for instance_a in list_a:
                    instance_b = _best_partner(instance_a, list_b, relation_name)
                    if instance_b is None:
                        continue
                    samples.append(pair_geometry(instance_a, instance_b))
                    pair_vectors.append(pair_feature(instance_a, instance_b))
            if samples:
                sample_array = np.stack(samples, axis=0)
                mean, inverse = fit_gaussian(sample_array)
                relations[_relation_key(relation_name, component_a, component_b)] = (
                    RelationMemory(
                        relation=relation_name,
                        component_a=component_a,
                        component_b=component_b,
                        mean_geometry=mean.astype(np.float32),
                        inverse_covariance=inverse.astype(np.float32),
                        pair_prototype=np.stack(pair_vectors, axis=0)
                        .mean(axis=0)
                        .astype(np.float32),
                        sample_count=len(samples),
                    )
                )
        elif atom.kind == "OneToOne":
            pair_vectors = []
            for image_index in range(count):
                list_a = instances_per_component[component_a][image_index]
                list_b = instances_per_component[component_b][image_index]
                if not list_a or not list_b:
                    continue
                cost = np.asarray(
                    [
                        [
                            1.0
                            - float(
                                np.dot(instance_a.feature, instance_b.feature)
                            )
                            for instance_b in list_b
                        ]
                        for instance_a in list_a
                    ],
                    dtype=np.float64,
                )
                rows, cols = linear_sum_assignment(cost)
                for row, col in zip(rows, cols):
                    pair_vectors.append(pair_feature(list_a[row], list_b[col]))
            if pair_vectors:
                stack = np.stack(pair_vectors, axis=0)
                # Pair prototypes have no meaningful Gaussian geometry.
                relations[_relation_key("OneToOne", component_a, component_b)] = (
                    RelationMemory(
                        relation="OneToOne",
                        component_a=component_a,
                        component_b=component_b,
                        mean_geometry=np.zeros(5, dtype=np.float32),
                        inverse_covariance=np.eye(5, dtype=np.float32),
                        pair_prototype=stack.mean(axis=0).astype(np.float32),
                        sample_count=len(pair_vectors),
                    )
                )

    attribute_stats: Dict[str, Dict[str, float]] = {}
    for atom in rule_kb.atoms:
        if atom.kind != "AttributeMatch" or not atom.components:
            continue
        component = atom.components[0]
        if component not in instances_per_component:
            continue
        scores: List[float] = []
        color_target = (
            named_color_lab(atom.attribute_value or "")
            if (atom.attribute or "").lower() in {"color", "colour"}
            else None
        )
        if color_target is not None:
            for instances in instances_per_component[component]:
                if instances:
                    scores.append(
                        max(
                            lab_color_similarity(item.color_feature, color_target)
                            for item in instances
                        )
                    )
        else:
            embeddings = encode_prompts_cached(
                backbone,
                atom.attribute_prompts,
                cache_dir=config.text_embedding_dir,
                enabled=grounding.text_embedding_cache,
            )
            for image_index in range(count):
                instances = instances_per_component[component][image_index]
                if not instances:
                    continue
                features = np.stack([item.feature for item in instances], axis=0)
                similarities = features @ embeddings.T
                scores.append(float(similarities.max()))
        if scores:
            values = np.asarray(scores, dtype=np.float64)
            attribute_stats[atom.atom_id] = {
                "q05": float(np.quantile(values, 0.05)),
                "q50": float(np.quantile(values, 0.50)),
                "q95": float(np.quantile(values, 0.95)),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "n": int(values.size),
            }

    baseline_stats: Dict[str, float] = {}
    if baseline is not None:
        baseline_stats = baseline.support_statistics(
            support_images,
            low_quantile=grounding.support_quantile_low,
            high_quantile=grounding.support_quantile_high,
        )

    return SupportMemory(
        category=rule_kb.category,
        grid_size=grid_size,
        components=components,
        relations=relations,
        baseline_stats=baseline_stats,
        attribute_stats=attribute_stats,
        feature_dim=int(patches.shape[1]),
        metadata={
            "support_count": int(count),
            "shot": int(config.shot),
            "seed": int(config.support_seed),
            "alpha": float(grounding.alpha),
            "threshold": float(grounding.component_threshold),
            "min_instance_area": int(grounding.min_instance_area),
            "morphology_kernel": int(grounding.morphology_kernel),
            "support_quantile_low": float(grounding.support_quantile_low),
            "support_quantile_high": float(grounding.support_quantile_high),
            "max_prototypes_per_component": int(
                grounding.max_prototypes_per_component
            ),
            "clip_model": str(backbone.model_name),
            "image_size": int(backbone.image_size),
            "grid_size": [int(height), int(width)],
            "feature_dim": int(patches.shape[1]),
            "preprocess_version": IMAGE_PREPROCESS_VERSION,
            "support_memory_version": SUPPORT_MEMORY_VERSION,
            "logic_calibration_version": LOGIC_CALIBRATION_VERSION,
            "baseline_signature": baseline_signature(config),
            "support_signature": support_image_signature(support_images),
            "rule_grounding_signature": rule_grounding_signature(rule_kb),
            "component_names": [symbol.name for symbol in rule_kb.components],
            "attribute_atom_ids": [
                atom.atom_id
                for atom in rule_kb.atoms
                if atom.kind == "AttributeMatch"
            ],
        },
    )


def save_support_memory(memory: SupportMemory, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {}
    manifest = {
        "category": memory.category,
        "grid_size": list(memory.grid_size),
        "feature_dim": memory.feature_dim,
        "baseline_stats": memory.baseline_stats,
        "attribute_stats": memory.attribute_stats,
        "atom_stats": memory.atom_stats,
        "logic_stats": memory.logic_stats,
        "metadata": memory.metadata,
        "components": [],
        "relations": [],
    }
    for index, component in enumerate(memory.components.values()):
        key = f"c{index}"
        payload[f"{key}_prototypes"] = component.prototypes.astype(np.float32)
        payload[f"{key}_global"] = component.global_prototype.astype(np.float32)
        payload[f"{key}_expected_mask"] = component.expected_mask.astype(np.float32)
        payload[f"{key}_slot_masks"] = component.slot_masks.astype(np.float32)
        payload[f"{key}_colors"] = component.color_prototypes.astype(np.float32)
        manifest["components"].append(
            {
                "key": key,
                "name": component.name,
                "score_q_low": component.score_q_low,
                "score_q_high": component.score_q_high,
                "expected_count": component.expected_count,
                "count_std": component.count_std,
                "exist_q05": component.exist_q05,
                "exist_q50": component.exist_q50,
            }
        )
    for index, relation in enumerate(memory.relations.values()):
        key = f"r{index}"
        payload[f"{key}_mean"] = relation.mean_geometry.astype(np.float32)
        payload[f"{key}_inverse"] = relation.inverse_covariance.astype(np.float32)
        if relation.pair_prototype is not None:
            payload[f"{key}_pair"] = relation.pair_prototype.astype(np.float32)
        manifest["relations"].append(
            {
                "key": key,
                "relation": relation.relation,
                "component_a": relation.component_a,
                "component_b": relation.component_b,
                "sample_count": relation.sample_count,
                "has_pair_prototype": relation.pair_prototype is not None,
            }
        )
    payload["manifest"] = np.asarray(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    )
    np.savez_compressed(path, **payload)
    return path


def load_support_memory(path: str | Path) -> SupportMemory:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        manifest = json.loads(str(payload["manifest"].item()))
        components: Dict[str, ComponentMemory] = {}
        for entry in manifest["components"]:
            key = entry["key"]
            components[entry["name"]] = ComponentMemory(
                name=entry["name"],
                prototypes=payload[f"{key}_prototypes"].astype(np.float32),
                global_prototype=payload[f"{key}_global"].astype(np.float32),
                score_q_low=float(entry["score_q_low"]),
                score_q_high=float(entry["score_q_high"]),
                expected_count=float(entry["expected_count"]),
                count_std=float(entry["count_std"]),
                exist_q05=float(entry.get("exist_q05", 0.0)),
                exist_q50=float(entry.get("exist_q50", 1.0)),
                expected_mask=payload[f"{key}_expected_mask"].astype(np.float32),
                slot_masks=payload[f"{key}_slot_masks"].astype(np.float32),
                color_prototypes=(
                    payload[f"{key}_colors"].astype(np.float32)
                    if f"{key}_colors" in payload
                    else np.zeros((0, 3), dtype=np.float32)
                ),
            )
        relations: Dict[str, RelationMemory] = {}
        for entry in manifest["relations"]:
            key = entry["key"]
            pair = (
                payload[f"{key}_pair"].astype(np.float32)
                if entry.get("has_pair_prototype")
                else None
            )
            relations[
                _relation_key(
                    entry["relation"], entry["component_a"], entry["component_b"]
                )
            ] = RelationMemory(
                relation=entry["relation"],
                component_a=entry["component_a"],
                component_b=entry["component_b"],
                mean_geometry=payload[f"{key}_mean"].astype(np.float32),
                inverse_covariance=payload[f"{key}_inverse"].astype(np.float32),
                pair_prototype=pair,
                sample_count=int(entry["sample_count"]),
            )
        return SupportMemory(
            category=manifest["category"],
            grid_size=tuple(int(value) for value in manifest["grid_size"]),
            components=components,
            relations=relations,
            baseline_stats=dict(manifest.get("baseline_stats", {})),
            attribute_stats=dict(manifest.get("attribute_stats", {})),
            atom_stats=dict(manifest.get("atom_stats", {})),
            logic_stats=dict(manifest.get("logic_stats", {})),
            feature_dim=int(manifest.get("feature_dim", 0)),
            metadata=dict(manifest.get("metadata", {})),
        )
