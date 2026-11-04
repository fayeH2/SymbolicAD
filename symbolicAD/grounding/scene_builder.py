"""Build a visual scene graph from frozen component prototype maps."""

from __future__ import annotations

from typing import Dict

import numpy as np

from symbolicAD.config import Config
from symbolicAD.rules.schema import RuleKB

from .component_maps import normalize_with_support
from .instance_extractor import (
    component_spatial_prior,
    extract_instances,
    image_to_lab_grid,
    select_instances,
)
from .scene_graph import SceneGraph
from .support_memory import SupportMemory


def build_scene_graph(
    image_index: int,
    rule_kb: RuleKB,
    component_maps: Dict[str, np.ndarray],
    support_memory: SupportMemory,
    patches: np.ndarray,
    config: Config,
    image=None,
) -> SceneGraph:
    """Ground frozen visual prototypes into component instances."""
    scene_graph = SceneGraph(component_maps={}, instances={}, patch_features=patches)
    color_grid = (
        image_to_lab_grid(image, scene_graph.grid_size()) if image is not None else None
    )
    for symbol in rule_kb.components:
        name = symbol.name
        raw = component_maps[name][image_index]
        if support_memory.has_component(name):
            memory = support_memory.components[name]
            normalized = normalize_with_support(
                raw, memory.score_q_low, memory.score_q_high
            )
        else:
            normalized = np.clip(raw, 0.0, 1.0)
        if image is not None:
            prior = component_spatial_prior(
                image,
                scene_graph.grid_size(),
                position=rule_kb.position_for(name),
                role=symbol.role,
            )
            normalized *= prior
            if "container" in symbol.role.lower():
                normalized = np.maximum(normalized, 0.6 * prior)
        scene_graph.component_maps[name] = normalized
        instances = extract_instances(
            component_map=normalized,
            patch_features=patches,
            concept=name,
            threshold=config.grounding.component_threshold,
            min_area=config.grounding.min_instance_area,
            morphology_kernel=config.grounding.morphology_kernel,
            color_grid=color_grid,
        )
        scene_graph.instances[name] = select_instances(
            instances, rule_kb.count_for(name)
        )
    return scene_graph
