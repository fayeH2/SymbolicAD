"""Support-only calibration used by the static-program compiler."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from symbolicAD.config import Config
from symbolicAD.localization.responsibility import build_logic_heatmap
from symbolicAD.models.clip_backbone import CLIPBackbone
from symbolicAD.rules.executor import execute_formula
from symbolicAD.rules.schema import RuleKB

from .component_maps import build_component_maps
from .relation_grounder import calibrate_atom_results, evaluate_atoms
from .scene_builder import build_scene_graph
from .support_memory import (
    LOGIC_CALIBRATION_VERSION,
    SupportMemory,
    build_support_memory,
)


def calibrate_logic_on_support(
    backbone: CLIPBackbone,
    support_images: Sequence,
    support_memory: SupportMemory,
    rule_kb: RuleKB,
    config: Config,
) -> SupportMemory:
    """Fit atom truth/noise statistics with leave-one-out normal support."""
    collected: Dict[str, list] = {}
    held_records: list[Dict[str, object]] = []
    count = len(support_images)
    for held in range(count):
        keep = [index for index in range(count) if index != held] or [held]
        calibration_memory = build_support_memory(
            backbone,
            [support_images[index] for index in keep],
            rule_kb,
            config,
            baseline=None,
        )
        patches = backbone.encode_images([support_images[held]])["patches"]
        component_maps = build_component_maps(patches, rule_kb, calibration_memory)
        scene_graph = build_scene_graph(
            0,
            rule_kb,
            component_maps,
            calibration_memory,
            patches[0],
            config,
            image=support_images[held],
        )
        atom_results = evaluate_atoms(scene_graph, rule_kb, calibration_memory, config)
        for atom_id, result in atom_results.items():
            if result.valid and result.weight > 0:
                collected.setdefault(atom_id, []).append(float(result.truth))
        held_records.append({"atom_results": atom_results})

    atom_stats: Dict[str, Dict[str, float]] = {}
    for atom_id, values in collected.items():
        array = np.asarray(values, dtype=np.float64)
        atom_stats[atom_id] = {
            "q05": float(np.quantile(array, 0.05)),
            "q50": float(np.quantile(array, 0.50)),
            "q95": float(np.quantile(array, 0.95)),
            "mean": float(array.mean()),
            "std": float(array.std()),
            "n": int(array.size),
            "support_failure_rate": float(
                np.mean(array < float(config.logic.fail_threshold))
            ),
        }
    support_memory.atom_stats = atom_stats

    logic_maps = []
    for record in held_records:
        atom_results = record["atom_results"]
        for atom_id, result in atom_results.items():
            if float(atom_stats.get(atom_id, {}).get("support_failure_rate", 0.0)) > 0.5:
                result.weight = 0.0
                result.valid = False
        atom_results = calibrate_atom_results(atom_results, support_memory, config)
        logic_result = execute_formula(
            rule_kb.formula,
            atom_results,
            fail_threshold=config.logic.fail_threshold,
            epsilon=config.logic.and_epsilon,
        )
        logic_map, _ = build_logic_heatmap(
            atom_results, config, logic_result=logic_result
        )
        logic_maps.append(logic_map)
    if logic_maps:
        flat = np.concatenate([item.reshape(-1) for item in logic_maps])
        support_memory.logic_stats = {
            "q_low": float(np.quantile(flat, 0.50)),
            "q_high": float(max(np.quantile(flat, 0.99), 0.02)),
        }
    support_memory.metadata["logic_calibration_version"] = LOGIC_CALIBRATION_VERSION
    return support_memory
