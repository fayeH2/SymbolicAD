"""Turn failed atoms into a logic anomaly heat-map."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from symbolicAD.config import Config
from symbolicAD.models.clip_backbone import CLIPBackbone
from symbolicAD.rules.executor import AtomicResult, LogicResult
from symbolicAD.rules.schema import RuleKB

from .semantic_rescore import rescore_responsibility


def build_atom_heatmap(
    atom_result: AtomicResult,
    rule_kb: RuleKB,
    patch_features: np.ndarray,
    backbone: Optional[CLIPBackbone],
    config: Config,
) -> np.ndarray:
    if not atom_result.valid or atom_result.weight <= 0:
        return np.zeros_like(atom_result.responsibility_map, dtype=np.float32)
    if config.logic.rescore_enabled:
        responsibility = rescore_responsibility(
            atom_result, rule_kb, patch_features, backbone, config
        )
    else:
        responsibility = np.asarray(
            atom_result.responsibility_map, dtype=np.float32
        )
    return (
        float(atom_result.weight) * (1.0 - float(atom_result.truth)) * responsibility
    ).astype(np.float32)


def build_logic_heatmap(
    atom_results: Dict[str, AtomicResult],
    rule_kb: RuleKB,
    patch_features: np.ndarray,
    backbone: Optional[CLIPBackbone],
    config: Config,
    logic_result: Optional[LogicResult] = None,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Fuse atom heat-maps as ``1 - prod(1 - H_i)``."""
    atom_heatmaps: Dict[str, np.ndarray] = {}
    grid_size = next(
        (result.responsibility_map.shape for result in atom_results.values()), None
    )
    if grid_size is None:
        height, width = patch_features.shape[1:]
        grid_size = (height, width)
    logic_map = np.zeros(grid_size, dtype=np.float32)

    for atom_id, atom_result in atom_results.items():
        heatmap = build_atom_heatmap(
            atom_result, rule_kb, patch_features, backbone, config
        )
        atom_heatmaps[atom_id] = heatmap
        logic_map = 1.0 - (1.0 - logic_map) * (1.0 - heatmap)

    return np.clip(logic_map, 0.0, 1.0).astype(np.float32), atom_heatmaps
