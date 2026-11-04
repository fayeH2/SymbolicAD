"""Turn failed atoms into a logic anomaly heat-map."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from symbolicAD.config import Config
from symbolicAD.rules.executor import AtomicResult, LogicResult

def build_atom_heatmap(
    atom_result: AtomicResult,
) -> np.ndarray:
    if not atom_result.valid or atom_result.weight <= 0:
        return np.zeros_like(atom_result.responsibility_map, dtype=np.float32)
    responsibility = np.asarray(atom_result.responsibility_map, dtype=np.float32)
    return (
        float(atom_result.weight) * (1.0 - float(atom_result.truth)) * responsibility
    ).astype(np.float32)


def build_logic_heatmap(
    atom_results: Dict[str, AtomicResult],
    config: Config,
    logic_result: Optional[LogicResult] = None,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Fuse atom heat-maps as ``1 - prod(1 - H_i)``."""
    atom_heatmaps: Dict[str, np.ndarray] = {}
    grid_size = next(
        (result.responsibility_map.shape for result in atom_results.values()), None
    )
    if grid_size is None:
        raise ValueError("cannot build logic heatmap without atom results")
    logic_map = np.zeros(grid_size, dtype=np.float32)

    active_failed = None
    if logic_result is not None:
        active_failed = set(logic_result.active_failed_atom_ids)

    for atom_id, atom_result in atom_results.items():
        if active_failed is not None and atom_id not in active_failed:
            heatmap = np.zeros_like(
                atom_result.responsibility_map, dtype=np.float32
            )
        else:
            heatmap = build_atom_heatmap(atom_result)
        atom_heatmaps[atom_id] = heatmap
        logic_map = 1.0 - (1.0 - logic_map) * (1.0 - heatmap)

    sigma = float(getattr(config.logic, "heatmap_sigma", 0.0))
    if sigma > 0.0:
        logic_map = gaussian_filter(logic_map, sigma=sigma, mode="nearest")
    return np.clip(logic_map, 0.0, 1.0).astype(np.float32), atom_heatmaps
