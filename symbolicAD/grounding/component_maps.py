"""Runtime component maps from frozen normal-support visual prototypes."""

from __future__ import annotations

from typing import Dict

import numpy as np

from symbolicAD.rules.schema import RuleKB


def prototype_similarity_map(
    patch_features: np.ndarray,
    prototypes: np.ndarray,
) -> np.ndarray:
    if prototypes is None or prototypes.size == 0:
        batch, _, height, width = patch_features.shape
        return np.zeros((batch, height, width), dtype=np.float32)
    similarities = np.einsum("bdhw,nd->bnhw", patch_features, prototypes)
    return similarities.max(axis=1).astype(np.float32)


def build_component_maps(
    patch_features: np.ndarray,
    rule_kb: RuleKB,
    support_memory,
) -> Dict[str, np.ndarray]:
    """Ground static semantic components using frozen visual prototypes only.

    Qwen prompts and CLIP text embeddings may seed prototypes during offline
    compilation, but deployment inference consumes only query CLIP visual
    patches and those frozen normal-support prototypes.
    """
    maps: Dict[str, np.ndarray] = {}
    for symbol in rule_kb.components:
        if not support_memory.has_component(symbol.name):
            raise RuntimeError(
                f"static support program has no prototype for {symbol.name}"
            )
        memory = support_memory.components[symbol.name]
        maps[symbol.name] = prototype_similarity_map(
            patch_features, memory.prototypes
        ).astype(np.float32)
    return maps


def normalize_with_support(
    scores: np.ndarray,
    q_low: float,
    q_high: float,
) -> np.ndarray:
    """Support-calibrated normalisation, never per-query min-max."""
    denominator = max(float(q_high) - float(q_low), 1e-6)
    return np.clip((scores - float(q_low)) / denominator, 0.0, 1.0).astype(np.float32)
