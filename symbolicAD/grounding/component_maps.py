"""CLIP text / prototype component response maps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from symbolicAD.models.clip_backbone import CLIPBackbone
from symbolicAD.models.text_embeddings import encode_prompts_cached
from symbolicAD.rules.schema import RuleKB


def text_similarity_map(
    backbone: CLIPBackbone,
    patch_features: np.ndarray,
    prompts,
    cache_dir: Optional[str | Path] = None,
    enabled: bool = True,
) -> np.ndarray:
    """Max cosine similarity between every patch and a prompt set."""
    embeddings = encode_prompts_cached(backbone, prompts, cache_dir, enabled=enabled)
    if embeddings.size == 0:
        batch, _, height, width = patch_features.shape
        return np.zeros((batch, height, width), dtype=np.float32)
    similarities = np.einsum("bdhw,td->bthw", patch_features, embeddings)
    return similarities.max(axis=1).astype(np.float32)


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
    backbone: CLIPBackbone,
    patch_features: np.ndarray,
    rule_kb: RuleKB,
    support_memory=None,
    alpha: float = 0.5,
    cache_dir: Optional[str | Path] = None,
    text_cache_enabled: bool = True,
) -> Dict[str, np.ndarray]:
    """Return raw component response maps ``{component: [B, h, w]}``."""
    maps: Dict[str, np.ndarray] = {}
    for symbol in rule_kb.components:
        prompts = rule_kb.prompts_for(symbol.name)
        text_map = text_similarity_map(
            backbone,
            patch_features,
            prompts,
            cache_dir=cache_dir,
            enabled=text_cache_enabled,
        )
        if support_memory is not None and support_memory.has_component(symbol.name):
            memory = support_memory.components[symbol.name]
            proto_map = prototype_similarity_map(patch_features, memory.prototypes)
            fused = alpha * text_map + (1.0 - alpha) * proto_map
        else:
            fused = text_map
        maps[symbol.name] = fused.astype(np.float32)
    return maps


def normalize_with_support(
    scores: np.ndarray,
    q_low: float,
    q_high: float,
) -> np.ndarray:
    """Support-calibrated normalisation, never per-query min-max."""
    denominator = max(float(q_high) - float(q_low), 1e-6)
    return np.clip((scores - float(q_low)) / denominator, 0.0, 1.0).astype(np.float32)
