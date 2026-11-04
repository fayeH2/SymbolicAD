"""Offline-only CLIP text seeding for compiling visual support prototypes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from symbolicAD.models.clip_backbone import CLIPBackbone
from symbolicAD.models.text_embeddings import encode_prompts_cached


def text_similarity_map(
    backbone: CLIPBackbone,
    patch_features: np.ndarray,
    prompts,
    cache_dir: Optional[str | Path] = None,
    enabled: bool = True,
) -> np.ndarray:
    """Seed normal visual prototypes from an offline semantic description."""
    embeddings = encode_prompts_cached(backbone, prompts, cache_dir, enabled=enabled)
    if embeddings.size == 0:
        batch, _, height, width = patch_features.shape
        return np.zeros((batch, height, width), dtype=np.float32)
    similarities = np.einsum("bdhw,td->bthw", patch_features, embeddings)
    return similarities.max(axis=1).astype(np.float32)
