"""Deterministic on-disk cache for CLIP text embeddings."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .clip_backbone import CLIPBackbone


def _cache_key(model_name: str, prompts: Sequence[str]) -> str:
    token = model_name + "|" + "||".join(prompts)
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]


def encode_prompts_cached(
    backbone: CLIPBackbone,
    prompts: Sequence[str],
    cache_dir: Optional[str | Path],
    enabled: bool = True,
) -> np.ndarray:
    prompts = list(prompts)
    if not prompts:
        return np.zeros((0, backbone.embed_dim), dtype=np.float32)
    if cache_dir is None or not enabled:
        return backbone.encode_text(prompts)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{_cache_key(backbone.model_name, prompts)}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as payload:
            cached_prompts = [str(item) for item in payload["prompts"].tolist()]
            if cached_prompts == prompts:
                return payload["embeddings"].astype(np.float32)

    embeddings = backbone.encode_text(prompts)
    np.savez_compressed(
        path,
        prompts=np.asarray(prompts, dtype=str),
        embeddings=embeddings.astype(np.float32),
    )
    return embeddings
