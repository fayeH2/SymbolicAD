"""Relation-conditioned CLIP rescoring inside responsibility regions."""

from __future__ import annotations

from typing import Optional

import numpy as np

from symbolicAD.config import Config
from symbolicAD.models.clip_backbone import CLIPBackbone
from symbolicAD.models.text_embeddings import encode_prompts_cached
from symbolicAD.rules.executor import AtomicResult
from symbolicAD.rules.schema import RuleKB


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-value))


def rescore_responsibility(
    atom_result: AtomicResult,
    rule_kb: RuleKB,
    patch_features: np.ndarray,
    backbone: Optional[CLIPBackbone],
    config: Config,
) -> np.ndarray:
    """Modulate a responsibility map with violation-vs-normal CLIP similarity."""
    responsibility = np.asarray(atom_result.responsibility_map, dtype=np.float32)
    if responsibility.size == 0 or responsibility.max() <= 0:
        return responsibility
    if atom_result.evidence.get("missing"):
        return responsibility
    if backbone is None:
        return responsibility

    prompts = rule_kb.atom_prompts(atom_result.atom_id)
    normal_prompt, violation_prompt = prompts.get("normal", ""), prompts.get("violation", "")
    if not normal_prompt or not violation_prompt:
        return responsibility

    embeddings = encode_prompts_cached(
        backbone,
        [normal_prompt, violation_prompt],
        cache_dir=config.text_embedding_dir,
        enabled=config.grounding.text_embedding_cache,
    )
    normal_embedding, violation_embedding = embeddings[0], embeddings[1]
    similarity_normal = np.einsum("dhw,d->hw", patch_features, normal_embedding)
    similarity_violation = np.einsum("dhw,d->hw", patch_features, violation_embedding)
    temperature = max(float(config.logic.rescore_temperature), 1e-6)
    probabilities = _sigmoid((similarity_violation - similarity_normal) / temperature)
    eta = float(config.logic.eta)
    return (responsibility * (eta * probabilities + (1.0 - eta))).astype(np.float32)
