"""Uniform baseline interface plus a CLIP zero-shot baseline.

Stage one wraps a CLIP zero-shot branch so the full pipeline can run without a
trained checkpoint.  The adapter can later be swapped for WinCLIP, AnomalyCLIP
or INP-Former without touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .clip_backbone import CLIPBackbone
from .text_embeddings import encode_prompts_cached


def category_phrase(category: str) -> str:
    return category.replace("_", " ")


def default_prompts(category: str) -> tuple[List[str], List[str]]:
    phrase = category_phrase(category)
    normal = [
        f"a photo of a normal {phrase}",
        f"a photo of a clean {phrase} without defects",
        f"a perfect {phrase}",
        f"a {phrase} in good condition",
    ]
    anomaly = [
        f"a photo of a damaged {phrase}",
        f"a photo of a defective {phrase}",
        f"a {phrase} with a scratch",
        f"a {phrase} with a dent",
        f"a {phrase} with contamination",
        f"a broken {phrase}",
        f"a {phrase} with missing parts",
        f"an anomalous {phrase}",
    ]
    return normal, anomaly


@dataclass
class BaselineOutput:
    anomaly_map: np.ndarray
    image_score: np.ndarray
    patch_features: np.ndarray
    global_features: np.ndarray
    prompt_similarity: Optional[Dict[str, np.ndarray]] = None


def topk_mean(maps: np.ndarray, percent: float = 1.0) -> np.ndarray:
    maps = np.asarray(maps, dtype=np.float64)
    flat = maps.reshape(maps.shape[0], -1)
    count = max(1, int(round(flat.shape[1] * float(percent) / 100.0)))
    count = min(count, flat.shape[1])
    partitioned = np.partition(flat, flat.shape[1] - count, axis=1)
    return partitioned[:, -count:].mean(axis=1).astype(np.float32)


class CLIPZeroShotBaseline:
    def __init__(
        self,
        backbone: CLIPBackbone,
        category: str,
        cache_dir: Optional[str] = None,
        normal_prompts: Optional[Sequence[str]] = None,
        anomaly_prompts: Optional[Sequence[str]] = None,
        topk_percent: float = 1.0,
    ):
        self.backbone = backbone
        self.category = category
        self.topk_percent = float(topk_percent)
        normal, anomaly = default_prompts(category)
        self.normal_prompts = list(normal_prompts) if normal_prompts else normal
        self.anomaly_prompts = list(anomaly_prompts) if anomaly_prompts else anomaly
        self.normal_embeddings = encode_prompts_cached(
            backbone, self.normal_prompts, cache_dir
        )
        self.anomaly_embeddings = encode_prompts_cached(
            backbone, self.anomaly_prompts, cache_dir
        )

    def score_from_features(
        self,
        patch_features: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute anomaly maps from pre-computed normalised patch features."""
        patches = np.asarray(patch_features, dtype=np.float32)
        # [B, D, h, w] x [N, D] -> [B, N, h, w]
        sim_normal = np.einsum("bdhw,nd->bnhw", patches, self.normal_embeddings)
        sim_anomaly = np.einsum("bdhw,md->bmhw", patches, self.anomaly_embeddings)
        sim_normal = sim_normal.max(axis=1)
        sim_anomaly = sim_anomaly.max(axis=1)
        scale = self.backbone.logit_scale
        logits = np.stack(
            [sim_normal * scale, sim_anomaly * scale], axis=1
        ).astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        anomaly_map = probabilities[:, 1].astype(np.float32)
        image_score = topk_mean(anomaly_map, self.topk_percent)
        return {
            "anomaly_map": anomaly_map,
            "image_score": image_score,
            "normal_similarity": sim_normal.astype(np.float32),
            "anomaly_similarity": sim_anomaly.astype(np.float32),
        }

    def forward(self, images: Sequence) -> BaselineOutput:
        encoded = self.backbone.encode_images(images)
        scored = self.score_from_features(encoded["patches"])
        return BaselineOutput(
            anomaly_map=scored["anomaly_map"],
            image_score=scored["image_score"],
            patch_features=encoded["patches"],
            global_features=encoded["global"],
            prompt_similarity={
                "normal": scored["normal_similarity"],
                "anomaly": scored["anomaly_similarity"],
            },
        )

    def support_statistics(
        self,
        support_images: Sequence,
        low_quantile: float = 0.5,
        high_quantile: float = 0.99,
    ) -> Dict[str, float]:
        output = self.forward(support_images)
        flat = output.anomaly_map.reshape(-1)
        return {
            "q_low": float(np.quantile(flat, low_quantile)),
            "q_high": float(np.quantile(flat, high_quantile)),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
        }


class CLIPFewShotBaseline:
    """Few-shot CLIP baseline: patch distance to normal support patches.

    The baseline deliberately lives in the same CLIP patch space as the
    component-grounding and rule branches, so the semantic rules and the visual
    anomaly map can be fused directly.
    """

    def __init__(
        self,
        backbone: CLIPBackbone,
        support_images: Sequence,
        topk_percent: float = 1.0,
        knn: int = 1,
        max_bank_size: int = 0,
    ):
        if not support_images:
            raise ValueError("few-shot baseline requires at least one support image")
        self.backbone = backbone
        self.topk_percent = float(topk_percent)
        self.knn = max(1, int(knn))
        encoded = backbone.encode_images(support_images)
        self.support_patches = np.asarray(encoded["patches"], dtype=np.float32)
        self.grid_size = (
            int(self.support_patches.shape[2]),
            int(self.support_patches.shape[3]),
        )
        bank = self.support_patches.transpose(0, 2, 3, 1).reshape(
            -1, self.support_patches.shape[1]
        )
        bank = bank / np.maximum(
            np.linalg.norm(bank, axis=1, keepdims=True), 1e-8
        )
        if max_bank_size and bank.shape[0] > int(max_bank_size):
            rng = np.random.default_rng(0)
            indices = rng.choice(bank.shape[0], size=int(max_bank_size), replace=False)
            bank = bank[indices]
        self.bank = bank.astype(np.float32)
        self._stats_cache: Dict[tuple, Dict[str, float]] = {}

    def _raw_maps(self, patches: np.ndarray, bank: np.ndarray) -> np.ndarray:
        patches = np.asarray(patches, dtype=np.float32)
        batch, channels, height, width = patches.shape
        flat = patches.reshape(batch, channels, -1).transpose(0, 2, 1)
        similarities = flat @ bank.T  # [B, P, N]
        if self.knn <= 1:
            best = similarities.max(axis=2)
        else:
            k = min(self.knn, similarities.shape[2])
            best = np.partition(similarities, -k, axis=2)[:, :, -k:].mean(axis=2)
        return (1.0 - best).reshape(batch, height, width).astype(np.float32)

    def score_from_features(self, patch_features: np.ndarray) -> Dict[str, np.ndarray]:
        anomaly_map = self._raw_maps(patch_features, self.bank)
        image_score = topk_mean(anomaly_map, self.topk_percent)
        return {
            "anomaly_map": anomaly_map,
            "image_score": image_score,
        }

    def forward(self, images: Sequence) -> BaselineOutput:
        encoded = self.backbone.encode_images(images)
        scored = self.score_from_features(encoded["patches"])
        return BaselineOutput(
            anomaly_map=scored["anomaly_map"],
            image_score=scored["image_score"],
            patch_features=encoded["patches"],
            global_features=encoded["global"],
            prompt_similarity=None,
        )

    def _leave_one_out_maps(self) -> np.ndarray:
        count = self.support_patches.shape[0]
        if count < 2:
            raise ValueError(
                "few-shot baseline needs at least 2 support images for leave-one-out calibration"
            )
        maps = []
        for index in range(count):
            keep = [item for item in range(count) if item != index]
            bank = self.support_patches[keep].transpose(0, 2, 3, 1).reshape(
                -1, self.support_patches.shape[1]
            )
            bank = bank / np.maximum(
                np.linalg.norm(bank, axis=1, keepdims=True), 1e-8
            )
            maps.append(
                self._raw_maps(self.support_patches[index:index + 1], bank)[0]
            )
        return np.stack(maps, axis=0)

    def support_statistics(
        self,
        support_images: Optional[Sequence] = None,
        low_quantile: float = 0.5,
        high_quantile: float = 0.99,
    ) -> Dict[str, float]:
        key = (float(low_quantile), float(high_quantile))
        if key not in self._stats_cache:
            flat = self._leave_one_out_maps().reshape(-1)
            self._stats_cache[key] = {
                "q_low": float(np.quantile(flat, low_quantile)),
                "q_high": float(np.quantile(flat, high_quantile)),
                "mean": float(flat.mean()),
                "std": float(flat.std()),
            }
        return dict(self._stats_cache[key])


def build_baseline(backbone: CLIPBackbone, support_images: Sequence, category: str, config):
    """Build the configured baseline in the shared CLIP patch space."""
    from symbolicAD.config import Config

    if not isinstance(config, Config):
        raise TypeError("config must be a symbolicAD.config.Config")
    if config.model.baseline == "clip_fewshot":
        return CLIPFewShotBaseline(
            backbone,
            support_images,
            topk_percent=config.fusion.topk_percent,
            knn=config.model.knn,
            max_bank_size=config.model.max_bank_size,
        )
    return CLIPZeroShotBaseline(
        backbone,
        category=category,
        cache_dir=config.text_embedding_dir,
        topk_percent=config.fusion.topk_percent,
    )
