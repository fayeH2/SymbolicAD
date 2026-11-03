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
