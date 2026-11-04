"""Visual CLIP feature-memory baselines for few-shot anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter

from .clip_backbone import CLIPBackbone


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
        spatial_radius: int = 1,
        spatial_weight: float = 0.5,
    ):
        if not support_images:
            raise ValueError("few-shot baseline requires at least one support image")
        self.backbone = backbone
        self.topk_percent = float(topk_percent)
        self.knn = max(1, int(knn))
        self.spatial_radius = max(0, int(spatial_radius))
        self.spatial_weight = float(np.clip(spatial_weight, 0.0, 1.0))
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

    def _global_maps(self, patches: np.ndarray, bank: np.ndarray) -> np.ndarray:
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

    def _spatial_maps(
        self,
        patches: np.ndarray,
        support_patches: np.ndarray,
    ) -> np.ndarray:
        """Patch distance to support features in the same local coordinates."""
        query = np.asarray(patches, dtype=np.float32)
        support = np.asarray(support_patches, dtype=np.float32)
        batch, _, height, width = query.shape
        if support.ndim != 4 or support.shape[2:] != (height, width):
            raise ValueError(
                f"support grid {support.shape} is incompatible with query {query.shape}"
            )
        candidates = []
        radius = self.spatial_radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                similarity = np.full(
                    (batch, support.shape[0], height, width),
                    -np.inf,
                    dtype=np.float32,
                )
                query_y = slice(max(0, -dy), min(height, height - dy))
                query_x = slice(max(0, -dx), min(width, width - dx))
                support_y = slice(max(0, dy), min(height, height + dy))
                support_x = slice(max(0, dx), min(width, width + dx))
                similarity[:, :, query_y, query_x] = np.einsum(
                    "bdhw,kdhw->bkhw",
                    query[:, :, query_y, query_x],
                    support[:, :, support_y, support_x],
                )
                candidates.append(similarity)
        similarities = np.concatenate(candidates, axis=1)
        if self.knn <= 1:
            best = similarities.max(axis=1)
        else:
            k = min(self.knn, support.shape[0])
            best = np.partition(similarities, -k, axis=1)[:, -k:].mean(axis=1)
        return (1.0 - best).astype(np.float32)

    def _raw_maps(
        self,
        patches: np.ndarray,
        bank: np.ndarray,
        spatial_support: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        global_maps = self._global_maps(patches, bank)
        weight = float(getattr(self, "spatial_weight", 0.0))
        if weight <= 0.0 or spatial_support is None or not len(spatial_support):
            return global_maps
        spatial_maps = self._spatial_maps(patches, spatial_support)
        return (
            (1.0 - weight) * global_maps + weight * spatial_maps
        ).astype(np.float32)

    def score_from_features(self, patch_features: np.ndarray) -> Dict[str, np.ndarray]:
        anomaly_map = self._raw_maps(
            patch_features, self.bank, spatial_support=self.support_patches
        )
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
        if count == 1:
            # 1-shot FSAD has no second normal image.  Calibrate each support
            # patch against the other spatial patches instead of self-matching
            # every patch to distance zero.
            patches = self.support_patches[0]
            channels, height, width = patches.shape
            bank = patches.reshape(channels, -1).T
            bank = bank / np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-8)
            similarities = bank @ bank.T
            if similarities.shape[0] <= 1:
                return np.zeros((1, height, width), dtype=np.float32)
            np.fill_diagonal(similarities, -np.inf)
            k = min(self.knn, similarities.shape[1] - 1)
            if k <= 1:
                best = similarities.max(axis=1)
            else:
                best = np.partition(similarities, -k, axis=1)[:, -k:].mean(axis=1)
            return (1.0 - best).reshape(1, height, width).astype(np.float32)
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
                self._raw_maps(
                    self.support_patches[index:index + 1],
                    bank,
                    spatial_support=self.support_patches[keep],
                )[0]
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
            leave_one_out = self._leave_one_out_maps()
            flat = leave_one_out.reshape(-1)
            image_scores = topk_mean(leave_one_out, self.topk_percent)
            self._stats_cache[key] = {
                "q_low": float(np.quantile(flat, low_quantile)),
                "q_high": float(np.quantile(flat, high_quantile)),
                "mean": float(flat.mean()),
                "std": float(flat.std()),
                "image_q_low": float(np.quantile(image_scores, low_quantile)),
                "image_q_high": float(np.quantile(image_scores, high_quantile)),
            }
        return dict(self._stats_cache[key])


def _local_smooth(features: np.ndarray, radius: int, weight: float) -> np.ndarray:
    radius = int(radius)
    weight = float(np.clip(weight, 0.0, 1.0))
    if radius <= 0 or weight <= 0.0:
        return np.asarray(features, dtype=np.float32)
    smoothed = uniform_filter(
        np.asarray(features, dtype=np.float32),
        size=(1, 1, 2 * radius + 1, 2 * radius + 1),
        mode="nearest",
    )
    return ((1.0 - weight) * features + weight * smoothed).astype(np.float32)


def _fit_whitening(
    bank: np.ndarray,
    max_components: int,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bank = np.asarray(bank, dtype=np.float32)
    mean = bank.mean(axis=0)
    centered = bank - mean
    # SVD is stable for the 1500 x 768 support bank and gives the same
    # whitening as an eigendecomposition of the covariance.
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    limit = int(max_components) if int(max_components) > 0 else vt.shape[0]
    limit = max(1, min(limit, vt.shape[0], vt.shape[1]))
    components = vt[:limit].T.astype(np.float32)
    scale = singular[:limit] / np.sqrt(max(bank.shape[0] - 1, 1))
    scale = np.maximum(scale, np.sqrt(max(float(epsilon), 1e-12)))
    return mean.astype(np.float32), components, (1.0 / scale).astype(np.float32)


def _apply_whitening(
    features: np.ndarray,
    whitening: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    channel_axis: int = -1,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if whitening is None:
        return features
    mean, components, inverse_scale = whitening
    moved = np.moveaxis(features, channel_axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    flat = (flat - mean) @ components * inverse_scale
    moved = flat.reshape(*moved.shape[:-1], flat.shape[-1])
    return np.moveaxis(moved, -1, channel_axis).astype(np.float32)


def _normalise_rows(features: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(features, axis=axis, keepdims=True)
    return (features / np.maximum(norm, 1e-8)).astype(np.float32)


class CLIPAdaptiveFewShotBaseline:
    """Pure visual CLIP-feature few-shot baseline.

    Three deliberate differences from the initial 1-NN baseline:

    * dense features are read from several ViT blocks instead of only the final
      projected layer (middle layers preserve local structure);
    * every layer is whitened on the normal support bank before cosine
      matching, which removes CLIP's dominant common directions;
    * a generic foreground prior suppresses background responses.

    Everything is estimated from the normal support images only; there are no
    category-specific constants or query labels.
    """

    def __init__(
        self,
        backbone: CLIPBackbone,
        support_images: Sequence,
        topk_percent: float = 1.0,
        knn: int = 5,
        max_bank_size: int = 0,
        spatial_radius: int = 1,
        spatial_weight: float = 0.5,
        local_radius: int = 1,
        local_weight: float = 0.35,
        whiten: bool = True,
        whiten_components: int = 128,
        whiten_eps: float = 1e-3,
        foreground_masking: bool = True,
        foreground_soft: float = 0.35,
        feature_source: str = "both",
        normalise_layers: bool = False,
        projected_weight: float = 0.25,
    ):
        if not support_images:
            raise ValueError("few-shot baseline requires at least one support image")
        self.backbone = backbone
        self.topk_percent = float(topk_percent)
        self.knn = max(1, int(knn))
        self.max_bank_size = int(max_bank_size)
        self.spatial_radius = max(0, int(spatial_radius))
        self.spatial_weight = float(np.clip(spatial_weight, 0.0, 1.0))
        self.local_radius = max(0, int(local_radius))
        self.local_weight = float(np.clip(local_weight, 0.0, 1.0))
        self.whiten = bool(whiten)
        self.whiten_components = int(whiten_components)
        self.whiten_eps = float(whiten_eps)
        self.foreground_masking = bool(foreground_masking)
        self.foreground_soft = float(np.clip(foreground_soft, 0.0, 1.0))
        self.feature_source = str(feature_source).lower()
        self.normalise_layers = bool(normalise_layers)
        self.projected_weight = float(np.clip(projected_weight, 0.0, 1.0))
        self._stats_cache: Dict[tuple, Dict[str, float]] = {}

        encoded = backbone.encode_images(support_images)
        self.support_patches = np.asarray(encoded["patches"], dtype=np.float32)
        self.grid_size = (
            int(self.support_patches.shape[2]),
            int(self.support_patches.shape[3]),
        )
        self.support_foreground = self._support_foreground(support_images)
        native_layers = list(encoded.get("patch_layers", []))
        if self.feature_source == "projected":
            layer_features = [self.support_patches]
            self.layer_kinds = ["projected"]
            self.layer_weights = [1.0]
        elif self.feature_source == "both":
            if native_layers:
                layer_features = native_layers + [self.support_patches]
                self.layer_kinds = ["native"] * len(native_layers) + ["projected"]
                native_weight = (1.0 - self.projected_weight) / len(native_layers)
                self.layer_weights = [native_weight] * len(native_layers) + [
                    self.projected_weight
                ]
            else:
                layer_features = [self.support_patches]
                self.layer_kinds = ["projected"]
                self.layer_weights = [1.0]
        else:
            layer_features = native_layers
            self.layer_kinds = ["native"] * len(native_layers)
            self.layer_weights = [1.0 / max(len(native_layers), 1)] * len(native_layers)
        self.layers = self._build_layers(layer_features or [self.support_patches])
        if len(self.layer_weights) != len(self.layers):
            self.layer_weights = [1.0 / max(len(self.layers), 1)] * len(self.layers)
            self.layer_kinds = ["native"] * len(self.layers)
        self._calibrate_from_support()

    # ------------------------------------------------------------------ setup
    def _support_foreground(self, support_images: Sequence) -> np.ndarray:
        if not self.foreground_masking or not support_images:
            return np.ones(self.grid_size, dtype=np.float32)
        masks = [
            self._foreground_grid(image, self.grid_size) for image in support_images
        ]
        return np.maximum.reduce(masks).astype(np.float32)

    @staticmethod
    def _foreground_grid(image, grid_size: Tuple[int, int]) -> np.ndarray:
        """Generic border-background estimate, not a black-background constant."""
        import cv2

        height, width = int(grid_size[0]), int(grid_size[1])
        if hasattr(image, "convert"):
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        else:
            rgb = np.asarray(image, dtype=np.uint8)[..., :3]
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        border = np.concatenate(
            [
                lab[0],
                lab[-1],
                lab[:, 0],
                lab[:, -1],
            ],
            axis=0,
        )
        background = np.median(border, axis=0)
        distance = np.linalg.norm(lab - background[None, None, :], axis=-1)
        lightness_gap = np.abs(lab[..., 0] - background[0])
        mask = ((distance > 0.055) | (lightness_gap > 0.06)).astype(np.uint8)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
        )
        # Remove isolated noise components while keeping small genuine parts.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        clean = np.zeros_like(mask)
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) >= 2:
                clean[labels == index] = 1
        if float(clean.mean()) < 0.02:
            return np.ones((height, width), dtype=np.float32)
        return clean.astype(np.float32)

    def _prior(self, image, grid_size: Tuple[int, int]) -> np.ndarray:
        if not self.foreground_masking:
            return np.ones(grid_size, dtype=np.float32)
        query = self._foreground_grid(image, grid_size)
        foreground = np.maximum(query, self.support_foreground)
        return (
            self.foreground_soft
            + (1.0 - self.foreground_soft) * foreground
        ).astype(np.float32)

    def _build_layers(self, patch_layers: Sequence[np.ndarray]) -> List[Dict[str, object]]:
        layers: List[Dict[str, object]] = []
        for raw in patch_layers:
            support = _local_smooth(
                np.asarray(raw, dtype=np.float32),
                self.local_radius,
                self.local_weight,
            )
            channels = support.shape[1]
            flat = support.transpose(0, 2, 3, 1).reshape(-1, channels)
            whitening = (
                _fit_whitening(flat, self.whiten_components, self.whiten_eps)
                if self.whiten
                else None
            )
            transformed = _normalise_rows(
                _apply_whitening(support, whitening, channel_axis=1),
                axis=1,
            )
            bank = transformed.transpose(0, 2, 3, 1).reshape(-1, transformed.shape[1])
            if self.max_bank_size and bank.shape[0] > self.max_bank_size:
                rng = np.random.default_rng(0)
                keep = rng.choice(
                    bank.shape[0], size=self.max_bank_size, replace=False
                )
                bank = bank[keep]
            layers.append(
                {
                    "support": transformed.astype(np.float32),
                    "bank": bank.astype(np.float32),
                    "transform": whitening,
                }
            )
        return layers

    # -------------------------------------------------------------- scoring
    def _transform_query(
        self, features: np.ndarray, layer: Dict[str, object]
    ) -> np.ndarray:
        features = _local_smooth(
            np.asarray(features, dtype=np.float32),
            self.local_radius,
            self.local_weight,
        )
        transform = layer.get("transform")
        return _normalise_rows(
            _apply_whitening(features, transform, channel_axis=1),
            axis=1,
        )

    def _global_maps(self, query: np.ndarray, bank: np.ndarray) -> np.ndarray:
        batch, channels, height, width = query.shape
        flat = query.reshape(batch, channels, -1).transpose(0, 2, 1)
        similarities = flat @ bank.T
        k = min(self.knn, similarities.shape[2])
        if k <= 1:
            best = similarities.max(axis=2)
        else:
            best = np.partition(similarities, -k, axis=2)[:, :, -k:].mean(axis=2)
        return (1.0 - best).reshape(batch, height, width).astype(np.float32)

    def _spatial_maps(
        self, query: np.ndarray, support: np.ndarray
    ) -> np.ndarray:
        batch, _, height, width = query.shape
        candidates = []
        radius = self.spatial_radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                similarity = np.full(
                    (batch, support.shape[0], height, width),
                    -np.inf,
                    dtype=np.float32,
                )
                query_y = slice(max(0, -dy), min(height, height - dy))
                query_x = slice(max(0, -dx), min(width, width - dx))
                support_y = slice(max(0, dy), min(height, height + dy))
                support_x = slice(max(0, dx), min(width, width + dx))
                similarity[:, :, query_y, query_x] = np.einsum(
                    "bdhw,kdhw->bkhw",
                    query[:, :, query_y, query_x],
                    support[:, :, support_y, support_x],
                )
                candidates.append(similarity)
        similarities = np.concatenate(candidates, axis=1)
        k = min(self.knn, similarities.shape[1])
        if k <= 1:
            best = similarities.max(axis=1)
        else:
            best = np.partition(similarities, -k, axis=1)[:, -k:].mean(axis=1)
        return (1.0 - best).astype(np.float32)

    def _raw_visual_maps(
        self, query: np.ndarray, layer: Dict[str, object]
    ) -> np.ndarray:
        global_maps = self._global_maps(query, layer["bank"])  # type: ignore[index]
        if self.spatial_weight <= 0.0:
            return global_maps
        spatial = self._spatial_maps(query, layer["support"])  # type: ignore[index]
        return (
            (1.0 - self.spatial_weight) * global_maps
            + self.spatial_weight * spatial
        ).astype(np.float32)

    def _calibrate_from_support(self) -> None:
        loo_maps = np.zeros(
            (self.support_patches.shape[0],) + self.grid_size, dtype=np.float32
        )
        unmasked_loo = np.zeros_like(loo_maps)
        count = self.support_patches.shape[0]
        for layer_index, layer in enumerate(self.layers):
            support = layer["support"]  # type: ignore[index]
            per_image = []
            for held in range(count):
                if count > 1:
                    keep = [item for item in range(count) if item != held]
                    query = support[held:held + 1]
                    bank = np.concatenate(
                        [support[item].reshape(support.shape[1], -1).T for item in keep],
                        axis=0,
                    )
                    spatial_support = support[keep]
                else:
                    query = support[held:held + 1]
                    bank = support[held].reshape(support.shape[1], -1).T
                    spatial_support = support[held:held + 1]
                global_maps = self._global_maps(query, bank)
                if self.spatial_weight > 0.0:
                    spatial = self._spatial_maps(query, spatial_support)
                    raw = (
                        (1.0 - self.spatial_weight) * global_maps
                        + self.spatial_weight * spatial
                    )
                else:
                    raw = global_maps
                per_image.append(raw[0])
            stack = np.stack(per_image, axis=0)
            low = float(np.quantile(stack, 0.50))
            high = float(np.quantile(stack, 0.99))
            layer["q_low"] = low
            layer["q_high"] = high
            normalised = np.clip(
                (stack - low) / max(high - low, 1e-6), 0.0, 1.0
            )
            weight = float(self.layer_weights[layer_index])
            unmasked_loo += normalised * weight
            normalised *= self.support_foreground[None, ...]
            loo_maps += normalised * weight
        self.visual_stats = {
            "q_low": float(np.quantile(loo_maps, 0.50)),
            "q_high": float(np.quantile(loo_maps, 0.99)),
        }
        fused = loo_maps
        fused *= self.support_foreground[None, ...]
        self.fused_stats = {
            "q_low": float(np.quantile(fused, 0.50)),
            "q_high": float(np.quantile(fused, 0.99)),
        }
        foreground_binary = self.support_foreground > 0.5
        if foreground_binary.any() and (~foreground_binary).any():
            contrast = float(
                unmasked_loo[:, foreground_binary].mean()
                - unmasked_loo[:, ~foreground_binary].mean()
            )
        else:
            contrast = 0.0
        self.feature_quality = {
            "foreground_contrast": contrast,
            "foreground_area": float(foreground_binary.mean()),
            "loo_map_mean": float(unmasked_loo.mean()),
            "loo_map_q99": float(np.quantile(unmasked_loo, 0.99)),
        }
        combined = topk_mean(fused, self.topk_percent)
        self.image_stats = {
            "q_low": float(np.quantile(combined, 0.50)),
            "q_high": float(np.quantile(combined, 0.99)),
        }

    def score_encoded(self, encoded: Dict[str, np.ndarray], images=None) -> Dict[str, np.ndarray]:
        native_layers = encoded.get("patch_layers", [])
        visual = np.zeros(
            (encoded["patches"].shape[0],) + self.grid_size, dtype=np.float32
        )
        for layer_index, layer in enumerate(self.layers):
            kind = self.layer_kinds[layer_index] if layer_index < len(self.layer_kinds) else "native"
            if kind == "projected":
                query_features = encoded["patches"]
            else:
                if layer_index >= len(native_layers):
                    continue
                query_features = native_layers[layer_index]
            query = self._transform_query(query_features, layer)
            raw = self._raw_visual_maps(query, layer)
            if self.normalise_layers:
                low = float(layer.get("q_low", 0.0))  # type: ignore[arg-type]
                high = float(layer.get("q_high", 1.0))  # type: ignore[arg-type]
                raw = np.clip((raw - low) / max(high - low, 1e-6), 0.0, 1.0)
            visual += raw * float(self.layer_weights[layer_index])

        fused = visual
        if images is not None and self.foreground_masking:
            priors = np.stack(
                [self._prior(image, self.grid_size) for image in images], axis=0
            )
            fused = fused * priors
        image_scores = topk_mean(fused, self.topk_percent)
        return {
            "anomaly_map": fused.astype(np.float32),
            "image_score": image_scores.astype(np.float32),
            "visual_map": visual.astype(np.float32),
        }

    def score_from_features(self, patch_features: np.ndarray) -> Dict[str, np.ndarray]:
        return self.score_encoded({"patches": patch_features})

    def forward(self, images: Sequence) -> BaselineOutput:
        encoded = self.backbone.encode_images(images)
        scored = self.score_encoded(encoded, images=images)
        return BaselineOutput(
            anomaly_map=scored["anomaly_map"],
            image_score=scored["image_score"],
            patch_features=encoded["patches"],
            global_features=encoded["global"],
            prompt_similarity=None,
        )

    def support_statistics(
        self,
        support_images: Optional[Sequence] = None,
        low_quantile: float = 0.5,
        high_quantile: float = 0.99,
    ) -> Dict[str, float]:
        key = (float(low_quantile), float(high_quantile))
        if key not in self._stats_cache:
            self._stats_cache[key] = {
                "q_low": float(self.fused_stats["q_low"]),
                "q_high": float(self.fused_stats["q_high"]),
                "mean": float(self.fused_stats["q_low"]),
                "std": float(
                    max(self.fused_stats["q_high"] - self.fused_stats["q_low"], 1e-6)
                ),
                "image_q_low": float(self.image_stats["q_low"]),
                "image_q_high": float(self.image_stats["q_high"]),
            }
        return dict(self._stats_cache[key])


def build_baseline(backbone: CLIPBackbone, support_images: Sequence, category: str, config):
    """Build the configured baseline in the shared CLIP patch space."""
    from symbolicAD.config import Config

    if not isinstance(config, Config):
        raise TypeError("config must be a symbolicAD.config.Config")
    if config.model.baseline == "clip_adaptive":
        feature_source = str(config.model.feature_source).lower()
        if feature_source not in {"native", "projected", "both"}:
            raise ValueError(
                "model.feature_source must be fixed to native/projected/both "
                "before static-program compilation"
            )
        return CLIPAdaptiveFewShotBaseline(
            backbone,
            support_images,
            topk_percent=config.model.image_score_percent,
            knn=config.model.knn,
            max_bank_size=config.model.max_bank_size,
            spatial_radius=config.model.spatial_radius,
            spatial_weight=config.model.spatial_weight,
            local_radius=config.model.local_radius,
            local_weight=config.model.local_weight,
            whiten=config.model.whiten,
            whiten_components=config.model.whiten_components,
            whiten_eps=config.model.whiten_eps,
            foreground_masking=config.model.foreground_masking,
            foreground_soft=config.model.foreground_soft,
            feature_source=feature_source,
            normalise_layers=config.model.normalise_layers,
            projected_weight=config.model.projected_weight,
        )
    if config.model.baseline == "clip_fewshot":
        return CLIPFewShotBaseline(
            backbone,
            support_images,
            topk_percent=config.fusion.topk_percent,
            knn=config.model.knn,
            max_bank_size=config.model.max_bank_size,
            spatial_radius=config.model.spatial_radius,
            spatial_weight=config.model.spatial_weight,
        )
    raise ValueError(
        f"unsupported visual baseline {config.model.baseline!r}; "
        "use clip_adaptive or clip_fewshot"
    )
