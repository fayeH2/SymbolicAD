"""CLIP backbone exposing global, patch and text embeddings.

The same backbone feeds the baseline branch and the component-grounding branch,
so query-time inference performs a single CLIP forward pass per image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import open_clip
from PIL import Image


IMAGE_PREPROCESS_VERSION = "open_clip_full_frame_squash_v1"


def _as_pil(image) -> Image.Image:
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            return handle.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class CLIPBackbone:
    def __init__(
        self,
        model_name: str = "hf-hub:timm/vit_base_patch16_clip_224.openai",
        device: str = "cuda:0",
        image_size: int = 224,
        patch_size: int = 16,
        feature_layers: Sequence[int] = (),
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model_name = model_name
        self.requested_image_size = int(image_size)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            device=self.device,
            force_image_size=self.requested_image_size,
            # LOCO frames are often strongly rectangular.  The OpenCLIP
            # default shortest-side resize + centre crop silently discards
            # large parts of those frames.  Squashing preserves the complete
            # field of view and gives a map that can be resized back directly.
            image_resize_mode="squash",
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.patch_size = int(patch_size)
        native = getattr(self.model.visual, "image_size", (image_size, image_size))
        self.image_size = int(native[0]) if isinstance(native, (tuple, list)) else int(native)
        self.grid_size = self.image_size // self.patch_size
        self.embed_dim = int(getattr(self.model.visual, "output_dim", 0))
        # Layer indices are resolved lazily: `forward_intermediates` supports
        # negative indices but we normalise them to positive block positions so
        # cached metadata stays stable across backbones.
        num_blocks = len(getattr(self.model.visual, "transformer").resblocks)
        requested = []
        for value in feature_layers:
            index = int(value)
            if index < 0:
                index += num_blocks
            if 0 <= index < num_blocks and index not in requested:
                requested.append(index)
        self.feature_layers = tuple(sorted(requested))
        if self.feature_layers and self.feature_layers[-1] != num_blocks - 1:
            self.feature_layers = tuple(sorted(set(self.feature_layers) | {num_blocks - 1}))

    # ------------------------------------------------------------------ images
    def _batch_tensor(self, images: Sequence) -> torch.Tensor:
        tensors = [self.preprocess(_as_pil(image)) for image in images]
        return torch.stack(tensors).to(self.device)

    @torch.inference_mode()
    def encode_images(self, images: Sequence) -> Dict[str, np.ndarray]:
        """Return global, projected patch and multi-layer dense features.

        ``patches`` keeps the historical contract: the final block projected
        into the CLIP text space (512-d), used by component/rule grounding.
        ``patch_layers`` exposes native-width intermediates (768-d for ViT-B)
        for the PromptAD/AnomalyCLIP-style visual memory baseline.
        """
        if not images:
            empty = np.zeros((0, self.embed_dim), dtype=np.float32)
            return {
                "global": empty,
                "patches": empty.reshape(0, self.embed_dim, 0, 0),
                "patch_layers": [],
            }
        tensor = self._batch_tensor(images)
        projected, layers, global_features = self._dense_features(tensor)
        return {
            "global": global_features.cpu().numpy().astype(np.float32),
            "patches": projected.cpu().numpy().astype(np.float32),
            "patch_layers": [
                layer.cpu().numpy().astype(np.float32) for layer in layers
            ],
        }

    def _dense_features(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        visual = self.model.visual
        if not hasattr(visual, "forward_intermediates"):
            raise RuntimeError(
                "CLIP vision tower does not expose forward_intermediates; "
                "cannot build component patch maps."
            )
        indices = list(self.feature_layers) or [-1]
        output = visual.forward_intermediates(
            tensor,
            indices=indices,
            normalize_intermediates=False,
            output_fmt="NCHW",
            output_extra_tokens=False,
        )
        intermediates = (
            output.get("image_intermediates") if isinstance(output, dict) else output[0]
        )
        if not intermediates:
            raise RuntimeError("CLIP forward_intermediates returned no layers")
        if isinstance(intermediates, torch.Tensor):
            intermediates = [intermediates]

        # The last requested block is always the final transformer block, so it
        # can be projected into the text space exactly like encode_image().
        final_native = intermediates[-1]
        feature = final_native.permute(0, 2, 3, 1)  # [B, h, w, C]
        feature = visual.ln_post(feature)
        projection = getattr(visual, "proj", None)
        if projection is not None:
            feature = feature @ projection
        projected = F.normalize(feature.permute(0, 3, 1, 2).float(), dim=1)

        layers = []
        for layer in intermediates:
            if layer.ndim == 3:
                batch, tokens, channels = layer.shape
                side = int(round(tokens ** 0.5))
                layer = layer.transpose(1, 2).reshape(batch, channels, side, side)
            if layer.ndim != 4:  # pragma: no cover - defensive
                raise RuntimeError(
                    f"unexpected patch feature shape {tuple(layer.shape)}"
                )
            layers.append(F.normalize(layer.float(), dim=1))

        global_features = output.get("image_features") if isinstance(output, dict) else None
        if global_features is None:
            global_features = self.model.encode_image(tensor)
        global_features = F.normalize(global_features.float(), dim=-1)
        return projected, layers, global_features

    # ------------------------------------------------------------------- text
    @torch.inference_mode()
    def encode_text(self, prompts: Sequence[str]) -> np.ndarray:
        if not prompts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        tokens = self.tokenizer(list(prompts)).to(self.device)
        features = self.model.encode_text(tokens)
        features = F.normalize(features.float(), dim=-1)
        return features.cpu().numpy().astype(np.float32)

    @property
    def logit_scale(self) -> float:
        scale = getattr(self.model, "logit_scale", None)
        if scale is None:
            return 100.0
        return float(scale.exp().item())

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
