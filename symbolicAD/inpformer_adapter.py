"""Thin inference-only adapter around the existing INP-Former baseline."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DEFAULT_INPFORMER_ROOT = "/home/hjh/Desktop/H2VLR_compare/INP-Former"


def _load_baseline_api(inpformer_root):
    """Import INP-Former as an external, read-only source dependency."""
    root = Path(inpformer_root).expanduser().resolve()
    if not (root / "models" / "uad.py").is_file():
        raise FileNotFoundError(f"Invalid INP-Former root: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Keep the external baseline tree read-only in practice as well as by design.
    # Python otherwise creates __pycache__ files beside imported INP-Former modules.
    previous_no_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from dataset import get_data_transforms
        from models import vit_encoder
        from models.uad import INP_Former
        from models.vision_transformer import Aggregation_Block, Mlp, Prototype_Block
    finally:
        sys.dont_write_bytecode = previous_no_bytecode
    return {
        "root": root,
        "get_data_transforms": get_data_transforms,
        "vit_encoder": vit_encoder,
        "INP_Former": INP_Former,
        "Aggregation_Block": Aggregation_Block,
        "Mlp": Mlp,
        "Prototype_Block": Prototype_Block,
    }


def _cal_anomaly_maps(fs_list, ft_list, out_size):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    map_list = []
    for source, target in zip(fs_list, ft_list):
        score = 1.0 - F.cosine_similarity(source, target)
        score = F.interpolate(
            score.unsqueeze(1), size=out_size, mode="bilinear", align_corners=True
        )
        map_list.append(score)
    return torch.cat(map_list, dim=1).mean(dim=1, keepdim=True), map_list


def _project_feature_grids(
    source_features, target_features, projection_dim: int = 16, seed: int = 1307,
):
    """Compress INP semantic/residual tokens into spatial feature grids.

    A fixed random projection keeps entity-token caches small while ensuring
    that symbolic predicates are computed from the same INP representation as
    the baseline rather than from a disconnected detector feature space.
    """
    grids = []
    for layer_index, (source, target) in enumerate(zip(
        source_features, target_features
    )):
        if source.shape != target.shape or source.ndim not in (3, 4):
            raise ValueError("INP features must be aligned [B,N,C] or [B,C,H,W]")
        channel_count = source.shape[-1] if source.ndim == 3 else source.shape[1]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + layer_index)
        projection = torch.randn(
            channel_count, projection_dim, generator=generator,
            dtype=torch.float32,
        ).to(device=source.device, dtype=source.dtype)
        projection = projection / math.sqrt(float(projection_dim))
        if source.ndim == 4:
            normalized_source = F.normalize(source, dim=1)
            normalized_target = F.normalize(target, dim=1)
            semantic = torch.einsum("bchw,cp->bphw", normalized_source, projection)
            residual = torch.einsum(
                "bchw,cp->bphw", normalized_source - normalized_target, projection
            )
            feature = torch.cat([semantic, residual], dim=1)
        else:
            side = int(round(math.sqrt(source.shape[1])))
            if side * side != source.shape[1]:
                raise ValueError("INP patch tokens do not form a square grid")
            normalized_source = F.normalize(source, dim=-1)
            normalized_target = F.normalize(target, dim=-1)
            semantic = normalized_source @ projection
            residual = (normalized_source - normalized_target) @ projection
            feature = torch.cat([semantic, residual], dim=-1)
            feature = feature.transpose(1, 2).reshape(
                source.shape[0], 2 * projection_dim, side, side
            )
        grids.append(feature)
    return torch.stack(grids, dim=1)


def _gaussian_kernel(kernel_size=5, sigma=4.0):
    coordinate = torch.arange(kernel_size)
    x_grid = coordinate.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2
    kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((grid - mean) ** 2.0, dim=-1) / (2.0 * variance)
    )
    kernel = (kernel / kernel.sum()).view(1, 1, kernel_size, kernel_size)
    layer = nn.Conv2d(1, 1, kernel_size, groups=1, bias=False,
                      padding=kernel_size // 2)
    layer.weight.data = kernel
    layer.weight.requires_grad = False
    return layer


def build_inpformer(
    checkpoint: Union[str, Path],
    device: str,
    inpformer_root: Union[str, Path] = DEFAULT_INPFORMER_ROOT,
    encoder_name: str = "dinov2reg_vit_base_14",
    inp_num: int = 6,
):
    api = _load_baseline_api(inpformer_root)
    previous_cwd = Path.cwd()
    try:
        os.chdir(api["root"])
        encoder = api["vit_encoder"].load(encoder_name)
    finally:
        os.chdir(previous_cwd)
    
    # Patch DINOv2 encoder to add prepare_tokens method for INP-Former compatibility
    if hasattr(encoder, 'forward_features') and not hasattr(encoder, 'prepare_tokens'):
        def prepare_tokens(x):
            # DINOv2 forward_features returns dict with 'x_norm_patchtokens' of shape [B, N_reg + N_patches, C]
            # INP-Former expects prepare_tokens to return [B, 1 + num_register + N_patches, C]
            # where first token is CLS, then register tokens, then patch tokens
            out = encoder.forward_features(x)
            patch_tokens = out['x_norm_patchtokens']  # [B, N_reg + N_patches, C]
            
            # DINOv2 patch_tokens includes register tokens but NOT CLS token
            # INP-Former expects: [CLS, REG1, REG2, ..., PATCH1, PATCH2, ...]
            # DINOv2 gives: [REG1, REG2, REG3, REG4, PATCH1, PATCH2, ...] (no CLS)
            # We need to prepend a dummy CLS token
            B, N, C = patch_tokens.shape
            device, dtype = patch_tokens.device, patch_tokens.dtype
            cls_token = torch.zeros(B, 1, C, device=device, dtype=dtype)
            tokens = torch.cat([cls_token, patch_tokens], dim=1)  # [B, 1+N_reg+N_patches, C]
            
            # Debug: verify token count
            expected = 1 + encoder.num_register_tokens + (encoder.patch_embed.num_patches if hasattr(encoder, 'patch_embed') else 784)
            if tokens.shape[1] != 1 + encoder.num_register_tokens + 784:
                print(f"WARNING: Token count mismatch! Expected {1+encoder.num_register_tokens+784}, got {tokens.shape[1]}")
            
            return tokens
        encoder.prepare_tokens = prepare_tokens
    
    if "small" in encoder_name:
        embed_dim, num_heads = 384, 6
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "base" in encoder_name:
        embed_dim, num_heads = 768, 12
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif "large" in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError(f"Unsupported encoder: {encoder_name}")

    Mlp = api["Mlp"]
    Aggregation_Block = api["Aggregation_Block"]
    Prototype_Block = api["Prototype_Block"]
    bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.0)])
    prototypes = nn.ParameterList([nn.Parameter(torch.randn(inp_num, embed_dim))])
    extractor = nn.ModuleList([
        Aggregation_Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
        )
    ])
    decoder = nn.ModuleList([
        Prototype_Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
        )
        for _ in range(8)
    ])
    model = api["INP_Former"](
        encoder=encoder,
        bottleneck=bottleneck,
        aggregation=extractor,
        decoder=decoder,
        target_layers=target_layers,
        remove_class_token=True,
        fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        prototype_token=prototypes,
    )
    try:
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(Path(checkpoint), map_location="cpu")
    model.load_state_dict(state, strict=True)
    
# Monkey-patch INP_Former.forward to handle token count mismatch
    # DINOv2 with register tokens causes token count mismatch in intermediate features
    original_forward = model.forward
    
    def patched_forward(x):
        # Call original forward but catch the reshape error and handle it
        try:
            return original_forward(x)
        except RuntimeError as e:
            if "shape" in str(e) and "invalid for input of size" in str(e):
                # The error is from the reshape in INP_Former.forward
                # This happens because DINOv2's intermediate features don't have register tokens
                # We need to compute the correct side length from the actual feature dimensions
                # For now, we need to fix this in the external uad.py
                # Workaround: monkey-patch the INP_Former.forward method on the model class
                raise RuntimeError(
                    "INP-Former token count mismatch with DINOv2 register tokens. "
                    "The encoder's intermediate features don't include register tokens. "
                    "Fix required in external uad.py: modify side calculation to use "
                    "sqrt(L - num_register_tokens) instead of sqrt(L - 1 - num_register_tokens) "
                    "when CLS token is not in intermediate features."
                ) from e
            raise
    
    model.forward = patched_forward
    
    try:
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(Path(checkpoint), map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class INPFormerAdapter:
    def __init__(
        self,
        checkpoint: Union[str, Path],
        device: str = "cuda:0",
        inpformer_root: Union[str, Path] = DEFAULT_INPFORMER_ROOT,
        encoder_name: str = "dinov2reg_vit_base_14",
        inp_num: int = 6,
        input_size: int = 448,
        crop_size: int = 392,
        map_size: int = 256,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.map_size = map_size
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.inpformer_root = str(Path(inpformer_root).expanduser().resolve())
        self.encoder_name = encoder_name
        self.inp_num = inp_num
        self.input_size = input_size
        self.crop_size = crop_size
        self.api = _load_baseline_api(inpformer_root)
        self.transform, _ = self.api["get_data_transforms"](input_size, crop_size)
        self.model = build_inpformer(
            checkpoint, device, inpformer_root, encoder_name, inp_num
        )
        self.gaussian = _gaussian_kernel(kernel_size=5, sigma=4).to(device)

    @staticmethod
    def _to_pil(image) -> Image.Image:
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

    def infer_images(
        self,
        images: Sequence,
        batch_size: int = 8,
        identifiers: Optional[Sequence[str]] = None,
        return_layer_maps: bool = False,
        return_feature_grids: bool = False,
        feature_projection_dim: int = 16,
    ) -> Dict[str, np.ndarray]:
        """Infer paths, PIL images, or RGB arrays without writing temp files."""
        maps = []
        scores = []
        layer_maps = []
        feature_grids = []
        if identifiers is None:
            identifiers = [
                str(Path(item).resolve()) if isinstance(item, (str, Path))
                else f"memory://{index}"
                for index, item in enumerate(images)
            ]
        if len(identifiers) != len(images):
            raise ValueError("identifier/image count mismatch")
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch_images = images[start:start + batch_size]
                tensors = [self.transform(self._to_pil(image)) for image in batch_images]
                batch_tensor = torch.stack(tensors).to(self.device)
                print(f"DEBUG: batch_tensor shape = {batch_tensor.shape}")
                if batch_tensor.shape[-2:] != (self.crop_size, self.crop_size):
                    batch_tensor = batch_tensor[:, :, :self.crop_size, :self.crop_size]
                    print(f"DEBUG: Cropped to {batch_tensor.shape}")
                encoder_features, decoder_features, _ = self.model(batch_tensor)
                anomaly_map, per_layer = _cal_anomaly_maps(
                    encoder_features, decoder_features, batch_tensor.shape[-1]
                )
                anomaly_map = F.interpolate(
                    anomaly_map, size=self.map_size, mode="bilinear", align_corners=False
                )
                anomaly_map = self.gaussian(anomaly_map)[:, 0]
                flat = anomaly_map.flatten(1)
                top_count = max(1, int(flat.shape[1] * 0.01))
                image_score = torch.topk(flat, top_count, dim=1).values.mean(dim=1)
                maps.append(anomaly_map.cpu().numpy().astype(np.float32))
                scores.append(image_score.cpu().numpy().astype(np.float32))
                if return_layer_maps:
                    resized_layers = []
                    for layer_map in per_layer:
                        layer_map = F.interpolate(
                            layer_map, size=self.map_size, mode="bilinear",
                            align_corners=False,
                        )
                        layer_map = self.gaussian(layer_map)
                        resized_layers.append(layer_map[:, 0])
                    layer_maps.append(
                        torch.stack(resized_layers, dim=1).cpu().numpy().astype(np.float32)
                    )
                if return_feature_grids:
                    feature_grids.append(
                        _project_feature_grids(
                            encoder_features, decoder_features,
                            projection_dim=feature_projection_dim,
                        ).cpu().numpy().astype(np.float32)
                    )
        result = {
            "paths": np.asarray(list(identifiers)),
            "maps": np.concatenate(maps, axis=0) if maps else np.empty((0, self.map_size, self.map_size)),
            "scores": np.concatenate(scores, axis=0) if scores else np.empty((0,)),
        }
        if return_layer_maps:
            result["layer_maps"] = (
                np.concatenate(layer_maps, axis=0)
                if layer_maps else np.empty((0, 0, self.map_size, self.map_size), dtype=np.float32)
            )
        if return_feature_grids:
            channels = 2 * int(feature_projection_dim)
            result["feature_grids"] = (
                np.concatenate(feature_grids, axis=0)
                if feature_grids else np.empty(
                    (0, 0, channels, 0, 0), dtype=np.float32
                )
            )
        return result

    def infer(self, paths: Sequence[Union[str, Path]], batch_size: int = 8) -> Dict[str, np.ndarray]:
        normalized_paths = [str(Path(path).resolve()) for path in paths]
        return self.infer_images(paths, batch_size=batch_size, identifiers=normalized_paths)

    def cache_metadata(self) -> Dict[str, object]:
        return {
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": file_sha256(self.checkpoint),
            "inpformer_root": self.inpformer_root,
            "encoder": self.encoder_name,
            "inp_num": self.inp_num,
            "input_size": self.input_size,
            "crop_size": self.crop_size,
            "map_size": self.map_size,
            "cache_schema": 2,
        }

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def file_sha256(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_prediction_cache(
    path: Union[str, Path], predictions: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "paths": predictions["paths"],
        # float16 made a cached rerun produce slightly different facts/rules
        # from the first in-memory float32 run.
        "maps": predictions["maps"].astype(np.float32),
        "scores": predictions["scores"].astype(np.float32),
        "metadata_json": np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    }
    if "layer_maps" in predictions:
        payload["layer_maps"] = predictions["layer_maps"].astype(np.float32)
    if "feature_grids" in predictions:
        payload["feature_grids"] = predictions["feature_grids"].astype(np.float32)
    np.savez_compressed(path, **payload)
    return path


def load_prediction_cache(
    path: Union[str, Path], expected_paths: Sequence[Union[str, Path]],
    expected_metadata: Optional[Dict[str, object]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    path = Path(path)
    if not path.is_file():
        return None
    expected = np.asarray([
        str(item) if "://" in str(item) else str(Path(item).resolve())
        for item in expected_paths
    ])
    with np.load(path, allow_pickle=False) as payload:
        cached_paths = payload["paths"].astype(str)
        if not np.array_equal(cached_paths, expected):
            return None
        if expected_metadata is not None:
            if "metadata_json" not in payload:
                return None
            cached_metadata = json.loads(str(payload["metadata_json"].item()))
            if cached_metadata != expected_metadata:
                return None
        result = {
            "paths": cached_paths,
            "maps": payload["maps"].astype(np.float32),
            "scores": payload["scores"].astype(np.float32),
        }
        if "layer_maps" in payload:
            result["layer_maps"] = payload["layer_maps"].astype(np.float32)
        if "feature_grids" in payload:
            result["feature_grids"] = payload["feature_grids"].astype(np.float32)
        return result
