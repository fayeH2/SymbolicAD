#!/usr/bin/env python3
"""Train an isolated INP-Former baseline on any MVTec-like dataset.

The baseline repository is imported read-only.  This script exists for local
datasets that are not hard-coded by INP_Former_Few_Shot.py or
INP_Former_Industrial.py (currently MVTec2, MVTec LOCO and VID-AD).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.init import trunc_normal_
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from neurosymbolic_fsad.data import select_inpformer_support
from neurosymbolic_fsad.training_utils import (
    WarmCosineScheduler,
    augmentation,
    global_cosine_hm_adaptive,
    loco_safe_augmentation,
    setup_seed,
)


class AugmentedSupports(Dataset):
    """Official 22-way augmentation cached in host RAM, optionally fp16."""

    def __init__(self, paths, transform, augmentation, storage_dtype: str):
        dtype = torch.float16 if storage_dtype == "float16" else torch.float32
        self.images = []
        for path in paths:
            with Image.open(path) as handle:
                image = transform(handle.convert("RGB"))
            self.images.extend(row.to(dtype=dtype) for row in augmentation(image))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index].float()


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inpformer-root", required=True)
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--support-seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--encoder", choices=("dinov2reg_vit_base_14",),
        default="dinov2reg_vit_base_14",
        help="This isolated trainer currently instantiates the base/768 architecture.",
    )
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--crop-size", type=int, default=392)
    parser.add_argument("--inp-num", type=int, default=6)
    parser.add_argument("--augmentation-storage", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--augmentation-policy",
        choices=("official22", "loco_safe22"),
        default="official22",
        help="Use the upstream views or a relation-preserving LOCO control.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if checkpoint.is_file() and not args.force:
        print(json.dumps({"status": "resumed", "checkpoint": str(checkpoint)}))
        return 0

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Check the NVIDIA driver/container device mapping before training."
        )

    inp_root = Path(args.inpformer_root).expanduser().resolve()
    if not inp_root.is_dir():
        raise FileNotFoundError(inp_root)
    sys.path.insert(0, str(inp_root))
    from dataset import get_data_transforms
    from models import vit_encoder
    from models.uad import INP_Former
    from models.vision_transformer import Aggregation_Block, Mlp, Prototype_Block
    from optimizers import StableAdamW
    setup_seed(args.support_seed)
    random.seed(args.support_seed)
    data_transform, _ = get_data_transforms(args.input_size, args.crop_size)
    support = []
    for class_name in args.classes:
        support.extend(select_inpformer_support(
            args.data_root, args.dataset, class_name, args.shot, args.support_seed
        ))
    augmentation_fn = (
        augmentation if args.augmentation_policy == "official22"
        else loco_safe_augmentation
    )
    train_data = AugmentedSupports(
        support, data_transform, augmentation_fn, args.augmentation_storage
    )
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
    )
    if not len(loader):
        raise ValueError("The augmented support set is smaller than one batch")

    device = args.device
    encoder = vit_encoder.load(args.encoder)
    embed_dim, num_heads = 768, 12
    bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.0)])
    prototypes = nn.ParameterList([
        nn.Parameter(torch.randn(args.inp_num, embed_dim)) for _ in range(1)
    ])
    extractor = nn.ModuleList([
        Aggregation_Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
        ) for _ in range(1)
    ])
    decoder = nn.ModuleList([
        Prototype_Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
        ) for _ in range(8)
    ])
    model = INP_Former(
        encoder=encoder,
        bottleneck=bottleneck,
        aggregation=extractor,
        decoder=decoder,
        target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
        remove_class_token=True,
        fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
        prototype_token=prototypes,
    ).to(device)
    trainable = nn.ModuleList([bottleneck, decoder, extractor, prototypes])
    for module in trainable.modules():
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    optimizer = StableAdamW(
        [{"params": trainable.parameters()}], lr=1e-3, betas=(0.9, 0.999),
        weight_decay=1e-4, amsgrad=True, eps=1e-10,
    )
    scheduler = WarmCosineScheduler(
        optimizer, base_value=1e-3, final_value=1e-4,
        total_iters=args.epochs * len(loader), warmup_iters=100,
    )
    started = time.time()
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses, reconstruction_losses, global_losses = [], [], []
        gradient_norms, clipped_gradient_norms = [], []
        gradient_parameter_fractions, finite_gradient_fractions = [], []
        for images in loader:
            images = images.to(device, non_blocking=True)
            encoded, decoded, global_loss = model(images)
            reconstruction_loss = global_cosine_hm_adaptive(encoded, decoded, y=3)
            loss = reconstruction_loss + 0.2 * global_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                trainable.parameters(), max_norm=0.1
            )
            parameters = list(trainable.parameters())
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            clipped_gradient_norm = torch.linalg.vector_norm(torch.stack([
                gradient.detach().norm(2) for gradient in gradients
            ]), 2) if gradients else torch.tensor(0.0, device=device)
            finite_gradient_fraction = (
                float(np.mean([
                    bool(torch.isfinite(gradient).all().item()) for gradient in gradients
                ])) if gradients else 0.0
            )
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))
            reconstruction_losses.append(float(reconstruction_loss.item()))
            global_losses.append(float(global_loss.item()))
            gradient_norms.append(float(gradient_norm.item()))
            clipped_gradient_norms.append(float(clipped_gradient_norm.item()))
            gradient_parameter_fractions.append(
                float(len(gradients) / max(len(parameters), 1))
            )
            finite_gradient_fractions.append(finite_gradient_fraction)
        epoch_row = {
            "epoch": epoch + 1,
            "total_loss": float(np.mean(losses)),
            "reconstruction_loss": float(np.mean(reconstruction_losses)),
            "global_loss": float(np.mean(global_losses)),
            "gradient_norm_pre_clip": float(np.mean(gradient_norms)),
            "gradient_norm_post_clip": float(np.mean(clipped_gradient_norms)),
            "gradient_parameter_fraction": float(np.mean(gradient_parameter_fractions)),
            "finite_gradient_fraction": float(np.mean(finite_gradient_fractions)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "finite": bool(
                np.isfinite(losses).all()
                and np.isfinite(gradient_norms).all()
                and np.isfinite(clipped_gradient_norms).all()
                and min(finite_gradient_fractions, default=0.0) == 1.0
            ),
            "elapsed_seconds": round(time.time() - started, 1),
        }
        history.append(epoch_row)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(json.dumps({
                "dataset": args.dataset,
                "epochs": args.epochs,
                **epoch_row,
            }), flush=True)

    window = min(10, len(history))
    first_mean = float(np.mean([row["total_loss"] for row in history[:window]]))
    last_mean = float(np.mean([row["total_loss"] for row in history[-window:]]))
    tail_x = np.arange(window, dtype=float)
    tail_y = np.asarray([row["total_loss"] for row in history[-window:]], dtype=float)
    tail_slope = float(np.polyfit(tail_x, tail_y, 1)[0]) if window > 1 else 0.0
    convergence = {
        "finite_all_epochs": bool(all(row["finite"] for row in history)),
        "first_window_mean": first_mean,
        "last_window_mean": last_mean,
        "last_over_first": float(last_mean / max(first_mean, 1e-12)),
        "tail_slope_per_epoch": tail_slope,
        "loss_decreased": bool(last_mean < first_mean),
    }

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    history_path = checkpoint.parent / "training_history.json"
    _dump(history_path, {
        "schema_version": "isolated-generic-inpformer-training-history-v2",
        "history": history,
        "convergence": convergence,
    })
    source_paths = [
        Path(__file__).resolve(),
        PROJECT / "neurosymbolic_fsad" / "data.py",
        PROJECT / "neurosymbolic_fsad" / "training_utils.py",
    ]
    _dump(checkpoint.parent / "training_manifest.json", {
        "schema_version": "isolated-generic-inpformer-v2",
        "dataset": args.dataset,
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "classes": args.classes,
        "shot": args.shot,
        "support_seed": args.support_seed,
        "support": [str(path.resolve()) for path in support],
        "support_sha256": {
            str(path.resolve()): _sha256(path) for path in support
        },
        "support_selector_schema": (
            "mvtec_loco_train_only_sha256_nested_v2"
            if args.dataset.lower() == "mvtec_loco" else "legacy_dataset_selector_v1"
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "augmentation_policy": args.augmentation_policy,
        "support_split_policy": (
            "official_train_only_excluding_flattened_val_suffix"
            if args.dataset.lower() == "mvtec_loco" else "dataset_train_normal"
        ),
        "encoder": args.encoder,
        "input_size": args.input_size,
        "crop_size": args.crop_size,
        "inp_num": args.inp_num,
        "checkpoint": str(checkpoint),
        "training_history": str(history_path),
        "convergence": convergence,
        "source_sha256": {
            str(path.relative_to(PROJECT)): _sha256(path) for path in source_paths
        },
        "baseline_repository_modified": False,
    })
    print(json.dumps({"status": "trained", "checkpoint": str(checkpoint)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
