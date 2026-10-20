"""Small INP-Former training primitives without evaluation-only dependencies.

The upstream ``utils.py`` imports scikit-image, pandas, matplotlib and the
CUDA AUPRO evaluator at module import time.  Generic baseline training needs
none of those packages, so the isolated trainer keeps the four required
operations here instead of mutating the upstream repository.
"""

from __future__ import annotations

import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _affine(image: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.to(dtype=image.dtype, device=image.device)
    matrix = matrix.unsqueeze(0).repeat(image.shape[0], 1, 1)
    grid = F.affine_grid(matrix, image.size(), align_corners=False)
    return F.grid_sample(
        image, grid, padding_mode="reflection", align_corners=False
    )


def _rotate(image: torch.Tensor, angle: float) -> torch.Tensor:
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    matrix = torch.tensor([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
    ])
    return _affine(image, matrix)


def _translate(image: torch.Tensor, horizontal: float, vertical: float) -> torch.Tensor:
    matrix = torch.tensor([
        [1.0, 0.0, horizontal],
        [0.0, 1.0, vertical],
    ])
    return _affine(image, matrix)


def augmentation(image: torch.Tensor) -> torch.Tensor:
    """Reproduce the upstream 22-view support augmentation."""
    image = image.unsqueeze(0)
    views = [image]
    for angle in (
        -np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16,
        np.pi / 16, np.pi / 8, 3 * np.pi / 16, np.pi / 4,
    ):
        views.append(_rotate(image, float(angle)))
    for horizontal, vertical in (
        (0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2),
        (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1),
    ):
        views.append(_translate(image, horizontal, vertical))
    views.append(torch.flip(image, dims=(-1,)))
    grayscale = (
        0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    )
    views.append(grayscale.repeat(1, 3, 1, 1))
    views.extend(torch.rot90(image, turns, dims=(-2, -1)) for turns in (1, 2, 3))
    augmented = torch.cat(views, dim=0)
    return augmented[torch.randperm(augmented.size(0))]


def loco_safe_augmentation(image: torch.Tensor) -> torch.Tensor:
    """Return 22 views that preserve LOCO logical relations.

    The upstream augmentation contains horizontal flips, 90-degree rotations,
    grayscale conversion, and translations as large as 20% of the frame.
    Those transforms can invert left/right predicates, erase color predicates,
    or crop countable components.  This controlled alternative retains the
    same number of views while limiting changes to small rigid motions and
    global illumination changes that preserve count, ordering, and color
    identity.
    """
    image = image.unsqueeze(0)
    views = [image]
    for angle in (
        -np.pi / 36, -np.pi / 48, -np.pi / 72,
        np.pi / 72, np.pi / 48, np.pi / 36,
    ):
        views.append(_rotate(image, float(angle)))
    for horizontal, vertical in (
        (0.03, 0.0), (-0.03, 0.0), (0.0, 0.03), (0.0, -0.03),
        (0.02, 0.02), (-0.02, 0.02), (-0.02, -0.02), (0.02, -0.02),
    ):
        views.append(_translate(image, horizontal, vertical))
    for factor in (0.94, 0.97, 1.03, 1.06):
        views.append(torch.clamp(image * factor, 0.0, 1.0))
    spatial_mean = image.mean(dim=(-2, -1), keepdim=True)
    for factor in (0.94, 1.02, 1.06):
        views.append(torch.clamp((image - spatial_mean) * factor + spatial_mean, 0.0, 1.0))
    augmented = torch.cat(views, dim=0)
    if augmented.size(0) != 22:
        raise AssertionError(f"expected 22 LOCO-safe views, got {augmented.size(0)}")
    return augmented[torch.randperm(augmented.size(0))]


def _modify_grad(tensor: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return tensor * factor.expand_as(tensor)


def global_cosine_hm_adaptive(encoded, decoded, y: int = 3):
    cosine = torch.nn.CosineSimilarity()
    loss = 0.0
    for source, target in zip(encoded, decoded):
        source = source.detach()
        with torch.no_grad():
            point_distance = 1 - cosine(source, target).unsqueeze(1).detach()
            factor = (point_distance / point_distance.mean().clamp_min(1e-12)) ** y
        loss = loss + torch.mean(
            1 - cosine(
                source.reshape(source.shape[0], -1),
                target.reshape(target.shape[0], -1),
            )
        )
        target.register_hook(partial(_modify_grad, factor=factor))
    return loss / len(encoded)


class WarmCosineScheduler(_LRScheduler):
    def __init__(
        self, optimizer, base_value: float, final_value: float,
        total_iters: int, warmup_iters: int = 0, start_warmup_value: float = 0.0,
    ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_iters = min(warmup_iters, total_iters)
        warmup = np.linspace(start_warmup_value, base_value, warmup_iters)
        remaining = total_iters - warmup_iters
        if remaining:
            iterations = np.arange(remaining)
            cosine = final_value + 0.5 * (base_value - final_value) * (
                1 + np.cos(np.pi * iterations / remaining)
            )
        else:
            cosine = np.empty((0,), dtype=float)
        self.schedule = np.concatenate((warmup, cosine))
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for _ in self.base_lrs]
        return [float(self.schedule[self.last_epoch]) for _ in self.base_lrs]
