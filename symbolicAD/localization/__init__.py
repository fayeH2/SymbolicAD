"""Responsibility maps, semantic rescoring and map fusion."""

from .responsibility import build_logic_heatmap, build_atom_heatmap
from .semantic_rescore import rescore_responsibility
from .fusion import (
    fuse_maps,
    logic_image_score,
    final_image_score,
    normalize_map,
    topk_mean,
)

__all__ = [
    "build_logic_heatmap",
    "build_atom_heatmap",
    "rescore_responsibility",
    "fuse_maps",
    "logic_image_score",
    "final_image_score",
    "normalize_map",
    "topk_mean",
]
