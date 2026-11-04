"""Vision backbones and baseline adapters."""

from .clip_backbone import CLIPBackbone
from .baseline_adapter import (
    CLIPFewShotBaseline,
    build_baseline,
)

__all__ = [
    "CLIPBackbone",
    "CLIPFewShotBaseline",
    "build_baseline",
]
