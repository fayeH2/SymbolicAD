"""Vision backbones and baseline adapters."""

from .clip_backbone import CLIPBackbone
from .baseline_adapter import (
    CLIPFewShotBaseline,
    CLIPZeroShotBaseline,
    build_baseline,
)

__all__ = [
    "CLIPBackbone",
    "CLIPZeroShotBaseline",
    "CLIPFewShotBaseline",
    "build_baseline",
]
