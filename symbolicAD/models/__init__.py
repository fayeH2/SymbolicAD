"""Vision backbones and baseline adapters."""

from .clip_backbone import CLIPBackbone
from .baseline_adapter import CLIPZeroShotBaseline

__all__ = ["CLIPBackbone", "CLIPZeroShotBaseline"]
