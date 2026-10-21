"""SymbolicAD: Neuro-Symbolic Few-Shot Anomaly Detection Package."""

from .inpformer_adapter import INPFormerAdapter, build_inpformer

__version__ = "1.0.0"
__all__ = [
    "INPFormerAdapter",
    "build_inpformer",
]