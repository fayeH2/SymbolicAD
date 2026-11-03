"""SymbolicAD V3.2: staged neuro-symbolic anomaly detection.

The package is organised around a "run first, then verify and replace"
principle.  Stage one produces baseline / logic / final anomaly maps and
metrics; every intermediate artefact is kept on disk for diagnosis.
"""

from .config import Config, load_config

__all__ = ["Config", "load_config"]
__version__ = "3.2.0"
