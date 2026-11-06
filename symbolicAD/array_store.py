"""Small, inspectable replacement for opaque ``.npz`` output files.

An array bundle is a directory containing one ``.npy`` file per array and a
JSON manifest.  Arrays remain lossless and memory-map friendly, while metadata
can be inspected without loading NumPy.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, Mapping, Tuple

import numpy as np


ARRAY_BUNDLE_VERSION = "symbolicad_array_bundle_v1"
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")


def bundle_path(path: str | Path) -> Path:
    """Return a suffix-free bundle path, stripping legacy ``.npz`` names."""
    path = Path(path)
    return path.with_suffix("") if path.suffix.lower() == ".npz" else path


def save_array_bundle(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    root = bundle_path(path)
    root.mkdir(parents=True, exist_ok=True)
    entries: Dict[str, Dict[str, object]] = {}
    for key, value in arrays.items():
        if not _SAFE_KEY.fullmatch(key):
            raise ValueError(f"unsafe array key: {key!r}")
        array = np.asarray(value)
        filename = f"{key}.npy"
        np.save(root / filename, array, allow_pickle=False)
        entries[key] = {
            "file": filename,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    manifest = {
        "format": ARRAY_BUNDLE_VERSION,
        "arrays": entries,
        "metadata": dict(metadata or {}),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


def load_array_bundle(
    path: str | Path,
    mmap_mode: str | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    root = bundle_path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != ARRAY_BUNDLE_VERSION:
        raise RuntimeError(f"unsupported array bundle format in {root}")
    arrays = {
        key: np.load(root / entry["file"], allow_pickle=False, mmap_mode=mmap_mode)
        for key, entry in manifest["arrays"].items()
    }
    return arrays, dict(manifest.get("metadata", {}))
