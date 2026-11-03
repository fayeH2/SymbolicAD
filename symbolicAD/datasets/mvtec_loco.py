"""MVTec LOCO dataset discovery and loading.

The local conversion stores every split directly below
``{category}/test/{group}`` and masks below
``{category}/ground_truth/{group}/{stem}_mask.png``.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
LOCO_CLASSES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]
GROUPS = ("good", "logical", "structural")


@dataclass(frozen=True)
class TestSample:
    path: Path
    label: int
    group: str
    defect_type: str
    mask_path: Optional[Path]

    @property
    def image_id(self) -> str:
        return self.path.stem

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "image_id": self.image_id,
            "label": int(self.label),
            "group": self.group,
            "defect_type": self.defect_type,
            "mask_path": str(self.mask_path) if self.mask_path else None,
        }


def _list_images(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.lower(),
    )


def normal_train_images(class_root: str | Path) -> List[Path]:
    class_root = Path(class_root)
    for name in ("good", "ok"):
        images = _list_images(class_root / "train" / name)
        if images:
            return images
    raise FileNotFoundError(f"No train/good images under {class_root}")


def select_support(
    class_root: str | Path,
    shot: int,
    seed: int = 668,
) -> List[Path]:
    """Deterministic K-shot support selection using train-only normal images."""
    class_root = Path(class_root)
    candidates = [
        path for path in normal_train_images(class_root) if not path.stem.endswith("_val")
    ]
    if shot < 1 or shot > len(candidates):
        raise ValueError(
            f"shot={shot}, but {len(candidates)} train-only normal images exist "
            f"for {class_root.name}"
        )
    token = f"symbolicad-v32:{class_root.name}:{seed}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    ordered = list(candidates)
    random.Random(derived).shuffle(ordered)
    return ordered[:shot]


def _group_from_dir(name: str) -> str:
    lowered = name.lower()
    if lowered in {"good", "ok"}:
        return "good"
    if "logical" in lowered:
        return "logical"
    if "structural" in lowered:
        return "structural"
    return lowered


def collect_test_samples(class_root: str | Path) -> List[TestSample]:
    class_root = Path(class_root)
    test_root = class_root / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(test_root)

    samples: List[TestSample] = []
    for group_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        group = _group_from_dir(group_dir.name)
        for image_path in _list_images(group_dir):
            mask_path = (
                class_root
                / "ground_truth"
                / group_dir.name
                / f"{image_path.stem}_mask.png"
            )
            samples.append(
                TestSample(
                    path=image_path,
                    label=0 if group == "good" else 1,
                    group=group,
                    defect_type=group_dir.name,
                    mask_path=mask_path if mask_path.is_file() else None,
                )
            )
    return samples


def filter_group(samples: Sequence[TestSample], group: str) -> List[TestSample]:
    """Return the evaluation subset for ``all`` / ``logical`` / ``structural``."""
    group = group.lower()
    if group == "all":
        return list(samples)
    if group == "good":
        return [item for item in samples if item.group == "good"]
    return [item for item in samples if item.group in {group, "good"}]


def limit_per_group(
    samples: Sequence[TestSample],
    max_per_group: int,
    seed: int = 668,
) -> List[TestSample]:
    if max_per_group <= 0:
        return list(samples)
    rng = random.Random(seed)
    buckets: Dict[str, List[TestSample]] = {}
    for sample in samples:
        buckets.setdefault(sample.group, []).append(sample)
    result: List[TestSample] = []
    for group in sorted(buckets):
        items = buckets[group]
        if len(items) > max_per_group:
            items = rng.sample(items, max_per_group)
        result.extend(items)
    return sorted(result, key=lambda item: str(item.path))


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGB")


def load_gt_mask(
    mask_path: Optional[str | Path],
    size: tuple[int, int],
) -> np.ndarray:
    """Load a binary mask at ``size=(height, width)``; missing masks are empty."""
    height, width = size
    if mask_path is None or not Path(mask_path).is_file():
        return np.zeros((height, width), dtype=np.uint8)
    with Image.open(mask_path) as handle:
        mask = handle.convert("L").resize((width, height), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 127).astype(np.uint8)
