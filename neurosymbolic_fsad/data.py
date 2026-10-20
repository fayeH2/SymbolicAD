"""Dataset discovery without importing benchmark-specific training code."""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Union


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
NORMAL_DIRS = {"good", "ok"}
MVTEC_CLASS_ORDER = [
    "carpet", "grid", "leather", "tile", "wood", "bottle", "cable",
    "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush",
    "transistor", "zipper",
]
VISA_CLASS_ORDER = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
MPDD_CLASS_ORDER = [
    "bracket_black", "bracket_brown", "bracket_white", "connector",
    "metal_plate", "tubes",
]

# The medical INP-Former checkpoints were trained with the canonical support
# indices used by all methods in H2VLR_compare.  Keeping this small mapping in
# the isolated neuro-symbolic project prevents an accidental support/checkpoint
# mismatch when the multi-dataset launcher builds a rule library.
MEDICAL_SUPPORT_SPECS = {
    "brainmri": {
        "class_name": "brain",
        "seed_file": Path(
            "/home/hjh/Desktop/H2VLR_V2/datasets/seeds_BrainAD/brain/"
            "selected_samples_per_run.txt"
        ),
    },
    "liverct": {
        "class_name": "liver",
        "seed_file": Path(
            "/home/hjh/Desktop/H2VLR_V2/datasets/seeds_liverAD/liver/"
            "selected_samples_per_run.txt"
        ),
    },
    "busi": {
        "class_name": "breast",
        "seed_file": Path(
            "/home/hjh/Desktop/H2VLR_V2/datasets/seeds_busi/breast/"
            "selected_samples_per_run.txt"
        ),
    },
}


def list_images(directory: Union[str, Path]) -> List[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )


def normal_train_images(class_root: Union[str, Path]) -> List[Path]:
    class_root = Path(class_root)
    for name in ("good", "ok"):
        images = list_images(class_root / "train" / name)
        if images:
            return images
    raise FileNotFoundError(f"No train/good or train/ok images under {class_root}")


def select_support(
    class_root: Union[str, Path], shot: int, seed: int = 668,
) -> List[Path]:
    candidates = normal_train_images(class_root)
    if shot < 1 or shot > len(candidates):
        raise ValueError(f"shot={shot}, but {len(candidates)} normal train images exist")
    token = f"normality:{Path(class_root).resolve()}:{shot}:{seed}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return random.Random(derived_seed).sample(candidates, shot)


def select_inpformer_support(
    data_root: Union[str, Path],
    dataset: str,
    class_name: str,
    shot: int,
    seed: int = 1,
) -> List[Path]:
    """Reproduce the support sampling used to train the provided checkpoints."""
    data_root = Path(data_root).expanduser().resolve()
    dataset_key = dataset.lower()
    if dataset_key == "mvtec":
        if class_name not in MVTEC_CLASS_ORDER:
            raise ValueError(f"Unknown MVTec class for official support: {class_name}")
        rng = random.Random(seed)
        for category in MVTEC_CLASS_ORDER:
            candidates = normal_train_images(data_root / category)
            chosen = rng.sample(candidates, shot)
            if category == class_name:
                return chosen
    if dataset_key == "visa":
        if class_name not in VISA_CLASS_ORDER:
            raise ValueError(f"Unknown VisA class for official support: {class_name}")
        # INP_Former_Few_Shot.py seeds Python's global RNG once, then samples
        # classes in this exact order for both MVTec and VisA.
        rng = random.Random(seed)
        for category in VISA_CLASS_ORDER:
            candidates = normal_train_images(data_root / category)
            chosen = rng.sample(candidates, shot)
            if category == class_name:
                return chosen
    if dataset_key in MEDICAL_SUPPORT_SPECS:
        spec = MEDICAL_SUPPORT_SPECS[dataset_key]
        if class_name != spec["class_name"]:
            raise ValueError(
                f"Medical dataset {dataset_key} expects class "
                f"{spec['class_name']!r}, got {class_name!r}"
            )
        seed_file = Path(spec["seed_file"])
        if not seed_file.is_file():
            raise FileNotFoundError(
                f"Canonical support manifest is missing for {dataset_key}: {seed_file}"
            )
        indices_by_shot = {}
        for line in seed_file.read_text(encoding="utf-8").splitlines():
            match = re.match(r"#(\d+):\s*(.*)", line.strip())
            if match:
                indices_by_shot[int(match.group(1))] = [
                    int(token) for token in match.group(2).split()
                ]
        if shot not in indices_by_shot:
            raise ValueError(
                f"No canonical {shot}-shot split for {dataset_key} in {seed_file}"
            )
        candidates = normal_train_images(data_root / class_name)
        indices = indices_by_shot[shot]
        if not indices or max(indices) >= len(candidates):
            raise ValueError(
                f"Invalid canonical indices {indices} for {len(candidates)} images"
            )
        return [candidates[index] for index in indices]
    if dataset_key == "mvtec_loco":
        # The local MVTec LOCO conversion flattened the official validation
        # split into ``train/good`` and appended ``_val`` to those filenames.
        # Validation objects are physically disjoint from the official train
        # split and must not silently enter a strict K-shot support set.
        candidates = [
            path for path in normal_train_images(data_root / class_name)
            if not path.stem.endswith("_val")
        ]
        if shot < 1 or shot > len(candidates):
            raise ValueError(
                f"shot={shot}, but {len(candidates)} train-only normal images "
                f"exist for MVTec LOCO class {class_name}"
            )
        # Versioned full-token hashing fixes the historical first-eight-byte
        # collision (all ``mvtec_loco`` classes/seeds previously shared the
        # bytes ``inp:mvte``).  ``shot`` is deliberately absent from the token
        # so K=1/2/4/8 supports are nested prefixes for a fixed seed.
        token = f"inp-support-v2:{dataset_key}:{class_name}:{seed}".encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
        ordered = list(candidates)
        random.Random(derived_seed).shuffle(ordered)
        return ordered[:shot]
    # Reproduce INP_Former_Industrial.py. Its historical seed uses the first
    # eight bytes of the token, so all classes of one dataset share a stream.
    candidates = normal_train_images(data_root / class_name)
    token = f"inp:{dataset_key}:{class_name}:{shot}:{seed}"
    derived_seed = int.from_bytes(token.encode("utf-8")[:8], "big")
    return random.Random(derived_seed).sample(candidates, shot)


def collect_test_samples(
    class_root: Union[str, Path], test_subdir: str = "test",
) -> List[Dict[str, object]]:
    class_root = Path(class_root)
    test_root = class_root / test_subdir
    if not test_root.is_dir():
        raise FileNotFoundError(test_root)
    samples = []
    for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        # MVTec AD 2 stores masks below test_public/ground_truth.  Those files
        # are supervision artifacts, not test images.
        if defect_dir.name.lower() in {"ground_truth", "masks", "mask"}:
            continue
        label = int(defect_dir.name.lower() not in NORMAL_DIRS)
        paths = sorted(
            (
                path for path in defect_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: str(path.relative_to(defect_dir)).lower(),
        )
        for path in paths:
            relative_parent = path.parent.relative_to(defect_dir)
            defect_type = defect_dir.name
            if relative_parent != Path("."):
                defect_type = f"{defect_type}/{relative_parent.as_posix()}"
            samples.append({
                "path": path,
                "label": label,
                "defect_type": defect_type,
            })
    return samples


def stratified_limit(
    samples: Sequence[Dict[str, object]],
    max_normal: int = 0,
    max_anomaly: int = 0,
    seed: int = 668,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    normal = [item for item in samples if int(item["label"]) == 0]
    anomaly = [item for item in samples if int(item["label"]) == 1]
    if max_normal and len(normal) > max_normal:
        normal = rng.sample(normal, max_normal)
    if max_anomaly and len(anomaly) > max_anomaly:
        anomaly = rng.sample(anomaly, max_anomaly)
    return sorted(normal + anomaly, key=lambda item: str(item["path"]))
