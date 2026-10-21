from pathlib import Path

import torch

from symbolicAD.data import select_inpformer_support
from symbolicAD.training_utils import loco_safe_augmentation


def test_mvtec_loco_support_excludes_flattened_validation(tmp_path: Path):
    good = tmp_path / "pushpins" / "train" / "good"
    good.mkdir(parents=True)
    for index in range(32):
        (good / f"{index:03d}.png").touch()
        (good / f"{index:03d}_val.png").touch()

    selected = select_inpformer_support(
        tmp_path, "mvtec_loco", "pushpins", shot=4, seed=1
    )

    assert len(selected) == 4
    assert all(not path.stem.endswith("_val") for path in selected)
    assert selected == select_inpformer_support(
        tmp_path, "mvtec_loco", "pushpins", shot=4, seed=1
    )
    assert selected != select_inpformer_support(
        tmp_path, "mvtec_loco", "pushpins", shot=4, seed=2
    )


def test_mvtec_loco_support_is_nested_across_shots(tmp_path: Path):
    good = tmp_path / "screw_bag" / "train" / "good"
    good.mkdir(parents=True)
    for index in range(16):
        (good / f"{index:03d}.png").touch()

    k1 = select_inpformer_support(tmp_path, "mvtec_loco", "screw_bag", 1, 7)
    k2 = select_inpformer_support(tmp_path, "mvtec_loco", "screw_bag", 2, 7)
    k4 = select_inpformer_support(tmp_path, "mvtec_loco", "screw_bag", 4, 7)

    assert k1 == k2[:1]
    assert k2 == k4[:2]


def test_loco_safe_augmentation_keeps_view_count_and_channels():
    image = torch.linspace(0.0, 1.0, 3 * 32 * 32).reshape(3, 32, 32)

    augmented = loco_safe_augmentation(image)

    assert augmented.shape == (22, 3, 32, 32)
    assert torch.isfinite(augmented).all()
    assert float(augmented.min()) >= 0.0
    assert float(augmented.max()) <= 1.0
