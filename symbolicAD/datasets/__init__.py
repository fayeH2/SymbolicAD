"""Dataset adapters for the staged pipeline."""

from .mvtec_loco import (
    LOCO_CLASSES,
    collect_test_samples,
    load_gt_mask,
    load_image,
    select_support,
)

__all__ = [
    "LOCO_CLASSES",
    "collect_test_samples",
    "load_gt_mask",
    "load_image",
    "select_support",
]
