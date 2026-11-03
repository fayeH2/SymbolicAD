from pathlib import Path

import numpy as np
import pytest

from symbolicAD.evaluation.image_metrics import image_metrics
from symbolicAD.evaluation.pixel_metrics import pixel_metrics
from symbolicAD.evaluation.pro_metric import compute_pro

LOCO_ROOT = Path("/home/hjh/Desktop/0Datasets/anomaly/industrial/MVTecLOCO/MVTecLOCO")


def test_image_metrics_perfect_separation():
    metrics = image_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["auroc"] == 1.0
    assert 0.0 <= metrics["ap"] <= 1.0
    assert 0.0 <= metrics["f1max"] <= 1.0


def test_pixel_metrics_pooled():
    predictions = [np.array([[0.1, 0.9]], dtype=np.float32)]
    masks = [np.array([[0, 1]], dtype=np.uint8)]
    metrics = pixel_metrics(predictions, masks)
    assert metrics["auroc"] == 1.0


def test_compute_pro_single_region():
    prediction = np.zeros((8, 8), dtype=np.float32)
    prediction[2:5, 2:5] = 1.0
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    metrics = compute_pro([prediction], [mask], thresholds=8)
    assert metrics["regions"] == 1
    assert 0.0 <= metrics["aupro"] <= 1.0


@pytest.mark.skipif(not LOCO_ROOT.is_dir(), reason="MVTec LOCO dataset not available")
def test_loco_dataset_discovery():
    from symbolicAD.datasets.mvtec_loco import collect_test_samples, select_support

    class_root = LOCO_ROOT / "juice_bottle"
    support = select_support(class_root, shot=2)
    assert len(support) == 2
    samples = collect_test_samples(class_root)
    groups = {sample.group for sample in samples}
    assert {"good", "logical", "structural"} <= groups
    anomaly = next(sample for sample in samples if sample.label == 1)
    assert anomaly.mask_path is not None
