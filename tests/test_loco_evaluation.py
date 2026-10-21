import numpy as np

from scripts.evaluate_loco_checkpoint import binary_metrics, defect_group_indices


def test_loco_subgroups_reuse_good_images():
    samples = [
        {"defect_type": "good", "label": 0},
        {"defect_type": "logical_anomalies", "label": 1},
        {"defect_type": "structural_anomalies", "label": 1},
    ]
    np.testing.assert_array_equal(defect_group_indices(samples, "logical"), [0, 1])
    np.testing.assert_array_equal(defect_group_indices(samples, "structural"), [0, 2])
    np.testing.assert_array_equal(defect_group_indices(samples, "all"), [0, 1, 2])


def test_binary_metrics_reject_single_class_subsets():
    try:
        binary_metrics(np.ones(2, dtype=np.uint8), np.asarray([0.1, 0.2]))
    except ValueError as error:
        assert "both good and anomalous" in str(error)
    else:
        raise AssertionError("single-class AUROC must fail closed")
