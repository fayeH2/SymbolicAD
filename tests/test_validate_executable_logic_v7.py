import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.validate_executable_logic_v7 import _hist_metric


def test_histogram_auc_and_ap_match_distinct_score_bins():
    scores = np.asarray([0.05, 0.15, 0.45, 0.75, 0.95])
    labels = np.asarray([0, 1, 0, 1, 1])
    bins = np.floor(scores * 10).astype(int)
    positive = np.bincount(bins[labels == 1], minlength=10)
    negative = np.bincount(bins[labels == 0], minlength=10)
    auc, ap = _hist_metric(positive, negative)
    assert np.isclose(auc, roc_auc_score(labels, scores))
    assert np.isclose(ap, average_precision_score(labels, scores))


def test_histogram_auc_uses_half_credit_for_ties():
    positive = np.asarray([0, 2, 0])
    negative = np.asarray([0, 2, 0])
    auc, ap = _hist_metric(positive, negative)
    assert auc == 0.5
    assert ap == 0.5
