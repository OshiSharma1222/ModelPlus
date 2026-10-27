import numpy as np
import pytest

from modelpulse.evaluation import (
    best_f1_threshold, calibration_table, classification_metrics, confusion_counts,
    expected_calibration_error, mcnemar_test, paired_bootstrap, roc_auc, roc_curve_points,
)


def test_confusion_counts():
    c = confusion_counts([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert c == {"tp": 2, "fp": 1, "tn": 1, "fn": 1}


def test_classification_metrics_hand_example():
    y = [1, 1, 0, 0, 1]
    prob = [0.9, 0.2, 0.1, 0.7, 0.6]
    m = classification_metrics(y, prob, threshold=0.5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(3 / 5)
    # positive scores {0.9, 0.2, 0.6} vs negatives {0.1, 0.7}: 4 of 6 pairs ordered correctly
    assert m["roc_auc"] == pytest.approx(4 / 6)


def test_roc_auc_perfect_random_and_ties():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5
    assert np.isnan(roc_auc([1, 1], [0.2, 0.3]))


def test_roc_auc_matches_trapezoid_area_under_curve():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    prob = np.clip(y * 0.3 + rng.normal(0.4, 0.2, 500), 0, 1)
    pts = roc_curve_points(y, prob)
    area = np.trapezoid(pts["tpr"], pts["fpr"])
    assert roc_auc(y, prob) == pytest.approx(area, abs=1e-9)


def test_f1_zero_when_no_true_positives():
    m = classification_metrics([1, 0, 0], [0.1, 0.2, 0.3], threshold=0.5)
    assert m["f1"] == 0.0
    assert np.isnan(m["precision"])


def test_best_f1_threshold_separates_classes():
    y = np.array([0] * 50 + [1] * 50)
    prob = np.r_[np.full(50, 0.3), np.full(50, 0.7)]
    t = best_f1_threshold(y, prob)
    assert 0.3 < t <= 0.7


def test_paired_bootstrap_identical_predictions_have_zero_delta():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    out = paired_bootstrap(y, p, p, 0.5, reps=50, seed=1)
    assert np.allclose(out["delta"], 0)
    assert np.allclose(out[["delta_low", "delta_high"]], 0)


def test_paired_bootstrap_detects_degradation():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 1000)
    good = np.clip(y * 0.5 + rng.normal(0.25, 0.15, 1000), 0, 1)
    bad = rng.random(1000)
    out = paired_bootstrap(y, good, bad, 0.5, reps=200, seed=2).set_index("metric")
    assert out.loc["roc_auc", "delta"] < -0.3
    assert out.loc["roc_auc", "delta_high"] < 0


def test_calibration_of_perfectly_calibrated_scores():
    rng = np.random.default_rng(3)
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    table = calibration_table(y, p)
    assert table["n"].sum() == 200_000


def test_mcnemar_no_difference():
    y = [1, 0, 1, 0]
    res = mcnemar_test(y, [1, 0, 1, 0], [1, 0, 1, 0])
    assert res["p_value"] == 1.0
