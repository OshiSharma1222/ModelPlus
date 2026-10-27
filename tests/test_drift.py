import numpy as np
import pandas as pd

from modelpulse import config
from modelpulse.drift import (
    benjamini_hochberg, categorical_drift, detect_drift, drift_status, numeric_drift,
    population_stability_index,
)
from tests.conftest import make_sessions


def test_benjamini_hochberg_matches_hand_calculation():
    p = [0.01, 0.04, 0.03, 0.20]
    # sorted: 0.01*4/1=0.04, 0.03*4/2=0.06, 0.04*4/3=0.0533, 0.20*4/4=0.20
    # step-up minimum from the top: 0.04, 0.0533, 0.0533, 0.20
    np.testing.assert_allclose(benjamini_hochberg(p), [0.04, 0.05333333, 0.05333333, 0.20])


def test_benjamini_hochberg_keeps_nan_and_caps_at_one():
    out = benjamini_hochberg([np.nan, 0.9, 0.95])
    assert np.isnan(out[0])
    assert np.all(out[1:] <= 1.0)


def test_numeric_drift_detects_location_shift():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(0, 1, 2000))
    res = numeric_drift(ref, pd.Series(rng.normal(1, 1, 2000)))
    assert res["p_value"] < 1e-10
    assert res["effect"] > 0.3
    assert res["secondary"] > 0.8  # standardised mean difference


def test_numeric_drift_same_distribution_is_small():
    rng = np.random.default_rng(2)
    res = numeric_drift(pd.Series(rng.normal(size=3000)), pd.Series(rng.normal(size=3000)))
    assert res["effect"] < 0.05


def test_numeric_drift_ignores_missing_values():
    ref = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
    res = numeric_drift(ref, pd.Series([1.0, 2.0, np.nan, 3.0, 4.0]))
    assert res["effect"] == 0.0


def test_categorical_drift_effect_is_total_variation_distance():
    ref = pd.Series(["a"] * 500 + ["b"] * 500)
    cur = pd.Series(["a"] * 800 + ["b"] * 200)
    res = categorical_drift(ref, cur)
    assert abs(res["effect"] - 0.3) < 1e-12
    assert res["test"] == "Chi-square"
    assert res["p_value"] < 1e-10


def test_categorical_drift_uses_monte_carlo_when_counts_are_small():
    ref = pd.Series(["a"] * 50 + ["b"] * 2 + ["c"] * 1)
    cur = pd.Series(["a"] * 20 + ["b"] * 1 + ["d"] * 1)
    res = categorical_drift(ref, cur)
    assert res["test"].endswith("(Monte-Carlo)")
    assert 0 < res["p_value"] <= 1


def test_psi_zero_for_identical_samples():
    x = np.random.default_rng(3).normal(size=1000)
    assert population_stability_index(x, x) < 1e-9


def test_detect_drift_flags_only_shifted_feature():
    ref = make_sessions(3000, seed=10)
    cur = make_sessions(3000, seed=11)
    cur["ProductRelated_Duration"] *= 3
    drift = detect_drift(ref, cur)
    assert set(drift["feature"]) == set(config.FEATURES)
    assert drift.loc[drift["drifted"], "feature"].tolist() == ["ProductRelated_Duration"]
    status = drift_status(drift)
    assert status["status"] == "Warning"
    assert status["n_drifted"] == 1


def test_no_drift_between_samples_of_same_population():
    drift = detect_drift(make_sessions(3000, seed=20), make_sessions(3000, seed=21))
    assert drift_status(drift)["status"] == "Stable"
