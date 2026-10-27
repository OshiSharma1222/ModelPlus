import numpy as np
import pandas as pd
import pytest

from modelpulse import config
from modelpulse.data import Preprocessor, standardise_types, stratified_split
from modelpulse.quality import quality_report


def test_stratified_split_keeps_class_balance(sessions):
    train, test = stratified_split(sessions, test_size=0.3, seed=1)
    assert len(train) + len(test) == len(sessions)
    rate = sessions[config.TARGET].mean()
    assert abs(train[config.TARGET].mean() - rate) < 0.01
    assert abs(test[config.TARGET].mean() - rate) < 0.01


def test_stratified_split_is_reproducible(sessions):
    a, _ = stratified_split(sessions, seed=5)
    b, _ = stratified_split(sessions, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_standardise_types_parses_boolean_target():
    raw = pd.DataFrame({c: [1, 2] for c in config.NUMERIC_FEATURES})
    for c in config.CATEGORICAL_FEATURES:
        raw[c] = ["x", "y"]
    raw[config.TARGET] = ["TRUE", "FALSE"]
    out = standardise_types(raw)
    assert out[config.TARGET].tolist() == [1, 0]


def test_standardise_types_rejects_missing_columns(sessions):
    with pytest.raises(ValueError):
        standardise_types(sessions.drop(columns=["PageValues"]))


def test_preprocessor_uses_training_statistics_only(sessions):
    train, test = stratified_split(sessions, seed=1)
    prep = Preprocessor().fit(train)
    test = test.copy()
    test.loc[0, "BounceRates"] = np.nan
    test.loc[1, "VisitorType"] = "Martian"
    out = prep.transform(test)
    assert out.loc[0, "BounceRates"] == pytest.approx(train["BounceRates"].median())
    assert out.loc[1, "VisitorType"] in prep.levels["VisitorType"]
    assert not out.isna().any().any()


def test_preprocessor_log_transforms_skewed_features(sessions):
    prep = Preprocessor().fit(sessions)
    out = prep.transform(sessions)
    np.testing.assert_allclose(out["PageValues"], np.log1p(sessions["PageValues"]))
    np.testing.assert_allclose(out["BounceRates"], sessions["BounceRates"])


def test_quality_report_counts_problems(sessions):
    prep = Preprocessor().fit(sessions)
    bad = sessions.copy()
    bad.loc[:9, "PageValues"] = np.nan
    bad.loc[10:14, "BounceRates"] = 1.5
    bad.loc[15, "Administrative_Duration"] = -3
    bad.loc[16:18, "Month"] = "Smarch"
    rep = quality_report(bad, prep)
    t = rep["table"].set_index("feature")
    assert rep["missing_cells"] == 10
    assert t.loc["BounceRates", "invalid"] == 5
    assert t.loc["Administrative_Duration", "invalid"] == 1
    assert t.loc["Month", "invalid"] == 3
    assert t.loc["Month", "unseen_levels"] == 3
    assert rep["invalid_values"] == 9
