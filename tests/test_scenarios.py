import numpy as np
import pandas as pd
import pytest

from modelpulse import config
from modelpulse.scenarios import SCENARIOS, simulate_batch


def test_control_returns_identical_copy(sessions):
    out, log = simulate_batch(sessions, "control", 1.0)
    pd.testing.assert_frame_equal(out, sessions)
    assert out is not sessions
    assert "No transformation" in log[-1]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_scenarios_preserve_rows_and_labels(sessions, scenario):
    out, _ = simulate_batch(sessions, scenario, 0.7)
    assert len(out) == len(sessions)
    assert list(out.columns) == list(sessions.columns)
    np.testing.assert_array_equal(out[config.TARGET], sessions[config.TARGET])


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_scenarios_are_reproducible(sessions, scenario):
    a, _ = simulate_batch(sessions, scenario, 0.5, seed=42)
    b, _ = simulate_batch(sessions, scenario, 0.5, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_zero_severity_is_a_no_op(sessions):
    out, _ = simulate_batch(sessions, "numeric_shift", 0.0)
    pd.testing.assert_frame_equal(out, sessions)


def test_numeric_shift_scales_and_clips(sessions):
    out, _ = simulate_batch(sessions, "numeric_shift", 1.0,
                            ["ProductRelated_Duration", "BounceRates", "Administrative"])
    np.testing.assert_allclose(out["ProductRelated_Duration"], sessions["ProductRelated_Duration"] * 3)
    assert out["BounceRates"].max() <= 1.0
    assert (out["Administrative"] % 1 == 0).all()
    pd.testing.assert_series_equal(out["ExitRates"], sessions["ExitRates"])


def test_categorical_shift_moves_mass_to_second_level(sessions):
    out, _ = simulate_batch(sessions, "categorical_shift", 1.0, ["VisitorType"])
    before = (sessions["VisitorType"] == "New_Visitor").mean()
    after = (out["VisitorType"] == "New_Visitor").mean()
    expected = before + 0.6 * (1 - before)
    assert after == pytest.approx(expected, abs=2 / len(sessions))


def test_missing_injection_rate(sessions):
    out, _ = simulate_batch(sessions, "missing_values", 1.0, ["PageValues"])
    assert out["PageValues"].isna().mean() == pytest.approx(0.5, abs=1 / len(sessions))
    assert out["ExitRates"].notna().all()


def test_relationship_shift_preserves_marginal_distribution(sessions):
    out, _ = simulate_batch(sessions, "relationship_shift", 1.0, ["PageValues"])
    np.testing.assert_array_equal(np.sort(out["PageValues"]), np.sort(sessions["PageValues"]))
    assert not out["PageValues"].equals(sessions["PageValues"])


def test_invalid_inputs_raise(sessions):
    with pytest.raises(ValueError):
        simulate_batch(sessions, "nope", 0.5)
    with pytest.raises(ValueError):
        simulate_batch(sessions, "numeric_shift", 1.5)
    with pytest.raises(ValueError):
        simulate_batch(sessions, "numeric_shift", 0.5, ["VisitorType"])
