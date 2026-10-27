"""End-to-end checks on the real dataset (downloaded on first run)."""

import numpy as np

from modelpulse import config
from modelpulse.experiment import run_experiment, severity_sweep
from modelpulse.report import build_html, build_markdown


def test_baseline_models_beat_chance(artifacts):
    for key in config.MODEL_LABELS:
        res = run_experiment(artifacts, "control", bootstrap_reps=0)
        assert res.metrics[key]["baseline"]["roc_auc"] > 0.8


def test_control_has_no_drift_and_no_change(artifacts):
    res = run_experiment(artifacts, "control", bootstrap_reps=50)
    assert res.drift_summary["n_drifted"] == 0
    assert res.verdict["title"] == "Neither drift nor decline"
    for comp in res.comparison.values():
        assert np.allclose(comp["delta"], 0)


def test_experiment_a_drift_without_decline(artifacts):
    res = run_experiment(artifacts, "numeric_shift", 0.8, bootstrap_reps=100)
    assert res.verdict["drift"] and not res.verdict["decline"]


def test_experiment_b_decline_without_drift(artifacts):
    res = run_experiment(artifacts, "relationship_shift", 0.8, ["PageValues"], bootstrap_reps=100)
    assert not res.verdict["drift"] and res.verdict["decline"]


def test_missing_values_reported_as_quality_issue(artifacts):
    res = run_experiment(artifacts, "missing_values", 0.6, bootstrap_reps=50)
    assert res.verdict["quality_issue"]
    assert res.verdict["title"].startswith("Data-quality issue")


def test_report_contains_all_sections(artifacts):
    res = run_experiment(artifacts, "categorical_shift", 0.5, bootstrap_reps=20)
    text = build_markdown(res, artifacts)
    for heading in ["Verdict", "Experiment settings", "Data quality", "Detected changes",
                    "Model performance", "Warnings", "Recommended investigations", "Limitations"]:
        assert heading in text
    assert "<table>" in build_html(text)


def test_severity_sweep_shape(artifacts):
    sweep = severity_sweep(artifacts, "relationship_shift", ["PageValues"], [0.0, 1.0])
    assert len(sweep) == 2 * len(config.MODEL_LABELS)
    tree = sweep[sweep["model"] == "Decision Tree"].set_index("severity")
    assert tree.loc[1.0, "roc_auc"] < tree.loc[0.0, "roc_auc"] - 0.1
