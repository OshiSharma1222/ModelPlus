"""Smoke test: run the Streamlit script headlessly and click the presets."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_runs_and_presets_work(artifacts):
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    assert len(at.tabs) == 5

    at.button[[b.label for b in at.button].index("Experiment B: impact, no drift")].click().run()
    assert not at.exception
    assert at.session_state.scenario == "relationship_shift"
    drift_metric = next(m for m in at.metric if m.label.startswith("Drift status"))
    assert drift_metric.value == "Stable"

    at.selectbox(key="scenario").set_value("missing_values").run()
    assert not at.exception
    assert at.session_state.features == ["PageValues", "ExitRates"]
