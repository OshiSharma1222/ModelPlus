"""Fast in-process app tests with Streamlit's AppTest (no browser)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PRESET_B = "B — failure that drift tests miss"


@pytest.fixture
def app(artifacts):
    return AppTest.from_file(APP, default_timeout=300).run()


def page_text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_app_renders_five_tabs_without_errors(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Overview", "Data drift", "Model reliability",
                                          "Drift simulator", "Audit report"]
    assert "Neither drift nor decline" in page_text(app)


def test_preset_b_shows_silent_failure(app):
    next(b for b in app.button if b.label == PRESET_B).click().run()
    assert not app.exception
    assert app.session_state.scenario == "relationship_shift"
    text = page_text(app)
    assert "Performance decline without detected drift" in text
    assert "0 / 17" in text


@pytest.mark.parametrize("scenario,expected", [
    ("numeric_shift", "Drift without measurable decline"),
    ("categorical_shift", "Drift without measurable decline"),
    ("missing_values", "Data-quality issue with performance decline"),
])
def test_every_scenario_renders(app, scenario, expected):
    app.selectbox(key="scenario").set_value(scenario).run()
    assert not app.exception
    assert app.session_state.features == list(
        __import__("modelpulse.scenarios", fromlist=["SCENARIOS"]).SCENARIOS[scenario].default_features)
    assert expected in page_text(app)


def test_switching_verdict_model_and_inspected_feature(app):
    app.selectbox(key="scenario").set_value("categorical_shift").run()
    assert app.session_state.drift_feature == "VisitorType"  # top-ranked change is preselected
    app.radio(key="primary").set_value("nb").run()
    app.selectbox(key="drift_feature").set_value("Month").run()
    assert not app.exception
    assert "ROC-AUC · NB" in page_text(app)
