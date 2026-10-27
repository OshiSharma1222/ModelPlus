"""End-to-end tests in a real browser against a running Streamlit server."""

import re

from playwright.sync_api import expect

PRESET_A = "A — drift that doesn't matter"
PRESET_B = "B — failure that drift tests miss"
TABS = ["Overview", "Data drift", "Model reliability", "Drift simulator", "Audit report"]


def settle(pg):
    pg.wait_for_timeout(400)
    pg.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden", timeout=120_000)
    pg.wait_for_timeout(500)


def click(pg, role, name):
    pg.get_by_role(role, name=name).click()
    settle(pg)


def expect_verdict(pg, text):
    """Auto-waits until the re-run has rendered the expected verdict."""
    expect(pg.locator(".mp-verdict .head")).to_have_text(text, timeout=60_000)


def assert_no_exception(pg):
    expect(pg.locator('[data-testid="stException"]')).to_have_count(0)


def test_initial_load_shows_control_run(page):
    expect(page).to_have_title(re.compile("ModelPulse"))
    expect_verdict(page, "Neither drift nor decline")
    for tab in TABS:
        expect(page.get_by_role("tab", name=tab)).to_be_visible()
    expect(page.locator(".mp-quad .cell.on")).to_have_text("Quiet")
    assert_no_exception(page)


def test_every_tab_renders_without_errors(page):
    click(page, "button", PRESET_B)
    for tab in TABS:
        click(page, "tab", tab)
        assert_no_exception(page)


def test_experiment_a_flags_drift_in_explorer(page):
    click(page, "button", PRESET_A)
    expect_verdict(page, "Drift without measurable decline")
    expect(page.locator(".mp-quad .cell.on")).to_have_text("Alert, no impact")

    click(page, "tab", "Data drift")
    first_row = page.locator(".mp-rank tbody tr").first
    expect(first_row).to_contain_text("ProductRelated_Duration")
    expect(page.locator(".mp-feature-head")).to_contain_text("drifted")
    panel = page.get_by_role("tabpanel", name="Data drift")
    expect(panel.locator(".js-plotly-plot")).to_have_count(3)  # distribution, ECDF, PCA


def test_experiment_b_is_a_silent_failure(page):
    click(page, "button", PRESET_B)
    expect_verdict(page, "Performance decline without detected drift")
    expect(page.locator(".mp-quad .cell.on")).to_have_text("Silent failure")
    expect(page.locator(".mp-readout")).to_contain_text("0 / 17")
    expect(page.locator(".mp-readout")).to_contain_text("0.669")


def test_reliability_tab_switches_models(page):
    click(page, "button", PRESET_B)
    click(page, "tab", "Model reliability")
    panel = page.get_by_role("tabpanel", name="Model reliability")
    expect(panel).to_contain_text("material decline")
    expect(panel).to_contain_text("decision threshold 0.41")
    expect(panel.locator(".mp-cm .c")).to_have_count(8)
    panel.get_by_role("radio", name="Naive Bayes").click()
    settle(page)
    expect(panel).to_contain_text("decision threshold 0.94")
    expect(panel.locator(".js-plotly-plot")).to_have_count(4)


def test_sidebar_scenario_select_runs_new_experiment(page):
    sidebar = page.locator('[data-testid="stSidebar"]')
    sidebar.locator('[data-testid="stSelectbox"]').first.click()
    page.get_by_role("option", name="Missing-value injection").click()
    expect(page.locator(".mp-eyebrow")).to_contain_text("missing_values", timeout=60_000)
    expect_verdict(page, re.compile("^Data-quality issue"))
    expect(page.locator(".mp-verdict .mp-tag")).to_have_text("+ data-quality failure")
    expect(page.locator(".mp-eyebrow")).to_contain_text("missing_values")


def test_severity_slider_reruns_experiment(page):
    click(page, "button", PRESET_A)
    expect(page.locator(".mp-eyebrow")).to_contain_text("severity 0.80")
    page.get_by_role("slider").focus()
    page.keyboard.press("ArrowLeft")
    settle(page)
    expect(page.locator(".mp-eyebrow")).to_contain_text("severity 0.75")


def test_simulator_shows_evidence_chain_and_sweep(page):
    click(page, "button", PRESET_B)
    click(page, "tab", "Drift simulator")
    panel = page.get_by_role("tabpanel", name="Drift simulator")
    expect(panel.locator(".mp-steps > div")).to_have_count(6)
    expect(panel.locator(".mp-card")).to_have_count(2)
    expect(panel.locator(".js-plotly-plot")).to_have_count(1)
    expect(panel).to_contain_text("PageValues: values shuffled jointly")


def test_report_downloads(page, tmp_path):
    click(page, "button", PRESET_B)
    click(page, "tab", "Audit report")
    with page.expect_download() as info:
        page.get_by_role("button", name="Download .md").click()
    md_path = tmp_path / info.value.suggested_filename
    info.value.save_as(md_path)
    text = md_path.read_text(encoding="utf-8")
    assert md_path.name == "modelpulse-relationship_shift-sev080-seed2026.md"
    assert text.startswith("# ModelPulse audit report")
    assert "Performance decline without detected drift" in text
    assert "## 8. Limitations" in text

    with page.expect_download() as info:
        page.get_by_role("button", name="Download .html").click()
    html_path = tmp_path / info.value.suggested_filename
    info.value.save_as(html_path)
    html = html_path.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and "<table>" in html


def test_narrow_screen_has_no_horizontal_scroll(browser, server_url):
    context = browser.new_context(viewport={"width": 420, "height": 900})
    pg = context.new_page()
    pg.goto(server_url, wait_until="networkidle")
    pg.locator(".mp-verdict .lbl").wait_for(timeout=120_000)
    settle(pg)
    overflow = pg.evaluate("""() => {
        const main = document.querySelector('[data-testid="stMain"]');
        return main.scrollWidth - main.clientWidth;
    }""")
    context.close()
    assert overflow <= 1
