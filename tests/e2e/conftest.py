"""Browser end-to-end fixtures: a real Streamlit server driven by Playwright.

The tests are skipped automatically when Playwright or a Chromium-based browser
is not available. Install with:  pip install -r requirements-dev.txt
and, if no Chrome/Edge is installed,  playwright install chromium
"""

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sync_api = pytest.importorskip("playwright.sync_api")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server_url(artifacts):
    """Start `streamlit run app.py` once for the whole e2e session."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
         "--server.port", str(port), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 120
    while True:
        try:
            urllib.request.urlopen(f"{url}/_stcore/health", timeout=2)
            break
        except OSError:
            if time.time() > deadline or proc.poll() is not None:
                proc.kill()
                pytest.fail("Streamlit server did not start")
            time.sleep(0.5)
    yield url
    proc.terminate()
    proc.wait(timeout=20)


@pytest.fixture(scope="session")
def browser():
    with sync_api.sync_playwright() as p:
        for kwargs in ({}, {"channel": "chrome"}, {"channel": "msedge"}):
            try:
                b = p.chromium.launch(headless=True, **kwargs)
                break
            except Exception:  # noqa: BLE001 - try the next installed browser
                continue
        else:
            pytest.skip("no Chromium-based browser available for Playwright")
        yield b
        b.close()


@pytest.fixture
def page(browser, server_url):
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
    pg = context.new_page()
    errors = []
    pg.on("pageerror", lambda exc: errors.append(str(exc)))
    pg.goto(server_url, wait_until="networkidle")
    pg.locator(".mp-verdict .lbl").wait_for(timeout=120_000)
    settle(pg)
    yield pg
    assert not errors, f"JavaScript errors in the page: {errors}"
    context.close()


def settle(pg):
    """Wait for Streamlit to finish re-running the script."""
    pg.wait_for_timeout(400)
    pg.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden", timeout=120_000)
    pg.wait_for_timeout(500)
