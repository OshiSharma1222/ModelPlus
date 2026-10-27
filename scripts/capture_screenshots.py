"""Capture the README screenshots of every tab with a headless browser.

Usage:  python scripts/capture_screenshots.py
Requires: pip install -r requirements-dev.txt  (Playwright). Uses an installed
Chromium-based browser (Playwright's own, Chrome or Edge).
"""

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
VIEWPORT = {"width": 1440, "height": 1080}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.port", str(port),
         "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2)
            return proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("Streamlit server did not start")


def launch(p):
    for kwargs in ({}, {"channel": "chrome"}, {"channel": "msedge"}):
        try:
            return p.chromium.launch(headless=True, **kwargs)
        except Exception:  # noqa: BLE001 - try the next browser
            continue
    raise RuntimeError("No Chromium-based browser available; run `playwright install chromium`.")


def settle(page):
    page.wait_for_timeout(600)
    page.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden", timeout=120_000)
    page.wait_for_timeout(1800)


def main():
    port = free_port()
    server = start_server(port)
    try:
        with sync_playwright() as p:
            browser = launch(p)
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.locator(".mp-verdict .lbl").wait_for(timeout=120_000)
            settle(page)

            shots = [
                ("B", "Overview", "overview.png"),
                ("A", "Data drift", "drift_explorer.png"),
                ("B", "Model reliability", "model_reliability.png"),
                ("B", "Drift simulator", "drift_simulator.png"),
                ("A", "Audit report", "audit_report.png"),
            ]
            for preset, tab, filename in shots:
                button = "A — drift that doesn't matter" if preset == "A" else "B — failure that drift tests miss"
                page.get_by_role("button", name=button).click()
                settle(page)
                page.get_by_role("tab", name=tab).click()
                settle(page)
                page.screenshot(path=str(OUT / filename))
                print(f"saved figures/{filename}")
            browser.close()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
