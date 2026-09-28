"""End-to-end smoke test of the dashboard in a real browser.

Drives the running app (backend on :8000, Vite on :5173) with Playwright:
Shanghai -> Rotterdam under SUEZ_BLOCK, card selection, the model view and the
supplier view. Fails on any console error or failed request, and saves
screenshots.

Usage: python tools/ui_smoke.py [out_dir] [--browser msedge|chrome|chromium]
(Playwright's own Chromium needs `playwright install chromium` first.)
"""
import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:5173/"


def pick_hub(page, index, text):
    page.locator(".sc-input").nth(index).fill(text)
    page.wait_for_selector(".search-result", timeout=10_000)
    page.locator(".search-result").first.click()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", nargs="?", default="ui-screens")
    parser.add_argument("--browser", default="msedge")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    problems = []

    with sync_playwright() as p:
        channel = None if args.browser == "chromium" else args.browser
        browser = p.chromium.launch(channel=channel, headless=True)
        page = browser.new_page(viewport={"width": 1680, "height": 1000})
        page.on("console", lambda m: m.type == "error" and problems.append(f"console: {m.text}"))
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.on("response", lambda r: r.status >= 400 and problems.append(f"{r.status} {r.url}"))

        page.goto(URL)
        page.wait_for_selector(".route-map .leaflet-tile-loaded", timeout=20_000)
        for _ in range(60):  # let the NLP warm-up finish so live news is scored
            if "FULLY OPERATIONAL" in page.inner_text("header"):
                break
            time.sleep(1)

        pick_hub(page, 0, "Shanghai")
        pick_hub(page, 1, "Rotterdam")
        page.select_option(".sc-select >> nth=4", "SUEZ_BLOCK")
        page.click(".sc-btn-execute")
        page.wait_for_selector(".path-card", timeout=30_000)
        cards = page.locator(".path-card")
        if cards.count() < 1:
            problems.append("no route cards rendered")
        cards.last.click()
        time.sleep(2)
        page.screenshot(path=str(out / "routes.png"))

        page.click("text=MODEL EVALUATION")
        page.wait_for_selector(".model-card", timeout=15_000)
        time.sleep(1)
        page.screenshot(path=str(out / "model.png"))

        page.click("text=ROUTE RECOMMENDER")
        page.click("text=SUPPLIER INTELLIGENCE")
        page.wait_for_selector(".supplier-table tbody tr", timeout=15_000)
        page.screenshot(path=str(out / "suppliers.png"))
        browser.close()

    for problem in problems:
        print("FAIL:", problem)
    print(f"screenshots in {out}/" if not problems else "")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
