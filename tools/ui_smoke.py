"""End-to-end smoke test of the dashboard in a real browser.

Drives the running app (backend on :8000, Vite on :5173) with Playwright:
plans Shanghai -> Rotterdam, switches to SUEZ_BLOCK and re-plans from the
scenario alert, picks an option, exports it as CSV, then opens the model and
supplier views and replays a recent plan at phone width. Fails on any console
error, failed request or missing element, and saves screenshots.

Usage: python tools/ui_smoke.py [out_dir] [--browser msedge|chrome|chromium]
(Playwright's own Chromium needs `playwright install chromium` first.)
"""
import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:5173/"


def pick_hub(page, field, text):
    page.fill(field, text)
    page.wait_for_selector(".suggestions button", timeout=10_000)
    page.locator(".suggestions button").first.click()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", nargs="?", default="ui-screens")
    parser.add_argument("--browser", default="msedge")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    problems = []

    def check(ok, what):
        if not ok:
            problems.append(what)

    with sync_playwright() as p:
        channel = None if args.browser == "chromium" else args.browser
        browser = p.chromium.launch(channel=channel, headless=True)
        page = browser.new_page(viewport={"width": 1680, "height": 1000}, accept_downloads=True)
        page.on("console", lambda m: m.type == "error" and problems.append(f"console: {m.text}"))
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.on("response", lambda r: r.status >= 400 and problems.append(f"{r.status} {r.url}"))

        page.goto(URL)
        page.wait_for_selector(".route-map .leaflet-tile-loaded", timeout=20_000)
        for _ in range(60):  # let the NLP warm-up finish so live news is scored
            if "Engine ready" in page.inner_text("header"):
                break
            time.sleep(1)

        pick_hub(page, "#hub-source", "Shanghai")
        pick_hub(page, "#hub-dest", "Rotterdam")
        page.click("button.primary")
        page.wait_for_selector(".option", timeout=30_000)

        # Choosing a scenario flags the recent plan it disrupts; re-plan from the alert.
        page.select_option("#scenario", "SUEZ_BLOCK")
        alert = page.locator(".notice.warn", has_text="disrupts")
        check(alert.count() == 1, "scenario alert missing for a plan through Suez")
        alert.locator("button").first.click()
        page.wait_for_selector("text=Planned under", timeout=30_000)
        options = page.locator(".option")
        check(options.count() >= 1, "no route options rendered")
        options.last.click()
        check(page.locator(".option").last.get_attribute("aria-pressed") == "true", "clicked option not selected")
        check(page.locator(".legs li").count() >= 2, "voyage plan has no legs")
        time.sleep(2)  # the route draws on the chart
        page.screenshot(path=str(out / "routes.png"))

        with page.expect_download() as download:
            page.click("text=Export CSV")
        csv_path = out / download.value.suggested_filename
        download.value.save_as(csv_path)
        rows = csv_path.read_text(encoding="utf-8-sig").splitlines()
        check(csv_path.suffix == ".csv" and rows[0].startswith('"leg"') and rows[-1].startswith('"total"'),
              f"unexpected CSV export {csv_path.name}")
        check(page.locator(".recent li").count() == 2, "recent plans should list the normal and the Suez plan")

        page.click("text=Model evaluation")
        page.wait_for_selector(".panel .recharts-surface", timeout=15_000)
        time.sleep(1)
        page.screenshot(path=str(out / "model.png"))

        page.click("text=Route planner")
        page.click("text=Supplier intelligence")
        page.wait_for_selector(".supplier-table tbody tr", timeout=15_000)
        time.sleep(0.5)
        page.screenshot(path=str(out / "suppliers.png"))

        # Phone width: replay the newest recent plan (kept in localStorage).
        page.click("text=Route planner")
        page.set_viewport_size({"width": 420, "height": 900})
        page.locator(".recent button").first.click()
        page.wait_for_selector(".option", timeout=30_000)
        time.sleep(2)
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        check(overflow <= 0, f"page scrolls sideways at phone width by {overflow}px")
        page.screenshot(path=str(out / "phone.png"), full_page=True)
        browser.close()

    for problem in problems:
        print("FAIL:", problem)
    print(f"screenshots in {out}/" if not problems else "")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
