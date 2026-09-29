"""End-to-end smoke test of the dashboard in a real browser.

Drives the running app (backend on :8000, Vite on :5173) with Playwright:
runs the LA port strike example (ships must divert via Oakland), checks that
editing an input marks those routes stale, plans Shanghai -> Rotterdam by
search (and swaps the ends once), switches to SUEZ_BLOCK and re-plans from
the scenario alert, picks an option, points at a leg, exports the route as
CSV and JSON and the printed report as PDF, shows costs in euros, plans
again to compare with the last run, then opens the model and supplier views
(and rejects a negative inventory), reopens a saved plan at phone width, and
leaves the planner right after planning a few times. Fails on any console
error, failed request or missing element, and saves screenshots.

Usage: python tools/ui_smoke.py [out_dir] [--browser msedge|chrome|chromium]
(Playwright's own Chromium needs `playwright install chromium` first.)
"""
import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OPTION = ".suggestions [role=option]"


def pick_hub(page, field, text):
    page.fill(field, text)
    page.wait_for_selector(OPTION, timeout=10_000)
    page.locator(OPTION).first.click()


def go(page, name):
    """Click a page link; every page is kept in the document, so pick the visible one."""
    page.locator("button.nav-button:visible", has_text=name).first.click()


def expected_refusal(url, status):
    # The engine refuses impossible plans with 422 and the reason, which the page shows.
    return status == 422 and url.endswith("/api/recommend")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", nargs="?", default="ui-screens")
    parser.add_argument("--browser", default="msedge")
    parser.add_argument("--url", default="http://localhost:5173/",
                        help="the dev server, or http://127.0.0.1:8000/ for the dashboard run.py serves")
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
        page.on("console", lambda m: m.type == "error" and "status of 422" not in m.text
                and problems.append(f"console: {m.text}"))
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.on("response", lambda r: r.status >= 400 and not expected_refusal(r.url, r.status)
                and problems.append(f"{r.status} {r.url}"))
        tiles = []  # OpenStreetMap blocks tile requests that carry no Referer
        page.on("request", lambda r: "tile.openstreetmap.org" in r.url and tiles.append(r))

        page.goto(args.url)
        page.wait_for_selector(".route-map .leaflet-tile-loaded", timeout=20_000)
        for _ in range(60):  # let the NLP warm-up finish so live news is scored
            if "Engine ready" in page.inner_text("header"):
                break
            time.sleep(1)

        # One-click example: during the LA port strike, ships divert via Oakland.
        page.click("text=Shanghai to Los Angeles by sea during the port strike")
        page.wait_for_selector("text=Planned under LA Port Strike", timeout=30_000)
        check("Oakland" in page.inner_text(".legs"), "LA strike example did not divert via Oakland")
        check(page.locator(".route-map .leaflet-tooltip.endpoint").count() == 2, "origin and destination not labelled")

        pick_hub(page, "#hub-source", "Shanghai")
        # The destination by keyboard: arrows move through the suggestions, Enter picks.
        page.fill("#hub-dest", "Rotterdam")
        page.wait_for_selector(OPTION, timeout=10_000)
        page.press("#hub-dest", "ArrowDown")
        check(page.get_attribute("#hub-dest", "aria-activedescendant") == "hub-dest-option-0", "arrow keys do not move through suggestions")
        page.press("#hub-dest", "Enter")
        check(page.input_value("#hub-dest") == "Port of Rotterdam" and page.locator(OPTION).count() == 0,
              "Enter does not pick the highlighted suggestion")
        check(page.locator(".notice.warn", has_text="Inputs changed").count() == 1,
              "editing an input does not mark the routes shown as stale")
        check(page.locator("button", has_text="Export CSV").is_disabled(), "stale routes can still be exported")
        page.click("text=Swap origin and destination")
        check(page.input_value("#hub-source") == "Port of Rotterdam", "swap did not exchange the ends")
        page.click("text=Swap origin and destination")
        page.select_option("#scenario", "NORMAL")
        page.select_option("#mode", "any")
        page.click("button.primary")
        page.wait_for_selector("text=Planned under", state="detached", timeout=30_000)
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
        page.locator(".legs li").nth(2).hover()
        check(page.locator(".route-map path.leg-focus").count() == 1, "pointing at a leg does not highlight it on the map")
        page.mouse.move(5, 5)
        page.screenshot(path=str(out / "routes.png"))

        with page.expect_download() as download:
            page.click("text=Export CSV")
        csv_path = out / download.value.suggested_filename
        download.value.save_as(csv_path)
        rows = csv_path.read_text(encoding="utf-8-sig").splitlines()
        check(csv_path.suffix == ".csv" and rows[0].startswith('"leg"') and rows[-1].startswith('"total"'),
              f"unexpected CSV export {csv_path.name}")
        check(page.locator(".recent li").count() == 3, "recent plans should list the LA, normal and Suez plans")

        with page.expect_download() as download:
            page.click("text=Export JSON")
        exported = json.loads(Path(download.value.path()).read_text(encoding="utf-8"))
        check(exported["request"]["scenario"] == "SUEZ_BLOCK"
              and len(exported["recommendations"]) == page.locator(".option").count(),
              "JSON export does not match the routes shown")
        page.pdf(path=str(out / "report.pdf"), format="A4")

        if page.locator("#currency option[value=EUR]").count():  # needs the ECB rates source
            page.select_option("#currency", "EUR")
            check("€" in page.inner_text(".options"), "costs not shown in euros")
            check(page.locator(".notice.warn", has_text="changed").count() == 0, "changing the currency made the routes stale")
            page.select_option("#currency", "USD")

        page.click("button.primary")  # the same plan again: compared with the last run
        page.wait_for_selector("text=Since your plan of", timeout=30_000)
        check(page.locator(".recent li").count() == 3, "planning the same thing again added a duplicate recent plan")

        go(page, "Model evaluation")
        check(page.url.endswith("#model"), "the model page has no address of its own")
        page.wait_for_selector(".panel .recharts-surface", timeout=15_000)
        check(page.locator(".scores tbody tr").count() == 7, "threat-intelligence scores missing")
        check(page.locator("text=Best possible").count() >= 1, "delay-model ceiling missing")
        time.sleep(1)
        page.screenshot(path=str(out / "model.png"))

        page.go_back()  # the browser's Back button returns to the planner, results intact
        page.wait_for_selector(".option:visible", timeout=15_000)
        check(page.locator(".notice", has_text="Since your plan of").count() == 1, "coming back lost the planned routes")
        go(page, "Supplier intelligence")
        page.wait_for_selector(".supplier-table tbody tr", timeout=15_000)
        page.wait_for_selector(".advice .urgency", timeout=15_000)
        time.sleep(0.5)
        page.screenshot(path=str(out / "suppliers.png"))
        page.fill("#inventory", "-5")
        check(page.locator("#inventory-error").count() == 1, "a negative inventory is not flagged")
        check(page.locator(".advice", has_text="Fix the highlighted inputs").count() == 1, "advice shown for a negative inventory")
        page.fill("#inventory", "1000")
        page.wait_for_selector(".advice .urgency", timeout=15_000)
        page.reload()  # a reload stays on the page it was on
        page.wait_for_selector(".supplier-table tbody tr", timeout=15_000)
        check(page.url.endswith("#suppliers") and page.locator(".supplier-table:visible").count() == 1,
              "reloading the supplier page went back to the planner")

        # Phone width: reopen the newest recent plan as saved (kept in localStorage).
        go(page, "Route planner")
        page.set_viewport_size({"width": 420, "height": 900})
        page.locator(".recent li").first.locator("button").first.click()
        page.wait_for_selector(".option", timeout=30_000)
        check(page.locator(".notice.warn", has_text="Saved result from").count() == 1, "a saved plan does not say it is saved")
        time.sleep(2)
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        check(overflow <= 0, f"page scrolls sideways at phone width by {overflow}px")
        page.screenshot(path=str(out / "phone.png"), full_page=True)

        # Leaving the planner right after a plan, several times: nothing may throw
        # (a Leaflet zoom animation still running once did), and the plan stays.
        page.set_viewport_size({"width": 1680, "height": 1000})
        for _ in range(4):
            page.locator(".recent li").first.locator("button.again").click()
            go(page, "Model evaluation")
            go(page, "Route planner")
            page.wait_for_selector(".option:visible", timeout=60_000)
        check(tiles and all(t.all_headers().get("referer") for t in tiles[:5]),
              "map tiles are requested without a Referer, so OpenStreetMap serves 'Access blocked'")
        browser.close()

    for problem in problems:
        print("FAIL:", problem)
    print(f"screenshots in {out}/" if not problems else "")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
