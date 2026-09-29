"""Accessibility scan of the dashboard with axe-core.

Loads the running app (backend on :8000, Vite on :5173), plans a route, and runs
axe-core on the route planner (also with place suggestions open and with stale
results), the model evaluation and the supplier view (also with an invalid count).
Prints every violation and exits non-zero if there is any. axe-core is fetched
from cdnjs, so this needs internet access.

Usage: python tools/a11y_check.py [--browser msedge|chrome|chromium] [--url http://127.0.0.1:8000/]
"""
import argparse
import sys
import time

from playwright.sync_api import sync_playwright

AXE = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js"
VIOLATIONS = """async () => (await axe.run(document, {resultTypes: ['violations']})).violations
  .map(v => ({id: v.id, impact: v.impact, help: v.help, targets: v.nodes.map(n => n.target.join(' '))}))"""


def scan(page, name):
    page.add_script_tag(url=AXE)
    violations = page.evaluate(VIOLATIONS)
    print(f"{name}: {len(violations)} violations")
    for v in violations:
        print(f"  [{v['impact']}] {v['id']}: {v['help']} ({', '.join(v['targets'][:3])})")
    return len(violations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", default="msedge")
    parser.add_argument("--url", default="http://localhost:5173/")
    args = parser.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=None if args.browser == "chromium" else args.browser, headless=True)
        # The page's Content-Security-Policy (when run.py serves it) would block axe from cdnjs.
        page = browser.new_page(viewport={"width": 1680, "height": 1000}, bypass_csp=True)
        page.goto(args.url)
        page.wait_for_selector(".route-map .leaflet-tile-loaded", timeout=20_000)
        page.click("text=Shanghai to Rotterdam")
        page.wait_for_selector(".option", timeout=30_000)
        time.sleep(2)
        found = scan(page, "route planner")
        page.fill("#hub-dest", "Hamb")
        page.wait_for_selector(".suggestions [role=option]", timeout=10_000)
        page.press("#hub-dest", "ArrowDown")
        found += scan(page, "route planner, suggestions open, results stale")
        page.press("#hub-dest", "Escape")
        page.locator("button.nav-button:visible", has_text="Model evaluation").click()
        page.wait_for_selector(".scores", timeout=15_000)
        found += scan(page, "model evaluation")
        page.locator("button.nav-button:visible", has_text="Route planner").click()
        page.locator("button.nav-button:visible", has_text="Supplier intelligence").click()
        page.wait_for_selector(".supplier-table tbody tr", timeout=15_000)
        found += scan(page, "supplier intelligence")
        page.fill("#inventory", "-5")
        page.wait_for_selector("#inventory-error", timeout=5_000)
        found += scan(page, "supplier intelligence, invalid count")
        browser.close()
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
