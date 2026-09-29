"""Accessibility scan of the dashboard with axe-core.

Loads the running app (backend on :8000, Vite on :5173), plans a route, and runs
axe-core on the route planner, the model evaluation and the supplier view.
Prints every violation and exits non-zero if there is any. axe-core is fetched
from cdnjs, so this needs internet access.

Usage: python tools/a11y_check.py [--browser msedge|chrome|chromium]
"""
import argparse
import sys
import time

from playwright.sync_api import sync_playwright

URL = "http://localhost:5173/"
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
    args = parser.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=None if args.browser == "chromium" else args.browser, headless=True)
        page = browser.new_page(viewport={"width": 1680, "height": 1000})
        page.goto(URL)
        page.wait_for_selector(".route-map .leaflet-tile-loaded", timeout=20_000)
        page.click("text=Shanghai to Rotterdam")
        page.wait_for_selector(".option", timeout=30_000)
        time.sleep(2)
        found = scan(page, "route planner")
        page.click("text=Model evaluation")
        page.wait_for_selector(".scores", timeout=15_000)
        found += scan(page, "model evaluation")
        page.click("text=Route planner")
        page.click("text=Supplier intelligence")
        page.wait_for_selector(".supplier-table tbody tr", timeout=15_000)
        found += scan(page, "supplier intelligence")
        browser.close()
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
