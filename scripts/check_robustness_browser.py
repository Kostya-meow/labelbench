"""Validate the served dashboard, reload persistence and real polygon overlay."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    with sync_playwright() as driver:
        browser = driver.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/api/llm/models*", lambda route: route.fulfill(json={"models": []}))
        page.goto(f"http://127.0.0.1:8000/?robustness={args.id}", wait_until="domcontentloaded")
        page.wait_for_function("document.querySelector('#robust-page').options.length > 0")
        assert page.locator("#robustness-panel").is_visible()
        page.locator("#robust-open").click()
        page.wait_for_function("""() => {
          const c=document.querySelector('#robust-overlay');
          return c.width>0 && c.getContext('2d').getImageData(0,0,c.width,c.height).data.some((v,i)=>i%4===3&&v);
        }""")
        assert page.locator("#robust-metrics tbody tr").count() > 3
        Path("temp").mkdir(exist_ok=True)
        page.screenshot(path="temp/robust-dashboard.png", full_page=True)
        page.reload(wait_until="domcontentloaded")
        page.wait_for_function("document.querySelector('#robust-history').value === " + repr(args.id))
        page.wait_for_function("document.querySelectorAll('#robust-counters strong').length === 5")
        assert page.locator("#robust-counters strong").count() == 5
        page.locator("#robust-heat-field").select_option("clean_recovery")
        assert "clean_recovery" in page.locator("#robust-heatmap").inner_text()
        for width in [1440, 1024, 390]:
            page.set_viewport_size({"width": width, "height": 900})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1"), width
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    main()
