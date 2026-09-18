"""Exercise the served UI in sandboxed headless Chrome; no model download required.

Run with .venv/Scripts/python.exe scripts/check_overlay_browser.py --run-id ID [--vlm].
Requires playwright installed in the development environment and Google Chrome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--vlm", action="store_true")
    arguments = parser.parse_args()
    output = Path("temp")
    output.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True, chromium_sandbox=True)
        page = browser.new_page(viewport={"width": 1668, "height": 1280})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8000/")
        page.wait_for_load_state("networkidle")
        # Use a completed real run to isolate UI rendering from GPU inference.
        page.evaluate("""async (runId) => {
          const run = await (await fetch('/api/runs/' + encodeURIComponent(runId))).json();
          showResult(run);
        }""", arguments.run_id)
        ink = """() => {
          const c = document.querySelector('#overlay');
          const p = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
          let count = 0;
          for (let i = 3; i < p.length; i += 4) if (p[i]) count++;
          return count;
        }"""
        page.wait_for_function(f"({ink})() > 100")
        colored = page.evaluate(ink)
        page.screenshot(path=str(output / "overlay-browser.png"), full_page=True)
        for checkbox in page.locator('[data-visible-provider]').all():
            checkbox.uncheck()
        assert page.evaluate(ink) == 0, "Hiding providers must clear every pixel"
        page.locator('[data-visible-provider]').first.check()
        page.wait_for_function(f"({ink})() > 100")
        page.set_viewport_size({"width": 900, "height": 800})
        page.wait_for_function(f"({ink})() > 100")
        # A second run of the same image must draw even when the image is cached.
        page.evaluate("showResult(state.result)")
        page.wait_for_function(f"({ink})() > 100")
        if arguments.vlm:
            page.locator('#llm-model').select_option('qwen/qwen3-vl-4b')
            with page.expect_response('**/api/llm/review', timeout=300000) as response:
                page.locator('#llm-send').click()
            reply = response.value.json()
            assert response.value.status == 200, reply
            assert reply['parsed'], reply
            assert isinstance(json.loads(reply['content'])['pick'], list), reply['content']
            final_ink = ink.replace('#overlay', '#llm-overlay')
            page.wait_for_function(f"({final_ink})() > 100")
            assert page.locator('#llm-preview').is_visible()
            assert not page.locator('#llm-apply').is_checked()
            page.locator('#llm-preview').screenshot(path=str(output / 'vlm-final-boxes.png'))
            page.set_viewport_size({"width": 1200, "height": 900})
            page.wait_for_function(f"({final_ink})() > 100")
            page.locator('#llm-apply').check()
            if reply['refined_annotations']:
                page.wait_for_function(f"({ink})() > 100")
            (output / 'qwen-review.json').write_text(json.dumps(reply, ensure_ascii=False, indent=2), encoding='utf-8')
            page.screenshot(path=str(output / 'qwen-browser.png'), full_page=True)
            # All review status filters must clear and restore the lower overlay.
            for checkbox in page.locator('[data-review-status]').all():
                checkbox.uncheck()
            assert page.evaluate(final_ink) == 0
            for checkbox in page.locator('[data-review-status]').all():
                checkbox.check()
            page.wait_for_function(f"({final_ink})() > 100")
            # Empty valid results stay visible, and a new run clears the old preview.
            page.evaluate("state.llm.refinedAnnotations = []; state.llm.reviewAnnotations = []; showLlmPreview(true)")
            page.wait_for_function(f"({final_ink})() === 0")
            assert page.locator('#llm-preview').is_visible()
            page.evaluate("showLlmPreview(false)")
            assert page.locator('#llm-preview').is_hidden()
            page.evaluate("showLlmPreview(true); showResult(state.result)")
            assert page.locator('#llm-preview').is_hidden()
        assert not errors, errors
        print(json.dumps({"colored_pixels": colored, "page_errors": errors, "filters_resize_cached": "passed"}))
        browser.close()


if __name__ == "__main__":
    main()
