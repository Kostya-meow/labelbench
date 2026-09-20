"""Exercise the served UI in sandboxed headless Chrome; no model download required.

Run with .venv/Scripts/python.exe scripts/check_overlay_browser.py --run-id ID [--vlm].
Requires playwright installed in the development environment and Google Chrome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import Page, Route, sync_playwright


def check_vlm_failure_states(page: Page) -> None:
    """Exercise network/partial failures without asking the real model again."""
    page.evaluate("state.llm.model = 'test-vlm'; updateLlmEnabled()")
    endpoint = '**/api/llm/review'
    page.route(endpoint, lambda route: route.fulfill(status=503, body='Internal Server Error'))
    page.locator('#llm-send').click()
    page.wait_for_function("document.querySelector('#llm-status').textContent.includes('Internal Server Error')")
    assert page.locator('#llm-preview').is_hidden()
    assert page.locator('#llm-apply').is_disabled()
    assert page.locator('#llm-send').is_enabled()
    assert 'Internal Server Error' in page.locator('#llm-console').inner_text()
    page.unroute(endpoint)
    reply = page.evaluate("""() => ({
      model: 'test-vlm', parsed: true, complete: false, batch_count: 2,
      content: '{"uncertain":[0]}', notes: 'Partial test response',
      refined_annotations: [],
      review_annotations: Object.values(state.result.providers)[0].annotations.slice(0, 1)
        .map(a => ({...a, attributes: {...a.attributes, review_status: 'uncertain'}})),
    })""")
    page.route(endpoint, lambda route: route.fulfill(json=reply))
    page.locator('#llm-send').click()
    page.wait_for_function("document.querySelector('#llm-status').textContent.includes('Частичный ответ')")
    assert page.locator('#llm-preview').is_visible()
    assert not page.locator('#llm-apply').is_checked()
    page.unroute(endpoint)

    def stale_response(route: Route) -> None:
        page.evaluate("updateLlmEnabled()")
        assert page.locator('#llm-send').is_disabled()
        page.evaluate("showResult({...state.result})")
        route.fulfill(json=reply)

    page.route(endpoint, stale_response)
    page.locator('#llm-send').click()
    page.wait_for_function("document.querySelector('#llm-status').textContent.includes('Открыт другой результат')")
    assert page.locator('#llm-preview').is_hidden()
    assert page.locator('#llm-apply').is_disabled()
    page.unroute(endpoint)


def check_routerai_controls(page: Page) -> None:
    """Test both modes and export using an intercepted remote response."""
    endpoint = '**/api/llm/review'
    page.locator('#llm-backend').select_option('routerai')
    assert page.locator('#routerai-key').is_visible()
    assert page.locator('#llm-model').is_hidden()
    assert page.locator('#llm-send').is_disabled()
    page.locator('#routerai-model').fill('deepseek/deepseek-v4-pro-0813')
    page.locator('#routerai-key').fill('test-key-placeholder')
    page.locator('#llm-max-tokens').fill('8192')
    captured: list[dict] = []
    fixture = Path('temp/routerai-review.json')
    reply = json.loads(fixture.read_text(encoding='utf-8')) if fixture.is_file() else {
        'parsed': True, 'complete': True, 'content': '{}', 'refined_annotations': [],
        'review_annotations': [], 'model': 'deepseek/deepseek-v4-pro-0813',
    }

    def intercept(route: Route) -> None:
        body = route.request.post_data_json
        captured.append(body)
        route.fulfill(json={**reply, 'backend': 'routerai', 'request_mode': body['request_mode']})

    page.route(endpoint, intercept)
    for mode in ('batched', 'all'):
        page.locator('#llm-mode').select_option(mode)
        page.locator('#llm-send').click()
        page.wait_for_function("!state.llm.busy && state.llm.resultBackend === 'routerai'")
        assert page.locator('#llm-preview').is_visible()
        assert captured[-1]['request_mode'] == mode
        assert captured[-1]['api_key'] == 'test-key-placeholder'
        assert captured[-1]['model'] == 'deepseek/deepseek-v4-pro-0813'
    page.locator('#llm-preview').screenshot(path='temp/routerai-preview.png')
    with page.expect_download() as download:
        page.locator('#llm-export').click()
    exported = json.loads(Path(download.value.path()).read_text(encoding='utf-8'))
    assert exported['source'] == 'routerai' and exported['request_mode'] == 'all'
    assert 'test-key-placeholder' not in json.dumps(exported)
    assert page.evaluate("!Object.values(localStorage).some(v => v.includes('test-key-placeholder'))")
    page.locator('#routerai-key').fill('')
    page.locator('#llm-backend').select_option('lm_studio')
    assert page.locator('#routerai-key').is_hidden()
    assert page.locator('#llm-model').is_visible()
    page.locator('#llm-mode').select_option('batched')
    page.unroute(endpoint)


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
        if page.locator('#llm-model option[value="qwen/qwen3-vl-4b"]').count():
            assert page.locator('#llm-model').input_value() == 'qwen/qwen3-vl-4b'
        # Use a completed real run to isolate UI rendering from GPU inference.
        page.evaluate("""async (runId) => {
          const response = await fetch('/api/runs/' + encodeURIComponent(runId));
          if (!response.ok) throw new Error('Run unavailable: ' + runId);
          const run = await response.json();
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
            for checkbox in page.locator('[data-visible-provider]').all():
                checkbox.check()
            page.locator('#llm-model').select_option('qwen/qwen3-vl-4b')
            with page.expect_response('**/api/llm/review', timeout=650000) as response:
                page.locator('#llm-send').click()
            reply = response.value.json()
            (output / 'qwen-review.json').write_text(json.dumps(reply, ensure_ascii=False, indent=2), encoding='utf-8')
            assert response.value.status == 200, reply
            assert reply['parsed'], reply
            assert reply['complete'], reply['notes']
            assert reply['invalid_decisions'] == 0, reply['notes']
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
        check_vlm_failure_states(page)
        check_routerai_controls(page)
        assert not errors, errors
        print(json.dumps({
            "colored_pixels": colored, "page_errors": errors, "filters_resize_cached": "passed",
            "vlm": {key: reply[key] for key in ("model", "batch_count", "complete", "invalid_decisions", "usage")}
            if arguments.vlm else None,
        }))
        browser.close()


if __name__ == "__main__":
    main()
