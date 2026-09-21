"""Check served backend controls, dataset overlays and responsive overflow."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/api/llm/models*", lambda route: route.fulfill(json={"models": []}))
        page.goto("http://127.0.0.1:8000", wait_until="domcontentloaded")
        backend = page.get_by_label("Backend ppocr6_tiny", exact=True)
        backend.select_option("ppocr6_tiny_onnx")
        assert page.locator('[data-threshold="ppocr6_tiny_onnx"]').count() == 1
        assert "ppocr6_tiny_onnx" in page.evaluate("selectProviders()")
        page.locator("#dataset-select").select_option(args.dataset)
        page.wait_for_function("document.querySelector('#image-select').options.length > 1")
        page.evaluate("""async dataset => {
          const meta = await json('/api/datasets/' + dataset);
          const image = meta.images.find(x => x.name === '1090.jpg') || meta.images[0];
          showResult({run_id:'browser-check', image_name:image.name, image_size:image.size,
            dataset_id:dataset, providers:{}, consensus_score:0});
        }""", args.dataset)
        page.wait_for_function("""() => {
          const canvas = document.querySelector('#overlay');
          if (!canvas.width || !canvas.height) return false;
          return canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data.some((v,i) => i%4 === 3 && v);
        }""")
        Path("temp").mkdir(exist_ok=True)
        page.screenshot(path="temp/workbench-lines.png", full_page=True)
        for width in [1440, 1024, 390]:
            page.set_viewport_size({"width": width, "height": 900})
            for tab in ["viewer", "experiments"]:
                page.locator(f'[data-tab="{tab}"]').click()
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), (width, tab)
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    main()
