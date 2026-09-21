"""Isolated PP-OCR worker used to keep Paddle CUDA DLLs out of the API process."""

from __future__ import annotations

import argparse
from pathlib import Path

from labelbench.inference_options import ProviderOptions, current_options
from labelbench.providers.ppocr import PPOCRProvider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.6)
    arguments = parser.parse_args()

    provider = PPOCRProvider(arguments.device)
    provider._load_model()
    if arguments.prefetch:
        import paddle

        device = paddle.device.get_device()
        if arguments.device == "cuda" and not device.startswith("gpu"):
            raise RuntimeError(f"PP-OCR loaded on {device}, expected GPU")
        print(f"ppocr: ready on {device}")
        return
    if arguments.image is None or arguments.output is None:
        parser.error("--image and --output are required unless --prefetch is used")
    context = current_options.set(ProviderOptions(confidence=arguments.confidence))
    try:
        result = provider._annotate_local(arguments.image)
    finally:
        current_options.reset(context)
    arguments.output.write_text(result.model_dump_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
