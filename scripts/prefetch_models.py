"""Download and initialise requested provider weights without processing images."""

from __future__ import annotations

import argparse
import subprocess
import sys

from labelbench.registry import default_registry
from labelbench.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--providers", nargs="+", choices=[
            "ppocr", "ppocr6", "mask2former", "sam2", "yolo26", "rfdetr_historical", "docufcn",
            "eynollah_textline"
        ], required=True
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if not arguments.worker:
        for name in arguments.providers:
            subprocess.run(
                [sys.executable, __file__, "--providers", name, "--worker"], check=True
            )
        return

    settings = Settings.from_environment()
    providers = default_registry(settings)
    for name in arguments.providers:
        provider = providers[name]
        available = provider.availability()
        if not available.available:
            raise RuntimeError(f"{name}: {available.detail}")
        print(f"Downloading/loading {name} ...")
        if name == "ppocr":
            provider.prefetch()  # type: ignore[attr-defined]
            device = "gpu:0 (isolated worker)"
        elif name == "mask2former":
            _, _, device = provider._load_model()  # type: ignore[attr-defined]
            if settings.device == "cuda" and device != "cuda":
                raise RuntimeError(f"Mask2Former loaded on {device}, expected GPU")
        elif name == "sam2":
            generator = provider._load_generator()  # type: ignore[attr-defined]
            device = next(generator.predictor.model.parameters()).device.type
            if settings.device == "cuda" and device != "cuda":
                raise RuntimeError(f"SAM2 loaded on {device}, expected GPU")
        else:
            if name == "rfdetr_historical":
                device = provider.prefetch()  # type: ignore[attr-defined]
                if settings.device == "cuda" and device != "cuda":
                    raise RuntimeError(f"RF-DETR historical loaded on {device}, expected GPU")
            elif name == "docufcn":
                device = provider.prefetch()  # type: ignore[attr-defined]
                if settings.device == "cuda" and device != "cuda":
                    raise RuntimeError(f"Doc-UFCN loaded on {device}, expected GPU")
            elif name == "eynollah_textline":
                device = provider.prefetch()  # type: ignore[attr-defined]
                if settings.device == "cuda" and device != "cuda":
                    raise RuntimeError(f"Eynollah loaded on {device}, expected GPU")
            else:
                device = provider.prefetch()  # type: ignore[attr-defined]
                if settings.device == "cuda" and device != "cuda":
                    raise RuntimeError(f"{name} loaded on {device}, expected GPU")
        print(f"{name}: ready on {device}")


if __name__ == "__main__":
    main()
