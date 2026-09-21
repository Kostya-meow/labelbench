"""Export dynamic PP-OCRv6 detectors; require real polygon parity on CUDA."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(Path("data/models/huggingface").resolve()))
    import torch
    from PIL import Image

    from labelbench.evaluation import evaluate
    from labelbench.providers.ppocr6 import PPOCR6Provider, PPOCR6SmallProvider, PPOCR6TinyProvider
    from labelbench.providers.ppocr6_onnx import (
        PPOCR6ONNXProvider,
        PPOCR6SmallONNXProvider,
        PPOCR6TinyONNXProvider,
    )
    from labelbench.settings import Settings
    from labelbench.storage import write_json

    class Export(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.model(pixel_values=pixel_values).last_hidden_state

    cfg = Settings.from_environment()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for native_type, converted_type in [(PPOCR6TinyProvider, PPOCR6TinyONNXProvider),
                                       (PPOCR6SmallProvider, PPOCR6SmallONNXProvider),
                                       (PPOCR6Provider, PPOCR6ONNXProvider)]:
        native, converted = native_type(cfg), converted_type(cfg)
        native._device = converted._device = "cuda"
        processor, model, _ = native._load_model()
        with Image.open(args.image) as source:
            inputs = processor(images=source.convert("RGB"), return_tensors="pt")["pixel_values"].cuda()
        converted.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        write_json(converted.checkpoint.with_suffix(".json"), {"verified": False})
        # Dynamic dimensions preserve the native processor's aspect-ratio handling.
        torch.onnx.export(Export(model).eval(), (inputs,), str(converted.checkpoint),
                          input_names=["pixel_values"], output_names=["probability"],
                          dynamic_axes={"pixel_values": {0: "batch", 2: "height", 3: "width"},
                                        "probability": {0: "batch", 2: "out_height", 3: "out_width"}},
                          opset_version=17, dynamo=False)
        expected = native.annotate(args.image)
        actual = converted.annotate(args.image)
        report = evaluate(actual.annotations, expected.annotations, threshold=0.95)
        report.update(verified=bool(expected.annotations) and not report["fp"] and not report["fn"],
                      image=str(args.image.resolve()), runtime="CUDAExecutionProvider")
        write_json(converted.checkpoint.with_suffix(".json"), report)
        if not report["verified"]:
            raise RuntimeError(f"{converted.name}: polygon parity failed")
        logging.getLogger(__name__).warning("%s: verified %d polygons", converted.name, len(actual.annotations))


if __name__ == "__main__":
    main()
