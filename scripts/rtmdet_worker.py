"""Isolated RTMDet Lines worker with a TorchVision fallback for MMCV NMS."""

from __future__ import annotations

import argparse
import contextlib
import importlib.machinery
import io
import json
import sys
import types
from pathlib import Path
from typing import Any


def _install_mmcv_extension_fallback() -> None:
    """Let mmcv-lite import while routing the only used compiled op to TorchVision."""

    import torch
    from torchvision.ops import nms as torchvision_nms

    class ExtensionFallback(types.ModuleType):
        def __getattr__(self, name: str) -> Any:
            if name.startswith("__"):
                raise AttributeError(name)
            if name == "nms":
                def nms(
                    boxes: torch.Tensor,
                    scores: torch.Tensor,
                    iou_threshold: float,
                    offset: int = 0,
                ) -> torch.Tensor:
                    if offset not in {0, 1}:
                        raise ValueError("NMS offset must be 0 or 1")
                    return torchvision_nms(boxes, scores, iou_threshold)

                return nms

            def unavailable(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError(f"MMCV operation {name!r} is unavailable in mmcv-lite")

            return unavailable

    extension = ExtensionFallback("mmcv._ext")
    extension.__file__ = str(Path(__file__))
    extension.__spec__ = importlib.machinery.ModuleSpec("mmcv._ext", loader=None)
    sys.modules.setdefault("mmcv._ext", extension)


def _device(requested: str) -> str:
    import torch

    resolved = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    resolved = "cpu" if resolved == "auto" else resolved
    if resolved == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the RTMDet worker")
    return resolved


def _load_model(checkpoint: Path, config: Path, device: str) -> Any:
    _install_mmcv_extension_fallback()
    import torch
    from htrflow.models.openmmlab.rtmdet import RTMDet

    original_load = torch.load

    def load_trusted_checkpoint(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return RTMDet(model=str(checkpoint), config=str(config), device=device)
    finally:
        torch.load = original_load


def _annotations(model: Any, image_path: Path) -> list[dict[str, object]]:
    import cv2
    import numpy as np

    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    with contextlib.redirect_stdout(io.StringIO()):
        result = model.predict([image], tqdm_kwargs={"disable": True})[0]
    annotations: list[dict[str, object]] = []
    for index, segment in enumerate(result.segments):
        polygon = [[float(x), float(y)] for x, y in segment.polygon]
        if len(polygon) < 3:
            continue
        x, y, width, height = segment.bbox.xywh
        annotations.append(
            {
                "id": f"rtmdet-lines-{index}",
                "label": "text_line",
                "score": float(segment.score or 0.0),
                "bbox_xywh": [float(x), float(y), float(width), float(height)],
                "polygon": polygon,
                "attributes": {"device": str(model.device)},
                "provider": "rtmdet_lines",
            }
        )
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prefetch", action="store_true")
    arguments = parser.parse_args()

    device = _device(arguments.device)
    model = _load_model(arguments.checkpoint, arguments.config, device)
    if arguments.prefetch:
        print(json.dumps({"device": device}))
        return
    if arguments.image is None:
        raise ValueError("--image is required unless --prefetch is used")
    print(json.dumps({"device": device, "annotations": _annotations(model, arguments.image)}))


if __name__ == "__main__":
    main()
