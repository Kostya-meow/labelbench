"""Mask2Former adapter using the Transformers universal-segmentation API."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from labelbench.contracts import Annotation, ProviderResult
from labelbench.providers.base import AnnotationProvider, ProviderAvailability


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    """Encode a boolean mask as compact COCO-style uncompressed RLE."""

    pixels = mask.astype(np.uint8).flatten(order="F")
    counts: list[int] = []
    previous = 0
    run_length = 0
    for pixel in pixels:
        if int(pixel) == previous:
            run_length += 1
        else:
            counts.append(run_length)
            previous = int(pixel)
            run_length = 1
    counts.append(run_length)
    return {"size": list(mask.shape), "counts": counts}


class Mask2FormerProvider(AnnotationProvider):
    name = "mask2former"

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self._device = device
        self._processor: Any | None = None
        self._model: Any | None = None

    def availability(self) -> ProviderAvailability:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return ProviderAvailability(False, "Install: uv sync --extra vision")
        return ProviderAvailability(True, "Ready; weights download on first inference")

    def _load_model(self) -> tuple[Any, Any, str]:
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        if self._model is None or self._processor is None:
            resolved_device = "cuda" if self._device == "auto" and torch.cuda.is_available() else self._device
            resolved_device = "cpu" if resolved_device == "auto" else resolved_device
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_name)
            self._model.to(resolved_device).eval()
        return self._processor, self._model, next(self._model.parameters()).device.type

    def annotate(self, image_path: Path) -> ProviderResult:
        import torch

        started_at = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        processor, model, device = self._load_model()
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs, target_sizes=[(image.height, image.width)], threshold=0.45
        )[0]
        segmentation = result["segmentation"].cpu().numpy()
        annotations: list[Annotation] = []
        labels = model.config.id2label
        for segment in result["segments_info"]:
            segment_id = int(segment["id"])
            mask = segmentation == segment_id
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            label_id = int(segment["label_id"])
            label = labels.get(label_id, labels.get(str(label_id), str(label_id)))
            annotations.append(
                Annotation(
                    id=f"mask2former-{uuid.uuid4().hex[:12]}",
                    label=str(label),
                    score=float(segment.get("score", 1.0)),
                    bbox_xywh=[
                        float(xs.min()),
                        float(ys.min()),
                        float(np.ptp(xs) + 1),
                        float(np.ptp(ys) + 1),
                    ],
                    mask_rle=encode_binary_mask(mask),
                    attributes={"category_id": label_id, "area": int(mask.sum())},
                    provider=self.name,
                )
            )
        return ProviderResult(
            provider=self.name,
            model=self.model_name,
            image_name=image_path.name,
            image_size=[image.width, image.height],
            annotations=annotations,
            elapsed_seconds=time.perf_counter() - started_at,
        )
