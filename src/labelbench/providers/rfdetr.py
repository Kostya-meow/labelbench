"""RF-DETR historical segmentation adapter exposing text lines only."""

from __future__ import annotations

import importlib.util
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from labelbench.contracts import Annotation, ProviderResult
from labelbench.providers.base import AnnotationProvider, ProviderAvailability
from labelbench.providers.mask2former import encode_binary_mask


class RFDETRHistoricalProvider(AnnotationProvider):
    """Run the public Kansallisarkisto RF-DETR historical segmentation checkpoint."""

    name = "rfdetr_historical"

    def __init__(self, checkpoint: Path, device: str) -> None:
        self.model_name = "Kansallisarkisto/rfdetr_textline_textregion_detection_model"
        self._checkpoint = checkpoint
        self._device = device
        self._model: Any | None = None

    def availability(self) -> ProviderAvailability:
        if importlib.util.find_spec("rfdetr") is None:
            return ProviderAvailability(False, "Install: uv pip install rfdetr supervision")
        if not self._checkpoint.is_file():
            return ProviderAvailability(False, "Run DOWNLOAD_WEIGHTS.bat to download RF-DETR weights")
        return ProviderAvailability(True, "Ready; historical text lines only")

    def _load_model(self) -> tuple[Any, str]:
        import torch
        from rfdetr import RFDETRSegPreview

        if self._model is None:
            resolved_device = "cuda" if self._device == "auto" and torch.cuda.is_available() else self._device
            resolved_device = "cpu" if resolved_device == "auto" else resolved_device
            self._model = RFDETRSegPreview(
                pretrain_weights=str(self._checkpoint), device=resolved_device, num_classes=2
            )
            if resolved_device == "cuda":
                self._model.inference(compile=False, inplace=True, dtype="float16")
        if getattr(self._model, "_is_optimized_for_inference", False):
            module = self._model.model.inference_model
        else:
            module = self._model.model.model
        device = next(module.parameters()).device.type
        return self._model, device

    def prefetch(self) -> str:
        """Load the checkpoint and put the inference model on the requested device."""

        _, device = self._load_model()
        return device

    def annotate(self, image_path: Path) -> ProviderResult:
        started_at = time.perf_counter()
        with Image.open(image_path) as image:
            image_size = [image.width, image.height]
        model, device = self._load_model()
        detections = model.predict(str(image_path), threshold=0.25)
        annotations: list[Annotation] = []
        masks = detections.mask
        for index, (box, score, class_id) in enumerate(
            zip(detections.xyxy, detections.confidence, detections.class_id)
        ):
            # The checkpoint uses class 1 for regions and class 2 for text lines.
            # Drop other classes before extracting or encoding their masks.
            if int(class_id) != 2:
                continue
            mask = np.asarray(masks[index], dtype=bool) if masks is not None else None
            polygon = self._polygon(mask) if mask is not None else None
            x1, y1, x2, y2 = (float(value) for value in box)
            annotations.append(
                Annotation(
                    id=f"rfdetr-historical-{uuid.uuid4().hex[:12]}",
                    label="text_line",
                    score=float(score),
                    bbox_xywh=[x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    polygon=polygon,
                    mask_rle=encode_binary_mask(mask) if mask is not None else None,
                    attributes={"class_id": int(class_id), "device": device},
                    provider=self.name,
                )
            )
        return ProviderResult(
            provider=self.name,
            model=self.model_name,
            image_name=image_path.name,
            image_size=image_size,
            annotations=annotations,
            elapsed_seconds=time.perf_counter() - started_at,
        )

    @staticmethod
    def _polygon(mask: np.ndarray) -> list[list[float]] | None:
        import cv2

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        if len(contour) < 3:
            return None
        return [[float(point[0]), float(point[1])] for point in contour]
