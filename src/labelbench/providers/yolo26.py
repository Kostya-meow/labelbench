"""Ultralytics YOLO26 instance-segmentation adapter."""

from __future__ import annotations

import importlib.util
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from labelbench.contracts import Annotation, ProviderResult
from labelbench.providers.base import AnnotationProvider, ProviderAvailability


class YOLO26Provider(AnnotationProvider):
    """Run a pretrained YOLO26 segmentation checkpoint on an image."""

    name = "yolo26"

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self._device = device
        self._model: Any | None = None

    def availability(self) -> ProviderAvailability:
        if importlib.util.find_spec("ultralytics") is None:
            return ProviderAvailability(False, "Install: uv pip install ultralytics")
        return ProviderAvailability(True, f"Ready; {self.model_name} loads on first inference")

    def _load_model(self) -> tuple[Any, str]:
        import torch
        from ultralytics import YOLO

        if self._model is None:
            resolved_device = (
                "cuda" if self._device == "auto" and torch.cuda.is_available() else self._device
            )
            resolved_device = "cpu" if resolved_device == "auto" else resolved_device
            self._model = YOLO(self.model_name)
            self._model.to(resolved_device)
        device = next(self._model.model.parameters()).device.type
        return self._model, device

    def prefetch(self) -> str:
        """Download the checkpoint and verify that it is placed on the requested device."""

        _, device = self._load_model()
        return device

    def annotate(self, image_path: Path) -> ProviderResult:
        started_at = time.perf_counter()
        with Image.open(image_path) as image:
            image_size = [image.width, image.height]
        model, device = self._load_model()
        results = model.predict(
            source=str(image_path),
            device=device,
            conf=0.25,
            verbose=False,
            save=False,
        )
        result = results[0]
        annotations: list[Annotation] = []
        boxes = result.boxes
        polygons = result.masks.xy if result.masks is not None else []
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().tolist()
            scores = boxes.conf.detach().cpu().tolist()
            class_ids = boxes.cls.detach().cpu().tolist()
            for index, (coordinates, score, raw_class_id) in enumerate(
                zip(xyxy, scores, class_ids)
            ):
                x1, y1, x2, y2 = (float(value) for value in coordinates)
                class_id = int(raw_class_id)
                label = result.names.get(class_id, str(class_id))
                polygon = self._polygon(polygons[index]) if index < len(polygons) else None
                annotations.append(
                    Annotation(
                        id=f"yolo26-{uuid.uuid4().hex[:12]}",
                        label=str(label),
                        score=float(score),
                        bbox_xywh=[x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        polygon=polygon,
                        attributes={"class_id": class_id, "device": device},
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
    def _polygon(points: Any) -> list[list[float]] | None:
        values = points.tolist()
        if len(values) < 3:
            return None
        return [[float(point[0]), float(point[1])] for point in values]
