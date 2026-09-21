"""PP-OCRv5 Server text-detection adapter using the PaddleOCR 3.x API."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from labelbench.contracts import Annotation, ProviderResult
from labelbench.inference_options import confidence
from labelbench.providers.base import AnnotationProvider, ProviderAvailability


class PPOCRProvider(AnnotationProvider):
    name = "ppocr"
    model_name = "PP-OCRv5 Server"

    def __init__(self, device: str) -> None:
        self._device = device
        self._model: Any | None = None

    def availability(self) -> ProviderAvailability:
        worker_python = self._worker_python()
        if worker_python != Path(sys.executable) and worker_python.is_file():
            return ProviderAvailability(True, "Ready on isolated Paddle GPU worker")
        if importlib.util.find_spec("paddleocr") is None:
            return ProviderAvailability(False, "Run bat/INSTALL_GPU.bat")
        return ProviderAvailability(True, "Ready on isolated Paddle GPU worker")

    @staticmethod
    def _worker_python() -> Path:
        return Path(os.getenv("LABELBENCH_OCR_PYTHON", sys.executable))

    def _load_model(self) -> Any:
        if self._model is None:
            # isort: off
            import torch  # noqa: F401  # ModelScope imports Torch; load its CPU DLLs first.
            import paddle  # noqa: F401
            # isort: on
            from paddleocr import TextDetection

            kwargs: dict[str, Any] = {
                "model_name": "PP-OCRv5_server_det",
            }
            if self._device != "auto":
                kwargs["device"] = "gpu:0" if self._device == "cuda" else "cpu"
            self._model = TextDetection(**kwargs)
        return self._model

    def _annotate_local(self, image_path: Path) -> ProviderResult:
        started_at = time.perf_counter()
        with Image.open(image_path) as image:
            size = [image.width, image.height]
        prediction = self._load_model().predict(str(image_path), box_thresh=confidence(0.6))[0]
        raw = prediction.get("res", prediction) if hasattr(prediction, "get") else prediction
        polygons = raw.get("dt_polys", [])
        scores = raw.get("dt_scores", [])
        annotations: list[Annotation] = []
        for index, polygon_value in enumerate(polygons):
            polygon = [[float(x), float(y)] for x, y in polygon_value]
            xs, ys = zip(*polygon, strict=True)
            score = float(scores[index]) if index < len(scores) else 1.0
            annotations.append(
                Annotation(
                    id=f"ppocr-{uuid.uuid4().hex[:12]}",
                    label="text",
                    score=max(0.0, min(score, 1.0)),
                    bbox_xywh=[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                    polygon=polygon,
                    provider=self.name,
                )
            )
        return ProviderResult(
            provider=self.name,
            model=self.model_name,
            image_name=image_path.name,
            image_size=size,
            annotations=annotations,
            elapsed_seconds=time.perf_counter() - started_at,
        )

    def annotate(self, image_path: Path) -> ProviderResult:
        """Run Paddle separately to avoid CUDA DLL conflicts with PyTorch."""

        output_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_file:
                output_path = Path(output_file.name)
            subprocess.run(
                [
                    str(self._worker_python()),
                    "-m",
                    "labelbench.providers.ppocr_worker",
                    "--image",
                    str(image_path),
                    "--output",
                    str(output_path),
                    "--device",
                    self._device,
                    "--confidence", str(confidence(0.6)),
                ],
                check=True,
            )
            return ProviderResult.model_validate_json(output_path.read_text(encoding="utf-8"))
        finally:
            if output_path is not None:
                output_path.unlink(missing_ok=True)

    def prefetch(self) -> None:
        subprocess.run(
            [
                str(self._worker_python()),
                "-m",
                "labelbench.providers.ppocr_worker",
                "--prefetch",
                "--device",
                self._device,
            ],
            check=True,
        )
