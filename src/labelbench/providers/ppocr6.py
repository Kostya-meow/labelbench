"""PP-OCRv6 safetensors detector running in the existing PyTorch environment."""

from __future__ import annotations

import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from PIL import Image

from labelbench.contracts import Annotation, ProviderResult
from labelbench.inference_options import confidence
from labelbench.providers.base import AnnotationProvider, ProviderAvailability
from labelbench.settings import Settings


class PPOCR6Provider(AnnotationProvider):
    name = "ppocr6"
    model_setting = "ppocr6_model"
    architecture = "PPOCRV6MediumDetForObjectDetection"

    def __init__(self, cfg: Settings) -> None:
        self.model_name = getattr(cfg, self.model_setting)
        self._device = cfg.device
        self._model: Any = None
        self._processor: Any = None

    def availability(self) -> ProviderAvailability:
        specification = find_spec("transformers")
        module = "pp_ocrv6_medium_det" if "Medium" in self.architecture else "pp_ocrv6_small_det"
        if not specification or not specification.origin or not find_spec("cv2"):
            return ProviderAvailability(False, "Run bat/INSTALL_PPOCR6.bat")
        if not (Path(specification.origin).parent / "models" / module).is_dir():
            return ProviderAvailability(False, "Run bat/INSTALL_PPOCR6.bat")
        return ProviderAvailability(True, "Ready; PP-OCRv6 text detection, no recognition")

    def _load_model(self) -> tuple[Any, Any, str]:
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        if self._model is None:
            device = self._device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForObjectDetection.from_pretrained(self.model_name)
            self._model.to(device).eval()
        return self._processor, self._model, next(self._model.parameters()).device.type

    def prefetch(self) -> str:
        return self._load_model()[2]

    def annotate(self, image_path: Path) -> ProviderResult:
        import torch

        started = time.perf_counter()
        processor, model, device = self._load_model()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        # target_sizes is preprocessing metadata, not a model input.
        inputs.pop("target_sizes", None)
        with torch.inference_mode():
            output = model(**inputs)
        result = processor.post_process_object_detection(
            output, target_sizes=torch.tensor([[image.height, image.width]]),
            box_threshold=confidence(0.6),
        )[0]
        annotations = []
        for index, (points, score) in enumerate(zip(result["boxes"], result["scores"], strict=True)):
            polygon = points.detach().cpu().tolist()
            xs, ys = zip(*polygon, strict=True)
            annotations.append(Annotation(
                id=f"{self.name}-{index}", label="text", score=float(score),
                bbox_xywh=[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                polygon=polygon, provider=self.name, attributes={"device": device},
            ))
        return ProviderResult(
            provider=self.name, model=self.model_name, image_name=image_path.name,
            image_size=[image.width, image.height], annotations=annotations,
            elapsed_seconds=time.perf_counter() - started,
        )


class PPOCR6SmallProvider(PPOCR6Provider):
    name = "ppocr6_small"
    model_setting = "ppocr6_small_model"
    architecture = "PPOCRV6SmallDetForObjectDetection"


class PPOCR6TinyProvider(PPOCR6SmallProvider):
    name = "ppocr6_tiny"
    model_setting = "ppocr6_tiny_model"
