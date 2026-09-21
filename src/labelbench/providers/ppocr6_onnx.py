"""PP-OCRv6 ONNX inference with the native processor and polygon decoder."""

from __future__ import annotations

import json
from typing import Any

from labelbench.providers.base import ProviderAvailability
from labelbench.providers.ppocr6 import PPOCR6Provider
from labelbench.settings import Settings


class PPOCR6ONNXProvider(PPOCR6Provider):
    name = "ppocr6_onnx"

    def __init__(self, cfg: Settings) -> None:
        super().__init__(cfg)
        self.checkpoint = cfg.models_dir / "onnx" / f"{self.name}.onnx"

    def availability(self) -> ProviderAvailability:
        try:
            import onnxruntime  # noqa: F401

            report = json.loads(self.checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
            if self.checkpoint.is_file() and report.get("verified"):
                return ProviderAvailability(True, "ONNX; CUDA parity verified")
        except (ImportError, OSError, ValueError):
            pass
        return ProviderAvailability(False, "Run bat/EXPORT_PPOCR6_ONNX.bat")

    def _load_model(self) -> tuple[Any, Any, str]:
        import onnxruntime as ort
        import torch
        from transformers import AutoImageProcessor

        if self._model is None:
            device = self._device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            providers = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(self.checkpoint), providers=providers)
            if device == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("ONNX CUDA unavailable; CPU fallback refused")
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = session
            self._runtime_device = device

        # The common provider expects a Torch-like callable and device parameter.
        from types import SimpleNamespace

        session = self._model

        class Adapter:
            def parameters(adapter_self: Any) -> Any:
                return iter([torch.empty(0, device=self._runtime_device)])

            def __call__(adapter_self: Any, pixel_values: Any, **kwargs: Any) -> Any:
                values = session.run(None, {"pixel_values": pixel_values.detach().cpu().numpy()})[0]
                return SimpleNamespace(last_hidden_state=torch.from_numpy(values))

        return self._processor, Adapter(), self._runtime_device


class PPOCR6SmallONNXProvider(PPOCR6ONNXProvider):
    name = "ppocr6_small_onnx"
    model_setting = "ppocr6_small_model"


class PPOCR6TinyONNXProvider(PPOCR6ONNXProvider):
    name = "ppocr6_tiny_onnx"
    model_setting = "ppocr6_tiny_model"
