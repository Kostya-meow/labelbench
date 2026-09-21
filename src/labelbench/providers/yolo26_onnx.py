"""Optional verified YOLO26 ONNX backend, sharing the segmentation contract."""

import importlib.util
import json

from labelbench.providers.base import ProviderAvailability
from labelbench.providers.yolo26 import YOLO26Provider
from labelbench.settings import Settings


class YOLO26ONNXProvider(YOLO26Provider):
    name = "yolo26_onnx"

    def __init__(self, cfg: Settings) -> None:
        super().__init__(str(cfg.models_dir / "onnx" / "yolo26n-seg.onnx"), cfg.device)

    def availability(self) -> ProviderAvailability:
        from pathlib import Path

        if not Path(self.model_name).is_file() or importlib.util.find_spec("onnxruntime") is None:
            return ProviderAvailability(False, "Run bat/EXPORT_YOLO_ONNX.bat")
        try:
            report = json.loads(Path(self.model_name).with_name("verification.json").read_text(encoding="utf-8"))
            if not report.get("verified"):
                return ProviderAvailability(False, "ONNX parity verification is incomplete")
        except (OSError, ValueError):
            return ProviderAvailability(False, "Run ONNX export and verification first")
        return super().availability()
