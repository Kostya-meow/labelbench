"""RF-DETR ONNX network with the upstream segmentation postprocessor."""

import json
from typing import Any

from labelbench.providers.base import ProviderAvailability
from labelbench.providers.rfdetr import RFDETRHistoricalProvider
from labelbench.settings import Settings


class RFDETRONNXProvider(RFDETRHistoricalProvider):
    name = "rfdetr_historical_onnx"

    def __init__(self, cfg: Settings) -> None:
        super().__init__(cfg.rfdetr_checkpoint, cfg.device)
        self._onnx_checkpoint = cfg.models_dir / "onnx" / "rfdetr_historical.onnx"

    def availability(self) -> ProviderAvailability:
        try:
            report = json.loads(self._onnx_checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
            if self._onnx_checkpoint.is_file() and report.get("verified"):
                return super().availability()
        except (OSError, ValueError):
            pass
        return ProviderAvailability(False, "Run bat/EXPORT_RFDETR_ONNX.bat")

    def _load_model(self) -> tuple[Any, str]:
        import onnxruntime as ort
        import torch
        from rfdetr import RFDETRSegPreview

        if self._model is None:
            device = "cuda" if self._device == "auto" and torch.cuda.is_available() else self._device
            device = "cpu" if device == "auto" else device
            providers = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(self._onnx_checkpoint), providers=providers)
            if device == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("RF-DETR ONNX CUDA unavailable")

            class Network(torch.nn.Module):
                def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
                    values = session.run(["dets", "labels", "masks"], {"input": tensor.cpu().numpy()})
                    return tuple(torch.from_numpy(value) for value in values)

            # Initialize upstream metadata/decoder; release the native network before inference.
            self._model = RFDETRSegPreview(pretrain_weights=str(self._checkpoint), device="cpu", num_classes=2)
            self._model.model.model = Network().eval()
            self._runtime_device = device
        return self._model, self._runtime_device
