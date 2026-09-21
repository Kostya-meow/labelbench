"""Verified ONNX Doc-UFCN; retain the official tiling and contour pipeline."""

import json

from labelbench.providers.base import ProviderAvailability
from labelbench.providers.docufcn import DocUFCNProvider
from labelbench.settings import Settings


class DocUFCNONNXProvider(DocUFCNProvider):
    name = "docufcn_onnx"

    def __init__(self, cfg: Settings) -> None:
        super().__init__(cfg.docufcn_checkpoint, cfg.dla_python, cfg.device)
        self._onnx_checkpoint = cfg.models_dir / "onnx" / "docufcn.onnx"

    def availability(self) -> ProviderAvailability:
        try:
            report = json.loads(self._onnx_checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
            if self._onnx_checkpoint.is_file() and report.get("verified"):
                return super().availability()
        except (OSError, ValueError):
            pass
        return ProviderAvailability(False, "Run bat/EXPORT_DOCUFCN_ONNX.bat")
