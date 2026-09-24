"""Historical text-line Mask2Former from the user's pinned manuscript-ocr checkout."""

import json
import os
import subprocess
import time
from pathlib import Path

from labelbench.contracts import Annotation, ProviderResult
from labelbench.inference_options import current_options
from labelbench.providers.base import AnnotationProvider, ProviderAvailability
from labelbench.settings import Settings


class ManuscriptMask2FormerProvider(AnnotationProvider):
    name = "manuscript_mask2former_onnx"

    def __init__(self, cfg: Settings) -> None:
        self.model_name = "manuscript-ocr/v_0_1_13/mask2former_line_v0_prev"
        self._device = cfg.device
        self._root = cfg.root_dir
        self._worker_python = cfg.root_dir / ".venv-manuscript/Scripts/python.exe"
        directory = cfg.models_dir / "manuscript/mask2former_line_v0_prev"
        self._checkpoint = directory / "mask2former_line_v0_prev.onnx"
        self._external_data = directory / "mask2former_line_v0_prev.onnx.data"
        self._onnx_checkpoint = directory / "mask2former_line_v0_prev.matmul.onnx"
        self._verification = directory / "matmul-verification.json"
        self._adapter_source = cfg.root_dir / "scripts/manuscript_runtime.py"
        self._config = directory / "mask2former_line_v0_prev.json"
        self._source = cfg.root_dir / "data/vendor/manuscript-ocr/src/manuscript/detectors/_mask2former/__init__.py"

    def availability(self) -> ProviderAvailability:
        if not self._worker_python.is_file() or not self._source.is_file():
            return ProviderAvailability(False, "Run bat/INSTALL_MANUSCRIPT.bat (source branch v_0_1_13)")
        if not all(p.is_file() for p in (self._checkpoint, self._external_data, self._config)):
            return ProviderAvailability(False, "Run bat/DOWNLOAD_MANUSCRIPT.bat (all three release artifacts)")
        try:
            ready = self._onnx_checkpoint.is_file() and json.loads(self._verification.read_text(encoding="utf-8")).get("verified")
        except (OSError, ValueError):
            ready = False
        if not ready:
            return ProviderAvailability(False, "Run bat/INSTALL_MANUSCRIPT.bat to verify the memory-efficient ONNX graph")
        return ProviderAvailability(True, "Manuscript 0.1.13 · text lines · ONNX · isolated GPU runtime")

    def _call(self, image: Path | None = None) -> dict:
        command = [str(self._worker_python), str(self._root / "scripts/manuscript_worker.py"), "--device", self._device]
        if image is None:
            command.append("--prefetch")
        else:
            command.extend(["--image", str(image.resolve())])
            options = current_options.get()
            threshold = options.confidence if options else None
            if threshold is not None:
                command.extend(["--confidence", str(threshold)])
        result = subprocess.run(command, cwd=self._root, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=600, check=False,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            raise RuntimeError(f"Manuscript Mask2Former: {(result.stderr or result.stdout)[-2500:]}")
        return json.loads(result.stdout)

    def prefetch(self) -> str:
        return self._call()["device"]

    def annotate(self, image_path: Path) -> ProviderResult:
        started = time.perf_counter()
        payload = self._call(image_path)
        return ProviderResult(provider=self.name, model=self.model_name, image_name=image_path.name,
                              image_size=payload["image_size"],
                              annotations=[Annotation.model_validate(row) for row in payload["annotations"]],
                              elapsed_seconds=time.perf_counter()-started)
