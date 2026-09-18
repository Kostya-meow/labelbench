"""Isolated ONNX adapter for the SBB Eynollah historical textline model."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from labelbench.contracts import Annotation, ProviderResult
from labelbench.providers.base import AnnotationProvider, ProviderAvailability


class EynollahTextlineProvider(AnnotationProvider):
    """Run SBB/eynollah-textline without changing the main PyTorch environment."""

    name = "eynollah_textline"

    def __init__(self, checkpoint: Path, worker_python: Path, device: str) -> None:
        self.model_name = "SBB/eynollah-textline"
        self._checkpoint = checkpoint
        self._worker_python = worker_python
        self._device = device

    def availability(self) -> ProviderAvailability:
        if not self._worker_python.is_file():
            return ProviderAvailability(False, "Run INSTALL_GPU.bat to create .venv-eynollah")
        if not self._checkpoint.is_file():
            return ProviderAvailability(False, "Run DOWNLOAD_WEIGHTS.bat to prepare Eynollah ONNX")
        return ProviderAvailability(True, "Ready; historical text lines on isolated ONNX GPU worker")

    def prefetch(self) -> str:
        payload = self._call_worker(prefetch=True)
        return str(payload["device"])

    def annotate(self, image_path: Path) -> ProviderResult:
        started_at = time.perf_counter()
        payload = self._call_worker(image_path=image_path)
        annotations = [Annotation.model_validate(item) for item in payload["annotations"]]
        return ProviderResult(
            provider=self.name,
            model=self.model_name,
            image_name=image_path.name,
            image_size=[int(payload["image_size"][0]), int(payload["image_size"][1])],
            annotations=annotations,
            elapsed_seconds=time.perf_counter() - started_at,
        )

    def _call_worker(
        self, image_path: Path | None = None, prefetch: bool = False
    ) -> dict[str, Any]:
        command = [
            str(self._worker_python),
            "scripts/eynollah_worker.py",
            "--model",
            str(self._checkpoint),
            "--device",
            self._device,
        ]
        if prefetch:
            command.append("--prefetch")
        else:
            if image_path is None:
                raise ValueError("image_path is required for inference")
            command.extend(["--image", str(image_path)])
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Eynollah worker failed: {detail[-2000:]}")
        try:
            return json.loads(completed.stdout.strip())
        except json.JSONDecodeError as error:
            raise RuntimeError("Eynollah worker returned invalid JSON") from error
