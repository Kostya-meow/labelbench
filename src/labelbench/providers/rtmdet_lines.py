"""Riksarkivet RTMDet historical text-line provider."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image

from labelbench.contracts import Annotation, ProviderResult
from labelbench.providers.base import AnnotationProvider, ProviderAvailability


class RTMDetLinesProvider(AnnotationProvider):
    """Run RTMDet Lines in an isolated OpenMMLab environment."""

    name = "rtmdet_lines"
    model_name = "Riksarkivet/rtmdet_lines"

    def __init__(
        self,
        checkpoint: Path,
        config: Path,
        worker_python: Path,
        device: str,
    ) -> None:
        self._checkpoint = checkpoint
        self._config = config
        self._worker_python = worker_python
        self._device = device

    def availability(self) -> ProviderAvailability:
        if not self._worker_python.is_file():
            return ProviderAvailability(False, "Run INSTALL_GPU.bat to create the RTMDet environment")
        if not self._checkpoint.is_file() or not self._config.is_file():
            return ProviderAvailability(False, "Run DOWNLOAD_WEIGHTS.bat to download RTMDet Lines")
        return ProviderAvailability(True, "Ready; Riksarkivet historical text-line segmentation")

    def prefetch(self) -> str:
        return str(self._call_worker(prefetch=True)["device"])

    def annotate(self, image_path: Path) -> ProviderResult:
        started_at = time.perf_counter()
        with Image.open(image_path) as image:
            image_size = [image.width, image.height]
        payload = self._call_worker(image_path=image_path)
        return ProviderResult(
            provider=self.name,
            model=self.model_name,
            image_name=image_path.name,
            image_size=image_size,
            annotations=[Annotation.model_validate(item) for item in payload["annotations"]],
            elapsed_seconds=time.perf_counter() - started_at,
        )

    def _call_worker(
        self, image_path: Path | None = None, prefetch: bool = False
    ) -> dict[str, Any]:
        command = [
            str(self._worker_python),
            "scripts/rtmdet_worker.py",
            "--checkpoint",
            str(self._checkpoint),
            "--config",
            str(self._config),
            "--device",
            self._device,
        ]
        if prefetch:
            command.append("--prefetch")
        else:
            command.extend(["--image", str(image_path)])
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=600
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"RTMDet Lines worker failed: {detail[-2000:]}")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("RTMDet Lines worker returned invalid JSON") from error
