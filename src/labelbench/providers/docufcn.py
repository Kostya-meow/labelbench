"""Doc-UFCN historical text-line adapter using an isolated worker environment."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image

from labelbench.contracts import Annotation, ProviderResult
from labelbench.providers.base import AnnotationProvider, ProviderAvailability


class DocUFCNProvider(AnnotationProvider):
    """Run Teklia's generic historical line model without touching the main Torch env."""

    name = "docufcn"
    model_name = "Teklia/doc-ufcn-generic-historical-line"

    def __init__(self, checkpoint: Path, worker_python: Path, device: str) -> None:
        self._checkpoint = checkpoint
        self._worker_python = worker_python
        self._device = device

    def availability(self) -> ProviderAvailability:
        if not self._worker_python.is_file():
            return ProviderAvailability(False, "Run INSTALL_GPU.bat to create the Doc-UFCN environment")
        if not self._checkpoint.is_file():
            return ProviderAvailability(False, "Run DOWNLOAD_WEIGHTS.bat to download Doc-UFCN weights")
        return ProviderAvailability(True, "Ready; generic historical text-line model")

    def prefetch(self) -> str:
        result = self._call_worker(prefetch=True)
        return str(result["device"])

    def annotate(self, image_path: Path) -> ProviderResult:
        started_at = time.perf_counter()
        with Image.open(image_path) as image:
            image_size = [image.width, image.height]
        payload = self._call_worker(image_path=image_path)
        annotations = [Annotation.model_validate(item) for item in payload["annotations"]]
        return ProviderResult(
            provider=self.name,
            model=self.model_name,
            image_name=image_path.name,
            image_size=image_size,
            annotations=annotations,
            elapsed_seconds=time.perf_counter() - started_at,
        )

    def _call_worker(
        self, image_path: Path | None = None, prefetch: bool = False
    ) -> dict[str, Any]:
        command = [
            str(self._worker_python),
            "scripts/docufcn_worker.py",
            "--checkpoint",
            str(self._checkpoint),
            "--device",
            self._device,
        ]
        if prefetch:
            command.append("--prefetch")
        else:
            command.extend(["--image", str(image_path)])
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Doc-UFCN worker failed: {detail[-1200:]}")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("Doc-UFCN worker returned invalid JSON") from error
