"""SAM 2.1 automatic mask proposal adapter."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from labelbench.contracts import Annotation, ProviderResult
from labelbench.providers.base import AnnotationProvider, ProviderAvailability
from labelbench.providers.mask2former import encode_binary_mask


class SAM2Provider(AnnotationProvider):
    name = "sam2"

    def __init__(
        self, model_name: str, checkpoint: Path, points_per_side: int, device: str
    ) -> None:
        self.model_name = model_name
        self._checkpoint = checkpoint
        self._points_per_side = points_per_side
        self._device = device
        self._generator: Any | None = None

    def availability(self) -> ProviderAvailability:
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa: F401
        except ImportError:
            return ProviderAvailability(False, "Install: uv sync --extra sam2")
        return ProviderAvailability(True, "Ready; 0.9 GB checkpoint downloads on first inference")

    def _load_generator(self) -> Any:
        if self._generator is None:
            import torch
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            resolved_device = (
                "cuda" if self._device == "auto" and torch.cuda.is_available() else self._device
            )
            resolved_device = "cpu" if resolved_device == "auto" else resolved_device
            generator_options = {
                "points_per_side": self._points_per_side,
                "pred_iou_thresh": 0.80,
                "stability_score_thresh": 0.92,
                "output_mode": "binary_mask",
            }
            if self._checkpoint.is_file():
                from sam2.build_sam import build_sam2

                model = build_sam2(
                    "configs/sam2.1/sam2.1_hiera_l.yaml",
                    ckpt_path=str(self._checkpoint),
                    device=resolved_device,
                )
                self._generator = SAM2AutomaticMaskGenerator(model, **generator_options)
            else:
                self._generator = SAM2AutomaticMaskGenerator.from_pretrained(
                    self.model_name, device=resolved_device, **generator_options
                )
        return self._generator

    def annotate(self, image_path: Path) -> ProviderResult:
        started_at = time.perf_counter()
        image = np.asarray(Image.open(image_path).convert("RGB"))
        proposals = self._load_generator().generate(image)
        annotations: list[Annotation] = []
        for proposal in proposals:
            x, y, width, height = (float(value) for value in proposal["bbox"])
            mask = np.asarray(proposal["segmentation"], dtype=bool)
            annotations.append(
                Annotation(
                    id=f"sam2-{uuid.uuid4().hex[:12]}",
                    label="object_proposal",
                    score=float(proposal["predicted_iou"]),
                    bbox_xywh=[x, y, width, height],
                    mask_rle=encode_binary_mask(mask),
                    attributes={
                        "area": int(proposal["area"]),
                        "stability_score": float(proposal["stability_score"]),
                        "point_coords": proposal["point_coords"],
                    },
                    provider=self.name,
                )
            )
        return ProviderResult(
            provider=self.name,
            model=self.model_name,
            image_name=image_path.name,
            image_size=[int(image.shape[1]), int(image.shape[0])],
            annotations=annotations,
            elapsed_seconds=time.perf_counter() - started_at,
        )
