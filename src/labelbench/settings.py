"""Immutable application settings loaded from environment variables."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Settings:
    """Runtime locations and model selections."""

    root_dir: Path
    images_dir: Path
    output_dir: Path
    models_dir: Path
    device: Literal["auto", "cpu", "cuda"]
    mask2former_model: str
    sam2_model: str
    sam2_checkpoint: Path
    sam2_points_per_side: int
    yolo26_model: str
    rfdetr_checkpoint: Path
    docufcn_checkpoint: Path
    dla_python: Path
    eynollah_checkpoint: Path
    eynollah_python: Path

    @classmethod
    def from_environment(cls) -> "Settings":
        import os

        root_dir = Path(os.getenv("LABELBENCH_ROOT", Path.cwd())).resolve()
        device = os.getenv("LABELBENCH_DEVICE", "auto")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("LABELBENCH_DEVICE must be auto, cpu, or cuda")
        return cls(
            root_dir=root_dir,
            images_dir=root_dir / "data" / "images",
            output_dir=root_dir / "data" / "output",
            models_dir=root_dir / "data" / "models",
            device=device,
            mask2former_model=os.getenv(
                "LABELBENCH_MASK2FORMER_MODEL", "facebook/mask2former-swin-large-coco-panoptic"
            ),
            sam2_model=os.getenv("LABELBENCH_SAM2_MODEL", "facebook/sam2.1-hiera-large"),
            sam2_checkpoint=Path(
                os.getenv(
                    "LABELBENCH_SAM2_CHECKPOINT",
                    root_dir / "data" / "models" / "sam2.1_hiera_large.pt",
                )
            ).resolve(),
            sam2_points_per_side=int(os.getenv("LABELBENCH_SAM2_POINTS_PER_SIDE", "24")),
            yolo26_model=os.getenv("LABELBENCH_YOLO26_MODEL", "yolo26n-seg.pt"),
            rfdetr_checkpoint=Path(
                os.getenv(
                    "LABELBENCH_RFDETR_CHECKPOINT",
                    root_dir / "data" / "models" / "rfdetr" / "rfdetr_text_seg_model_202510.pth.complete",
                )
            ).resolve(),
            docufcn_checkpoint=Path(
                os.getenv(
                    "LABELBENCH_DOCUFCN_CHECKPOINT",
                    root_dir / "data" / "models" / "docufcn" / "generic_historical_line_model.pth",
                )
            ).resolve(),
            dla_python=Path(
                os.getenv("LABELBENCH_DLA_PYTHON", root_dir / ".venv-dla" / "Scripts" / "python.exe")
            ).resolve(),
            eynollah_checkpoint=Path(
                os.getenv(
                    "LABELBENCH_EYNOLLAH_CHECKPOINT",
                    root_dir / "data" / "models" / "eynollah" / "eynollah_textline.onnx",
                )
            ).resolve(),
            eynollah_python=Path(
                os.getenv(
                    "LABELBENCH_EYNOLLAH_PYTHON",
                    root_dir / ".venv-eynollah" / "Scripts" / "python.exe",
                )
            ).resolve(),
        )

    def ensure_directories(self) -> None:
        for directory in (self.images_dir, self.output_dir, self.models_dir):
            directory.mkdir(parents=True, exist_ok=True)
