"""Provider registry; the sole extension point for model integrations."""

from __future__ import annotations

from labelbench.providers.base import AnnotationProvider
from labelbench.providers.docufcn import DocUFCNProvider
from labelbench.providers.eynollah import EynollahTextlineProvider
from labelbench.providers.mask2former import Mask2FormerProvider
from labelbench.providers.ppocr import PPOCRProvider
from labelbench.providers.ppocr6 import PPOCR6Provider, PPOCR6SmallProvider, PPOCR6TinyProvider
from labelbench.providers.rfdetr import RFDETRHistoricalProvider
from labelbench.providers.rtmdet_lines import RTMDetLinesProvider
from labelbench.providers.sam2 import SAM2Provider
from labelbench.providers.yolo26 import YOLO26Provider
from labelbench.settings import Settings


def default_registry(settings: Settings) -> dict[str, AnnotationProvider]:
    """Build all built-in providers lazily; no model weights load here."""

    return {
        "ppocr": PPOCRProvider(settings.device),
        "ppocr6": PPOCR6Provider(settings),
        "ppocr6_small": PPOCR6SmallProvider(settings),
        "ppocr6_tiny": PPOCR6TinyProvider(settings),
        "mask2former": Mask2FormerProvider(settings.mask2former_model, settings.device),
        "sam2": SAM2Provider(
            settings.sam2_model,
            settings.sam2_checkpoint,
            settings.sam2_points_per_side,
            settings.device,
        ),
        "yolo26": YOLO26Provider(settings.yolo26_model, settings.device),
        "rfdetr_historical": RFDETRHistoricalProvider(
            settings.rfdetr_checkpoint, settings.device
        ),
        "docufcn": DocUFCNProvider(
            settings.docufcn_checkpoint, settings.dla_python, settings.device
        ),
        "eynollah_textline": EynollahTextlineProvider(
            settings.eynollah_checkpoint, settings.eynollah_python, settings.device
        ),
        "rtmdet_lines": RTMDetLinesProvider(
            settings.rtmdet_checkpoint,
            settings.rtmdet_config,
            settings.rtmdet_python,
            settings.device,
        ),
    }
