"""Abstract provider interface and visualization helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from labelbench.contracts import ProviderResult


@dataclass(frozen=True)
class ProviderAvailability:
    available: bool
    detail: str


class AnnotationProvider(ABC):
    """A lazy model integration that produces a neutral result contract."""

    name: str
    model_name: str

    @abstractmethod
    def availability(self) -> ProviderAvailability:
        """Report whether the optional runtime dependency is installed."""

    @abstractmethod
    def annotate(self, image_path: Path) -> ProviderResult:
        """Run inference on an image without writing output files."""
