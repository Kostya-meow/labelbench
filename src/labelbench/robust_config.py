"""Frozen, GT-independent robustness protocol."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class View(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["clean", "brightness", "contrast", "blur", "noise", "jpeg", "scale", "rotate"]
    value: float = 1

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.value:g}"


def default_views() -> list[View]:
    return [View(kind=k, value=v) for k, v in [
        ("clean", 1), ("brightness", .8), ("contrast", .75), ("blur", .8),
        ("noise", 4), ("jpeg", 65), ("scale", .8), ("rotate", -1.5), ("rotate", 1.5),
    ]]


class RobustConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(default="Line robustness and fusion", min_length=1, max_length=120)
    dataset_id: str
    providers: list[str] = Field(default_factory=lambda: [
        "ppocr6_onnx", "rfdetr_historical_onnx", "docufcn_onnx", "eynollah_textline"])
    confidence: dict[str, float] = Field(default_factory=dict)
    views: list[View] = Field(default_factory=default_views)
    images: list[str] | None = None
    include_unlabelled: bool = True
    max_side: int = Field(default=2560, ge=512, le=8192)
    group_iou: float = Field(default=.45, gt=0, le=1)
    stable_fraction: float = Field(default=.6, gt=0, le=1)
    min_families: int = Field(default=2, ge=1, le=10)
    vote_resolution: int = Field(default=512, ge=128, le=2048)
    boundary_tolerance: float = Field(default=.002, gt=0, le=.05)
    seed: int = 42
    bootstrap_samples: int = Field(default=2000, ge=100, le=10000)

    @model_validator(mode="after")
    def validate_protocol(self) -> "RobustConfig":
        if not self.providers or len(set(self.providers)) != len(self.providers):
            raise ValueError("Choose distinct providers")
        if not self.views or self.views[0].kind != "clean" or len({v.key for v in self.views}) != len(self.views):
            raise ValueError("Views must be unique and start with clean")
        if len(self.views) > 20 or len(self.providers) > 12:
            raise ValueError("At most 20 views and 12 providers")
        if set(self.confidence) - set(self.providers) or any(not 0 <= v <= 1 for v in self.confidence.values()):
            raise ValueError("Invalid provider confidence")
        for view in self.views:
            bounds = {"clean": (1, 1), "brightness": (.2, 2), "contrast": (.2, 2),
                      "blur": (0, 3), "noise": (0, 20), "jpeg": (20, 100),
                      "scale": (.5, 1.5), "rotate": (-5, 5)}[view.kind]
            if not bounds[0] <= view.value <= bounds[1]:
                raise ValueError(f"Unsafe/unsupported perturbation magnitude: {view.key}")
        return self


def family(provider: str) -> str:
    if provider.startswith("ppocr"):
        return "paddle_ocr"
    return provider.removesuffix("_onnx")
