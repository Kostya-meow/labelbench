"""Versioned, provider-neutral annotation contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class Annotation(BaseModel):
    """One visual candidate produced by a model."""

    id: str
    label: str
    score: float = Field(ge=0.0, le=1.0)
    bbox_xywh: list[float] = Field(min_length=4, max_length=4)
    polygon: list[list[float]] | None = None
    mask_rle: dict[str, Any] | None = None
    text: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    provider: str


class ProviderResult(BaseModel):
    """A complete result emitted by one provider for one image."""

    schema_version: str = "1.0"
    provider: str
    model: str
    image_name: str
    image_size: list[int] = Field(min_length=2, max_length=2)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    annotations: list[Annotation]
    elapsed_seconds: float = Field(ge=0.0)


class RunRequest(BaseModel):
    image_name: str
    providers: list[str] = Field(min_length=1)


class RunResult(BaseModel):
    run_id: str
    image_name: str
    image_size: list[int]
    providers: dict[str, ProviderResult]
    consensus_score: float = Field(ge=0.0, le=1.0)


class LLMMessage(BaseModel):
    """One text message preserved by the browser-side LM Studio session."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class LLMReviewRequest(BaseModel):
    """Request for a VLM review of one completed annotation run."""

    run_id: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=12000)
    providers: list[str] = Field(min_length=1, max_length=20)
    history: list[LLMMessage] = Field(default_factory=list, max_length=20)


def safe_child_path(root: Path, user_name: str) -> Path:
    """Resolve a user-supplied child name and reject directory traversal."""

    candidate = (root / user_name).resolve()
    if root.resolve() not in candidate.parents:
        raise ValueError("Path must remain inside the configured directory")
    return candidate
