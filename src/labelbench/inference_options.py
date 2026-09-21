"""Request-local thresholds; no mutation of shared model configuration."""

from contextvars import ContextVar

from pydantic import BaseModel, ConfigDict, Field


class ProviderOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    confidence: float | None = Field(default=None, ge=0, le=1)
    labels: list[str] | None = None


current_options: ContextVar[ProviderOptions | None] = ContextVar("provider_options", default=None)


def confidence(default: float) -> float:
    options = current_options.get()
    value = options.confidence if options else None
    return default if value is None else value
