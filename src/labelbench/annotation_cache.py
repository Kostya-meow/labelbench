"""Persistent detector results keyed by image content and inference configuration."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from pathlib import Path

from PIL import Image
from pydantic import ValidationError

from labelbench.contracts import ProviderResult
from labelbench.providers.base import AnnotationProvider


def cache_key(image_path: Path, provider: AnnotationProvider) -> str:
    configuration: dict[str, object] = {"provider": provider.name, "model": provider.model_name, "version": 1}
    for field in ("_device", "_checkpoint", "_config", "_points_per_side"):
        value = getattr(provider, field, None)
        if isinstance(value, Path):
            metadata = value.stat() if value.is_file() else None
            configuration[field] = [str(value.resolve()), metadata.st_size if metadata else None,
                                    metadata.st_mtime_ns if metadata else None]
        else:
            configuration[field] = value
    source = inspect.getsourcefile(type(provider))
    if source:
        configuration["implementation"] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    model_path = Path(provider.model_name)
    if model_path.is_file():
        stat = model_path.stat()
        configuration["model_file"] = [str(model_path.resolve()), stat.st_size, stat.st_mtime_ns]
    workers = Path(__file__).resolve().parents[2] / "scripts"
    helper_paths = sorted(workers.glob('*worker.py'))
    helper_paths += sorted(Path(__file__).parent.joinpath('providers').glob('*worker.py'))
    configuration["workers"] = [hashlib.sha256(path.read_bytes()).hexdigest() for path in helper_paths]
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    payload = json.dumps([image_hash, configuration], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def load_cached(output: Path, key: str) -> ProviderResult | None:
    try:
        result = ProviderResult.model_validate_json((output / "cache" / f"{key}.json").read_text(encoding="utf-8"))
        if result.cache_key != key:
            return None
        return result.model_copy(update={"cached": True})
    except (OSError, ValueError, ValidationError):
        return None


def load_legacy(output: Path, image_path: Path, image_name: str,
                provider: AnnotationProvider) -> ProviderResult | None:
    """Reuse pre-cache results if identity, dimensions and timestamps still match."""
    with Image.open(image_path) as image:
        size = [image.width, image.height]
    for path in sorted((output / provider.name).glob("*/result.json"),
                       key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            result = ProviderResult.model_validate_json(path.read_text(encoding="utf-8"))
            # Once signed results exist, never fall back to unverifiable old settings.
            if result.image_name == image_name and result.cache_key:
                return None
            if result.cache_key or result.model != provider.model_name or result.provider != provider.name:
                continue
            if result.image_name != image_name or result.image_size != size:
                continue
            if image_path.stat().st_mtime > result.created_at.timestamp():
                continue
            return result.model_copy(update={"cached": True, "cache_origin": "legacy"})
        except (OSError, ValueError, ValidationError):
            continue
    return None


def save_cached(output: Path, key: str, result: ProviderResult) -> ProviderResult:
    result = result.model_copy(update={"cache_key": key})
    directory = output / "cache"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(result.model_dump_json())
        os.replace(temporary, directory / f"{key}.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result
