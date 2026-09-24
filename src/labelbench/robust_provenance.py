"""Version the executable protocol and model files, not just model display names."""

import hashlib
import subprocess
from pathlib import Path

from labelbench.registry import default_registry
from labelbench.settings import Settings


def source_signature(root: Path) -> str:
    digest = hashlib.sha256()
    for directory in (root / "src", root / "scripts"):
        for path in sorted(directory.rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def model_signatures(settings: Settings, providers: list[str]) -> dict:
    registry = default_registry(settings)
    signatures = {}
    for name in providers:
        provider = registry[name]
        files = set()
        for attribute in ("_checkpoint", "checkpoint", "_onnx_checkpoint", "_external_data", "_config", "_source", "_adapter_source"):
            value = getattr(provider, attribute, None)
            if isinstance(value, Path) and value.is_file():
                files.add(value)
        if Path(provider.model_name).is_file():
            files.add(Path(provider.model_name))
        signatures[name] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
        signatures[name]["model_id"] = provider.model_name
    return signatures


def gpu_info() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                        "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return "nvidia-smi unavailable; inspect worker log"
