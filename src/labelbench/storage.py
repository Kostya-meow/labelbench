"""Atomic JSON persistence for experiment manifests and reports."""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    """Read an atomic snapshot, retrying transient Windows sharing violations."""
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.025)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        # Windows readers/antivirus can briefly hold a destination without delete-sharing.
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.025)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
