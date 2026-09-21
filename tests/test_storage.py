"""Windows snapshot readers tolerate transient sharing violations, not bad JSON."""

from pathlib import Path

import pytest

from labelbench.storage import read_json, write_json


def test_snapshot_read_retries_sharing_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "experiment.json"
    write_json(path, {"status": "running"})
    original = Path.read_text
    attempts = []

    def read(candidate: Path, *args: object, **kwargs: object) -> str:
        attempts.append(candidate)
        if len(attempts) == 1:
            raise PermissionError(13, "Windows sharing violation")
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert read_json(path) == {"status": "running"}
    assert len(attempts) == 2


def test_snapshot_read_preserves_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("broken", encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(broken)
