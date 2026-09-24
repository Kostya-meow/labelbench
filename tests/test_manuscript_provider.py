"""The isolated integration preserves polygons, threshold options and error reporting."""

import json
import subprocess
from pathlib import Path

import pytest

from labelbench.inference_options import ProviderOptions, current_options
from labelbench.providers.manuscript_mask2former import ManuscriptMask2FormerProvider
from labelbench.registry import default_registry
from labelbench.settings import Settings


def test_registry_and_artifact_availability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LABELBENCH_ROOT", str(tmp_path))
    registry = default_registry(Settings.from_environment())
    provider = registry["manuscript_mask2former_onnx"]
    assert registry["mask2former"].model_name != provider.model_name
    assert not provider.availability().available
    for path in (provider._worker_python, provider._source, provider._checkpoint, provider._external_data, provider._config,
                 provider._onnx_checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    provider._verification.write_text(json.dumps({"verified": True}), encoding="utf-8")
    assert provider.availability().available
    provider._external_data.unlink()
    assert not provider.availability().available


@pytest.mark.parametrize("threshold", [None, .8])
def test_worker_contract_and_threshold(threshold: float | None, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ManuscriptMask2FormerProvider(Settings.from_environment())
    commands = []
    polygon = [[1., 2.], [8., 2.], [7., 5.], [1., 5.]]

    def worker(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        row = {"id": "line-0", "provider": provider.name, "label": "text_line", "score": .9,
               "polygon": polygon, "bbox_xywh": [1., 2., 7., 3.]}
        return subprocess.CompletedProcess(command, 0, json.dumps({"image_size": [20, 20], "annotations": [row]}), "")

    monkeypatch.setattr(subprocess, "run", worker)
    token = current_options.set(ProviderOptions(confidence=threshold) if threshold is not None else None)
    try:
        result = provider.annotate(Path("sample.png"))
    finally:
        current_options.reset(token)
    assert result.annotations[0].polygon == polygon
    assert ("--confidence" in commands[0]) == (threshold is not None)
    if threshold is not None:
        assert commands[0][-1] == str(threshold)


def test_worker_error_is_not_an_empty_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ManuscriptMask2FormerProvider(Settings.from_environment())
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 1, "", "CUDA unavailable"))
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        provider.annotate(Path("sample.png"))
