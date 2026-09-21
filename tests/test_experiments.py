from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from labelbench.contracts import Annotation, ProviderResult
from labelbench.datasets import DatasetStore
from labelbench.evaluation import evaluate
from labelbench.experiment_api import experiment_router
from labelbench.experiments import ExperimentRequest, ExperimentStore
from labelbench.fusion import fuse
from labelbench.geometry import shape
from labelbench.inference_options import ProviderOptions, confidence
from labelbench.providers.base import AnnotationProvider, ProviderAvailability
from labelbench.service import AnnotationService
from labelbench.settings import Settings


def box(identifier: str = "a", x: float = 0, score: float = 0.9,
        provider: str = "fake") -> Annotation:
    return Annotation(id=identifier, label="text", score=score, bbox_xywh=[x, 0, 10, 10],
                      polygon=[[x, 0], [x+10, 0], [x+10, 10], [x, 10]], provider=provider)


def test_polygon_matching_counts_duplicates_and_not_rectangles() -> None:
    result = evaluate([box("a"), box("b"), box("c", 30)], [box("gt"), box("gt2", 60)])
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 1)
    assert result["f1"] == pytest.approx(0.4)
    triangle = box().model_copy(update={"polygon": [[0, 0], [10, 0], [0, 10]]})
    assert evaluate([triangle], [box("gt")], 0.75)["tp"] == 0
    assert evaluate([], [box()])["fn"] == 1
    assert evaluate([box()], [])["fp"] == 1
    assert evaluate([], [])["mean_matched_iou"] is None
    assert shape([[0, 0], [1], [2, 2]]) is None
    assert evaluate([box().model_copy(update={"polygon": None})], [box()])["invalid_predictions"] == 1


def test_fusion_deduplicates_and_requires_distinct_providers() -> None:
    assert len(fuse([box("a"), box("b", 0.2)], "nms")) == 1
    assert fuse([box("a"), box("b", 0.2)], "consensus") == []
    result = fuse([box("a"), box("b", 0.2, provider="second")], "consensus")
    assert len(result) == 1 and result[0].attributes["votes"] == 2


def archive(tmp_path: Path) -> Path:
    directory = tmp_path / "archive"
    directory.mkdir()
    Image.new("RGB", (30, 30), "white").save(directory / "one.png")
    content = {"one.png": {"free_lines": [{"words": [{"polygon": box().polygon},
                                                    {"polygon": box("b", 15).polygon}]}]},
               "missing.png": {"free_words": []}}
    (directory / "dataset.json").write_text(json.dumps(content), encoding="utf-8")
    return directory


def test_dataset_preserves_each_word_and_rejects_changed_truth(tmp_path: Path) -> None:
    directory = archive(tmp_path)
    store = DatasetStore(tmp_path / "out")
    manifest = store.register(directory)
    assert manifest["missing_images"] == ["missing.png"]
    assert manifest["polygons"] == 2
    assert len(store.truth(manifest["id"], "one.png")) == 2
    with pytest.raises(ValueError):
        store.image_path(manifest["id"], "../one.png")
    (directory / "dataset.json").write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        DatasetStore(tmp_path / "out").truth(manifest["id"], "one.png")


class Detector(AnnotationProvider):
    name = "fake"
    model_name = "fixture-v1"

    def __init__(self) -> None:
        self.calls = 0
        self.seen = []

    def availability(self) -> ProviderAvailability:
        return ProviderAvailability(True, "fixture")

    def annotate(self, image_path: Path) -> ProviderResult:
        self.calls += 1
        self.seen.append(confidence(0.2))
        return ProviderResult(provider=self.name, model=self.model_name, image_name=image_path.name,
                              image_size=[30, 30], annotations=[box(score=0.4), box("b", 15, 0.9)], elapsed_seconds=0.01)


def test_experiment_api_cache_settings_and_persistence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LABELBENCH_ROOT", str(tmp_path))
    settings = Settings.from_environment()
    settings.ensure_directories()
    detector = Detector()
    service = AnnotationService(settings, {"fake": detector})
    store = ExperimentStore(service)
    app = FastAPI()
    app.include_router(experiment_router(service, store))
    client = TestClient(app)
    dataset = client.post("/api/datasets", json={"root": str(archive(tmp_path))}).json()
    request = {"name": "test", "dataset_id": dataset["id"], "providers": ["fake"],
               "split": "all", "limit": 1, "options": {"fake": {"confidence": 0.5}}}
    response = client.post("/api/experiments", json=request)
    assert response.status_code == 202
    identifier = response.json()["id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get(f"/api/experiments/{identifier}").json()
        if result["status"] not in {"queued", "running"}:
            break
        time.sleep(0.02)
    store._future.result(timeout=5)
    assert result["status"] == "done", result
    assert result["summary"]["fake"]["tp"] == 1
    assert result["summary"]["fake"]["fn"] == 1
    page = client.get(f"/api/experiments/{identifier}/pages/0").json()
    assert page["run"]["image_sha256"]
    opened = client.post(f"/api/experiments/{identifier}/pages/0/open?method=fusion:nms")
    assert opened.status_code == 200
    assert set(opened.json()["providers"]) == {"fusion:nms"}
    assert (settings.output_dir / "combined" / opened.json()["run_id"] / "result.json").is_file()
    external = {"dataset_id": dataset["id"], "image_name": "one.png", "method": "notebook",
                "annotations": [{"polygon": box().polygon}]}
    imported = client.post("/api/evaluations", json=external)
    assert imported.status_code == 200
    assert imported.json()["metrics"]["notebook"]["tp"] == 1
    assert client.post("/api/evaluations", json={**external, "annotations": [{"polygon": []}]}).status_code == 400
    reused = service.run("one.png", ["fake"], dataset_id=dataset["id"],
                         options={"fake": ProviderOptions(confidence=0.5)}, persist=False)
    assert reused.providers["fake"].cached
    lowered = service.run("one.png", ["fake"], dataset_id=dataset["id"],
                          options={"fake": ProviderOptions(confidence=0.1)}, persist=False)
    assert len(lowered.providers["fake"].annotations) == 2
    assert detector.calls == 2
    assert confidence(0.123) == 0.123
    assert ExperimentStore(service).get(identifier)["status"] == "done"
    assert client.get(f"/api/datasets/{dataset['id']}/image?name=../no.png").status_code == 404
    assert client.post("/api/experiments", json={**request, "options": {"fake": {"confidence": 2}}}).status_code == 422
    store._executor.shutdown(wait=True)


def test_experiment_request_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        ExperimentRequest(name="x", dataset_id="x", providers=["a"], match_iou=0)
