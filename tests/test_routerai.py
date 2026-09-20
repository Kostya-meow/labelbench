import base64
import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from labelbench import api
from labelbench.contracts import Annotation, LLMReviewRequest, ProviderResult, RunResult
from labelbench.llm import LMStudioError, _request_json, _review_batch


@pytest.mark.parametrize("backend", ["routerai", "lm_studio"])
@pytest.mark.parametrize("mode", ["all", "batched"])
def test_review_api_routes_backend_and_mode(monkeypatch, tmp_path: Path, backend: str, mode: str) -> None:
    annotation = Annotation(id="a", label="text", score=0.9, bbox_xywh=[0, 0, 10, 10],
                            polygon=[[0, 0], [10, 0], [10, 10]], provider="test")
    run = RunResult(run_id="r", image_name="page.jpg", image_size=[20, 20], consensus_score=0,
                    providers={"test": ProviderResult(provider="test", model="test", image_name="page.jpg",
                        image_size=[20, 20], annotations=[annotation], elapsed_seconds=0)})
    (tmp_path / "page.jpg").write_bytes(b"test-image")
    monkeypatch.setattr(api.service, "load_run", lambda _: run)
    monkeypatch.setattr(api.service, "settings", replace(api.service.settings, images_dir=tmp_path))
    captured = {}

    def request(method, url, payload, timeout, api_key="") -> dict:
        captured.update(url=url, payload=payload, key=api_key)
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"groups":{"0":0}}'}}]}

    monkeypatch.setattr("labelbench.llm._request_json", request)
    response = TestClient(api.app).post("/api/llm/review", json={
        "run_id": "r", "model": "vendor/vision", "prompt": "check", "providers": ["test"],
        "backend": backend, "request_mode": mode, "api_key": "test-key-placeholder",
        "max_output_tokens": 8192,
    })
    assert response.status_code == 200
    result = response.json()
    assert result["complete"] and result["request_mode"] == mode and result["backend"] == backend
    assert len(result["refined_annotations"]) == 1
    assert captured["payload"]["max_tokens"] == 8192
    assert ("response_format" in captured["payload"]) == (backend == "lm_studio")
    if backend == "routerai":
        assert captured["url"] == "https://routerai.ru/api/v1/chat/completions"
        assert captured["key"] == "test-key-placeholder"
    else:
        assert captured["key"] == api.settings.lm_studio_api_key
    assert "test-key-placeholder" not in response.text


def test_missing_key_is_rejected_and_key_is_not_serialized() -> None:
    payload = {"run_id": "r", "model": "m", "prompt": "check", "providers": ["test"], "backend": "routerai"}
    assert TestClient(api.app).post("/api/llm/review", json=payload).status_code == 400
    request = LLMReviewRequest(**payload, api_key="test-key-placeholder")
    assert "test-key-placeholder" not in repr(request)
    assert "api_key" not in request.model_dump()


def test_transport_authentication_and_error_redaction(monkeypatch) -> None:
    def urlopen(request, timeout):
        assert request.get_header("Authorization") == "Bearer test-key-placeholder"
        assert json.loads(request.data)["model"] == "vision"
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, BytesIO(b"bad test-key-placeholder"))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(LMStudioError) as error:
        _request_json("POST", "https://routerai.ru/api/v1/chat/completions", {"model": "vision"}, 1, "test-key-placeholder")
    assert "401" in str(error.value) and "test-key-placeholder" not in str(error.value)


def test_router_image_conversion_and_text_first(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "page.tif"
    Image.new("RGB", (20, 20)).save(path)

    def request(method, url, payload, timeout, api_key="") -> dict:
        text, image = payload["messages"][-1]["content"]
        assert text["type"] == "text"
        data = image["image_url"]["url"]
        assert data.startswith("data:image/png;base64,")
        assert Image.open(BytesIO(base64.b64decode(data.split(",")[1]))).size == (20, 20)
        return {"choices": [{"message": {"content": '{"groups":{"0":0}}'}}]}

    monkeypatch.setattr("labelbench.llm._request_json", request)
    assert _review_batch("https://routerai.ru/api/v1", "test-key-placeholder", 1, "vision", "check", path,
                         [[0, [[]]]], structured=False)["parsed"]
