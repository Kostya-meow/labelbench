from pathlib import Path

import pytest

from labelbench.contracts import Annotation, ProviderResult, RunResult
from labelbench.llm import _parse_json, apply_actions, review_run


def test_truncated_json_cannot_be_mistaken_for_success() -> None:
    assert _parse_json('{"actions":[{"action":"remove","id":"line-1"},') is None
    assert _parse_json('{"action":"remove","id":"line-1"}') is None
    assert _parse_json('{"actions":[],"notes":"checked"}') == {"actions": [], "notes": "checked"}
    assert _parse_json('```json\n{"actions":[]}\n```') == {"actions": []}


def test_apply_llm_actions_keeps_valid_contract_and_bounds() -> None:
    annotations = [
        Annotation(
            id="line-1",
            label="text_line",
            score=0.8,
            bbox_xywh=[10, 10, 40, 12],
            provider="ppocr",
        ),
        Annotation(
            id="line-2",
            label="text_line",
            score=0.7,
            bbox_xywh=[10, 40, 40, 12],
            provider="docufcn",
        ),
    ]
    refined, removed, notes = apply_actions(
        {
            "actions": [
                {"action": "modify", "id": "line-1", "bbox_xywh": [90, 90, 30, 30]},
                {"action": "remove", "id": "line-2"},
                {"action": "add", "label": "text_line", "bbox_xywh": [0, 0, 20, 20]},
            ],
            "notes": "checked",
        },
        annotations,
        [100, 100],
    )

    assert removed == ["line-2"]
    assert notes == "checked"
    assert len(refined) == 2
    assert refined[0].bbox_xywh == [90.0, 90.0, 10.0, 10.0]
    assert refined[0].provider == "llm_review"


def test_apply_llm_actions_clamps_polygon_and_skips_invalid_polygon() -> None:
    annotations, _, _ = apply_actions(
        {
            "annotations": [
                {
                    "id": "line-1",
                    "label": "text_line",
                    "score": 0.9,
                    "bbox_xywh": [-5, -4, 30, 30],
                    "polygon": [[-5, -4], [30, 0], [30, 40]],
                },
                {
                    "id": "line-2",
                    "label": "text_line",
                    "score": 0.9,
                    "bbox_xywh": [1, 1, 10, 10],
                    "polygon": [[1, 1]],
                },
            ]
        },
        [],
        [20, 20],
    )

    assert annotations[0].polygon == [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0]]
    assert annotations[1].polygon is None


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_review_run_sends_image_and_detection_context(monkeypatch, tmp_path: Path, finish_reason: str) -> None:
    image_path = tmp_path / "page.jpg"
    image_path.write_bytes(b"fake-image")
    run = RunResult(
        run_id="run-1",
        image_name="page.jpg",
        image_size=[100, 80],
        providers={
            "ppocr": ProviderResult(
                provider="ppocr",
                model="PP-OCRv5 detection",
                image_name="page.jpg",
                image_size=[100, 80],
                annotations=[
                    Annotation(
                        id="line-1",
                        label="text_line",
                        score=0.8,
                        bbox_xywh=[10, 10, 40, 8],
                        polygon=[[10, 10], [50, 10], [50, 18]],
                        provider="ppocr",
                    )
                ],
                elapsed_seconds=0.1,
            )
        },
        consensus_score=0.0,
    )
    captured: dict[str, object] = {}

    def fake_request(method: str, url: str, payload: dict[str, object] | None, timeout: float, api_key: str = "") -> dict[str, object]:
        captured["method"] = method
        captured["url"] = url
        captured["payload"] = payload
        return {"choices": [{"finish_reason": finish_reason, "message": {"content": '{"actions":[{"action":"keep","id":"line-1"}]}'}}]}

    monkeypatch.setattr("labelbench.llm._request_json", fake_request)
    result = review_run("http://localhost:1234/v1", "key", 10.0, "qwen/qwen3-vl-4b", "check", [], run, image_path, ["ppocr"])

    payload = captured["payload"]
    assert isinstance(payload, dict)
    message = payload["messages"][-1]
    assert message["content"][1]["type"] == "image_url"
    assert "Machine detections JSON" in message["content"][0]["text"]
    if finish_reason == "length":
        assert result["parsed"] is False
        assert result["refined_annotations"] == []
    else:
        assert result["refined_annotations"][0].id == "line-1"
