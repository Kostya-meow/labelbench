from io import BytesIO
from urllib.error import HTTPError
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from labelbench import api
from labelbench.llm import LMStudioError, _request_json
from labelbench.llm_batches import collect_reviews
from labelbench.progress import ProgressJournal


def test_journal_isolates_requests_and_freezes_finished_elapsed() -> None:
    journal = ProgressJournal()
    first, second = str(uuid4()), str(uuid4())
    report = journal.start(first)
    journal.start(second)("second")
    report("first")
    assert journal.snapshot(first)["events"][0]["message"] == "first"
    assert journal.snapshot(second)["events"][0]["message"] == "second"
    journal.finish(first, "done")
    assert journal.snapshot(first)["status"] == "done"
    with pytest.raises(ValueError):
        journal.start(first)


def test_api_progress_is_visible_during_execution(monkeypatch) -> None:
    identifier = str(uuid4())
    client = TestClient(api.app)

    def operation(request, report):
        report("cached detector result")
        live = client.get(f"/api/progress/{identifier}").json()
        assert live["status"] == "running"
        assert live["events"][-1]["message"] == "cached detector result"
        return {"complete": True}

    monkeypatch.setattr(api, "_create_run", operation)
    response = client.post("/api/runs", json={"image_name": "x.jpg", "providers": ["fake"], "progress_id": identifier})
    assert response.status_code == 200
    assert client.get(f"/api/progress/{identifier}").json()["status"] == "done"
    assert client.get('/api/progress/unknown').status_code == 404


def test_402_is_clear_and_does_not_retry_other_batches(monkeypatch) -> None:
    calls = []

    def urlopen(request, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, 402, "Payment required", {}, BytesIO(b'{"error":"Insufficient balance"}'))

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    events = []
    result = collect_reviews([[i, [[]]] for i in range(100)],
        lambda batch: _request_json("POST", "https://routerai.ru/api/v1/chat/completions", {}, 1),
        progress=events.append)
    assert len(calls) == 1
    assert not result["parsed"]
    assert "402" in result["decisions"]["notes"]
    assert any("Недостаточно средств" in event for event in events)


def test_api_marks_failed_requests(monkeypatch) -> None:
    identifier = str(uuid4())
    client = TestClient(api.app)

    def operation(request, report):
        raise LMStudioError("failure")

    monkeypatch.setattr(api, "_llm_review", operation)
    with pytest.raises(LMStudioError):
        client.post('/api/llm/review', json={"run_id": "r", "model": "m", "prompt": "check",
            "providers": ["fake"], "progress_id": identifier})
    assert client.get(f"/api/progress/{identifier}").json()["status"] == "error"
