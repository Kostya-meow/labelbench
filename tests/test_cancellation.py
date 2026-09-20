import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from labelbench import api
from labelbench.cancellation import (
    Cancellation,
    ReviewCancelled,
    cancellable_request,
    current_cancellation,
)
from labelbench.llm_batches import collect_reviews


def test_cancellation_closes_inflight_connection() -> None:
    async def scenario() -> None:
        entered, disconnected = asyncio.Event(), asyncio.Event()

        async def slow_server(reader, writer):
            await reader.readuntil(b'\r\n\r\n')
            entered.set()
            await reader.read()
            disconnected.set()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(slow_server, '127.0.0.1', 0)
        async with server:
            port = server.sockets[0].getsockname()[1]
            token = Cancellation()
            task = asyncio.create_task(cancellable_request('GET', f'http://127.0.0.1:{port}', None, {}, 30, token))
            await asyncio.wait_for(entered.wait(), 5)
            token.event.set()
            with pytest.raises(ReviewCancelled):
                await asyncio.wait_for(task, 2)
            await asyncio.wait_for(disconnected.wait(), 2)

    asyncio.run(scenario())


def test_cancellation_is_not_swallowed_or_retried() -> None:
    calls = []

    def request(batch):
        calls.append(batch)
        raise ReviewCancelled()

    with pytest.raises(ReviewCancelled):
        collect_reviews([[i, [[]]] for i in range(100)], request)
    assert len(calls) == 1


def test_cancel_endpoint_stops_only_selected_review_and_allows_next(monkeypatch) -> None:
    entered = Event()
    identifier = str(uuid4())
    client = TestClient(api.app)

    def operation(request, report):
        if request.progress_id == identifier:
            token = current_cancellation.get()
            entered.set()
            assert token.event.wait(5)
            token.check()
        return {"parsed": True, "complete": True}

    monkeypatch.setattr(api, '_llm_review', operation)
    payload = {"run_id": "r", "model": "m", "prompt": "check", "providers": ["fake"], "progress_id": identifier}
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(client.post, '/api/llm/review', json=payload)
        assert entered.wait(5)
        assert client.post('/api/llm/cancel/' + identifier).json()['requested']
        assert pending.result(timeout=5).status_code == 499
    assert client.get('/api/progress/' + identifier).json()['status'] == 'cancelled'
    assert not client.post('/api/llm/cancel/' + identifier).json()['requested']
    assert client.post('/api/llm/review', json={**payload, 'progress_id': str(uuid4())}).status_code == 200
    assert current_cancellation.get() is None
