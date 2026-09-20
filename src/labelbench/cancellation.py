"""Cooperative cancellation with immediate closure of an in-flight HTTP request."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from threading import Event

import httpx


class ReviewCancelled(Exception):
    pass


class Cancellation:
    def __init__(self) -> None:
        self.event = Event()

    def check(self) -> None:
        if self.event.is_set():
            raise ReviewCancelled("Проверка VLM отменена")


current_cancellation: ContextVar[Cancellation | None] = ContextVar("vlm_cancellation", default=None)


async def cancellable_request(method: str, url: str, body: bytes | None,
                              headers: dict[str, str], timeout: float,
                              token: Cancellation) -> tuple[int, bytes]:
    token.check()
    async with httpx.AsyncClient(timeout=timeout) as client:
        task = asyncio.create_task(client.request(method, url, content=body, headers=headers))
        try:
            while not task.done():
                token.check()
                await asyncio.wait({task}, timeout=0.1)
            token.check()
            response = await task
            return response.status_code, response.content
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def request_with_cancellation(method: str, url: str, body: bytes | None,
                              headers: dict[str, str], timeout: float,
                              token: Cancellation) -> tuple[int, bytes]:
    return asyncio.run(cancellable_request(method, url, body, headers, timeout, token))
