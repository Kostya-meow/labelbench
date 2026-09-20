"""Bounded in-memory progress journal; stores stages, never request payloads."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Lock
from typing import Any

from labelbench.cancellation import Cancellation

Progress = Callable[[str], None]


def silent_progress(message: str) -> None:
    pass


class ProgressJournal:
    def __init__(self) -> None:
        self._lock = Lock()
        self._entries: dict[str, dict[str, Any]] = {}

    def start(self, identifier: str | None, cancellable: bool = False) -> Progress:
        if identifier is None:
            return silent_progress
        with self._lock:
            if identifier in self._entries:
                raise ValueError("Progress ID already used")
            if len(self._entries) >= 200:
                finished = [key for key, item in self._entries.items() if item["status"] != "running"]
                if not finished:
                    raise ValueError("Too many active requests")
                del self._entries[finished[0]]
            self._entries[identifier] = {"started": time.monotonic(), "status": "running", "events": [],
                                         "cancellation": Cancellation() if cancellable else None}
        return lambda message: self.append(identifier, message)

    def token(self, identifier: str | None) -> Cancellation | None:
        with self._lock:
            return self._entries.get(identifier, {}).get("cancellation")

    def cancel(self, identifier: str) -> bool:
        with self._lock:
            entry = self._entries.get(identifier)
            if entry is None or entry["cancellation"] is None:
                raise KeyError(identifier)
            if entry["status"] != "running":
                return False
            entry["cancellation"].event.set()
            return True

    def append(self, identifier: str, message: str) -> None:
        with self._lock:
            entry = self._entries[identifier]
            entry["events"].append({"seconds": round(time.monotonic() - entry["started"], 1), "message": message[:1500]})
            entry["events"] = entry["events"][-200:]

    def finish(self, identifier: str | None, status: str) -> None:
        if identifier is None:
            return
        with self._lock:
            entry = self._entries[identifier]
            entry["status"] = status
            entry["elapsed"] = round(time.monotonic() - entry["started"], 1)

    def snapshot(self, identifier: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._entries.get(identifier)
            if entry is None:
                return None
            return {"status": entry["status"], "events": list(entry["events"]),
                    "elapsed": entry.get("elapsed", round(time.monotonic() - entry["started"], 1))}
