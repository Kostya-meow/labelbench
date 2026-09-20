"""Bound VLM context without separating competing polygons for one object."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from labelbench.progress import Progress, silent_progress

MAX_BATCH_CHARS = 16000
MAX_BATCH_GROUPS = 40
DECISION_KEYS = ("pick", "fuse", "drop", "uncertain", "edit", "add")


def split_groups(groups: list[Any]) -> list[list[Any]]:
    batches: list[list[Any]] = []
    batch: list[Any] = []
    size = 2
    for group in groups:
        length = len(json.dumps(group, ensure_ascii=False, separators=(",", ":"))) + 1
        if batch and (size + length > MAX_BATCH_CHARS or len(batch) >= MAX_BATCH_GROUPS):
            batches.append(batch)
            batch, size = [], 2
        batch.append(group)
        size += length
    if batch:
        batches.append(batch)
    return batches


def collect_reviews(
    groups: list[Any], request: Callable[[list[Any]], dict[str, Any]],
    request_mode: str = "batched",
    progress: Progress = silent_progress,
) -> dict[str, Any]:
    """Keep successful responses and mark every failed group uncertain."""
    if request_mode not in ("batched", "all"):
        raise ValueError("Unknown request mode")
    pending = [groups] if groups and request_mode == "all" else split_groups(groups)
    progress(f"Подготовлено групп: {len(groups)}; частей: {len(pending)}")
    decisions: dict[str, Any] = {key: [] for key in DECISION_KEYS}
    usage = dict.fromkeys(("prompt_tokens", "completion_tokens", "total_tokens"), 0)
    reports: list[dict[str, Any]] = []
    ignored = 0
    while pending:
        batch = pending.pop(0)
        group_ids = [group[0] for group in batch]
        progress(f"Отправляю часть {len(reports) + 1}/{len(reports) + len(pending) + 1}: {len(batch)} групп")
        try:
            reply = request(batch)
        except RuntimeError as error:
            detail = str(error)
            if request_mode == "batched" and len(batch) > 1 and any(term in detail.lower() for term in (
                "exceed_context", "context size", "context length",
            )):
                middle = len(batch) // 2
                pending[0:0] = [batch[:middle], batch[middle:]]
                progress("Контекст переполнен: делю часть на две")
                continue
            reply = {"parsed": False, "notes": detail, "content": ""}
            progress(f"Ошибка API: {detail}")
            for remaining in pending:
                ids = [group[0] for group in remaining]
                decisions["uncertain"].extend(ids)
                reports.append({"parsed": False, "groups": ids, "content": ""})
            pending.clear()
        for key in usage:
            usage[key] += (reply.get("usage") or {}).get(key, 0) or 0
        parsed = reply.get("decisions")
        if reply["parsed"] and isinstance(parsed, dict):
            for key in DECISION_KEYS:
                for entry in parsed.get(key, []):
                    identifier = entry if key in ("drop", "uncertain") else (
                        entry[0] if isinstance(entry, list) and entry else None
                    )
                    if key == "add" or (type(identifier) is int and identifier in group_ids):
                        decisions[key].append(entry)
                    else:
                        ignored += 1
        else:
            decisions["uncertain"].extend(group_ids)
        reports.append({**reply, "groups": group_ids})
        progress("Часть проверена" if reply["parsed"] else "Часть не проверена; исходные сегменты сохранены")
    complete = bool(reports) and all(item["parsed"] for item in reports)
    successful = sum(bool(item["parsed"]) for item in reports)
    notes = [str(item["notes"]) for item in reports if item.get("notes")]
    if not complete:
        notes.insert(0, f"Получен ответ для {successful}/{len(reports)} частей; остальные требуют проверки.")
    decisions["notes"] = " ".join(notes)
    return {
        "decisions": decisions, "usage": usage, "batches": reports,
        "complete": complete, "parsed": successful > 0, "ignored_decisions": ignored,
        "retried": any(item.get("retried", False) for item in reports),
    }
