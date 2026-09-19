"""Validate model decisions before treating any candidate as accepted."""

from __future__ import annotations

from collections import Counter
from typing import Any

from labelbench.llm_batches import DECISION_KEYS
from labelbench.llm_protocol import pixel_polygon


def decision_errors(payload: dict[str, Any], groups: list[Any]) -> list[str]:
    sizes = {group_id: len(options) for group_id, options in groups}
    seen: Counter[int] = Counter()
    errors = []
    for kind in DECISION_KEYS:
        for entry in payload.get(kind, []):
            try:
                if kind == "add":
                    label, polygon = entry
                    if not isinstance(label, str) or not label.strip():
                        raise ValueError("add needs a label")
                    pixel_polygon(polygon, [1, 1])
                    continue
                group_id = entry if kind in ("drop", "uncertain") else entry[0]
                if type(group_id) is not int or group_id not in sizes:
                    raise ValueError("unknown group ID")
                seen[group_id] += 1
                if kind in ("pick", "edit"):
                    if len(entry) != (2 if kind == "pick" else 3):
                        raise ValueError("wrong entry length")
                    index = entry[1]
                    if type(index) is not int or not 0 <= index < sizes[group_id]:
                        raise ValueError("candidate index outside group")
                if kind in ("edit", "fuse"):
                    if kind == "fuse" and len(entry) != 2:
                        raise ValueError("wrong fuse length")
                    pixel_polygon(entry[-1], [1, 1])
            except (ValueError, TypeError, IndexError, KeyError) as error:
                errors.append(f"Invalid {kind}: {error}")
    for group_id, count in seen.items():
        if count > 1:
            errors.append(f"Group {group_id} has {count} decisions; choose exactly ONE candidate, not all")
    missing = [key for key in sizes if not seen[key]]
    if missing:
        errors.append(f"Missing decisions for groups {missing}; choose one or mark uncertain")
    return errors
