"""LM Studio client and deterministic application of VLM annotation edits."""

from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from labelbench.contracts import Annotation, LLMMessage, RunResult
from labelbench.llm_protocol import REVIEW_PROMPT, apply_review, compact_candidates, marked

SYSTEM_PROMPT = REVIEW_PROMPT


class LMStudioError(RuntimeError):
    """Raised when the local OpenAI-compatible endpoint cannot answer."""


def list_models(base_url: str, timeout: float) -> list[str]:
    """Return model ids exposed by LM Studio."""

    payload = _request_json("GET", f"{base_url.rstrip('/')}/models", None, timeout)
    return [str(item["id"]) for item in payload.get("data", []) if item.get("id")]


def review_run(
    base_url: str,
    api_key: str,
    timeout: float,
    model: str,
    prompt: str,
    history: list[LLMMessage],
    run: RunResult,
    image_path: Path,
    providers: list[str],
) -> dict[str, Any]:
    """Send the image and compact detections to LM Studio, then apply its actions."""

    detections, aliases, groups = compact_candidates(run, providers)
    image_bytes = base64.b64encode(image_path.read_bytes()).decode("ascii")
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    user_text = (
        f"User instruction:\n{prompt}\n\n"
        f"Machine detections JSON:\n{json.dumps(detections, ensure_ascii=False, separators=(',', ':'))}"
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Each review is independent: do not resend verbose previous JSON or stale IDs.
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_bytes}"}},
            ],
        }
    )
    request = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 4000,
    }
    response = _request_json(
        "POST", f"{base_url.rstrip('/')}/chat/completions", request, timeout, api_key
    )
    usage = {key: response.get("usage", {}).get(key, 0)
             for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    content, truncated = _completion_content(response)
    parsed = None if truncated else _parse_json(content)
    retried = parsed is None
    if retried:
        # Regenerate from the image, never apply guessed closures or incomplete actions.
        retry_request = {
            **request,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + (
                    "\nPrevious output was incomplete. Start over. Omit valid singleton groups. "
                    "Resolve every multi-candidate group. Compact complete JSON only."
                )},
                messages[-1],
            ],
        }
        response = _request_json(
            "POST", f"{base_url.rstrip('/')}/chat/completions", retry_request, timeout, api_key
        )
        usage = {key: value + response.get("usage", {}).get(key, 0) for key, value in usage.items()}
        content, truncated = _completion_content(response)
        parsed = None if truncated else _parse_json(content)
    base_annotations = [annotation for provider in providers for annotation in run.providers[provider].annotations]
    refined, removed_ids, notes = apply_actions(parsed, base_annotations, run.image_size)
    review = [marked(a, "kept") for a in refined]
    decisions = {"refined_annotations": refined, "review_annotations": review,
                 "removed_ids": removed_ids, "notes": notes, "invalid_decisions": 0}
    if parsed is not None and any(
        key in parsed for key in ("pick", "fuse", "drop", "uncertain", "edit", "add")
    ):
        decisions = apply_review(parsed, aliases, groups, run.image_size)
    if parsed is None:
        notes = (
            "Ответ VLM оборван по лимиту токенов. Выберите меньше моделей или сузьте задачу."
            if truncated else "VLM не вернул полный JSON с actions. Повторите запрос."
        )
    return {
        "model": model,
        "content": content,
        "parsed": parsed is not None,
        "retried": retried,
        **decisions,
        "notes": notes if parsed is None else decisions["notes"],
        "usage": usage,
        "context_chars": len(user_text),
        "compact_review": {"coordinates": "normalized_0_1", "candidates": detections, "decisions": parsed},
    }


def _completion_content(response: dict[str, Any]) -> tuple[str, bool]:
    try:
        choice = response["choices"][0]
        return str(choice["message"]["content"]), choice.get("finish_reason") == "length"
    except (KeyError, IndexError, TypeError) as error:
        raise LMStudioError("LM Studio returned no assistant content") from error


def apply_actions(
    payload: dict[str, Any] | None,
    annotations: list[Annotation],
    image_size: list[int],
) -> tuple[list[Annotation], list[str], str]:
    """Apply VLM actions while validating every resulting annotation."""

    if not payload:
        return [], [], ""
    full = payload.get("annotations") or payload.get("final_annotations")
    if isinstance(full, list):
        refined = [_annotation_from_payload(item, image_size) for item in full if isinstance(item, dict)]
        return [item for item in refined if item is not None], [], str(payload.get("notes", ""))
    current = {annotation.id: annotation for annotation in annotations}
    removed: list[str] = []
    actions = payload.get("actions", [])
    if not isinstance(actions, list):
        return [], [], str(payload.get("notes", ""))
    for action in actions[:500]:
        if not isinstance(action, dict):
            continue
        kind = str(action.get("action", "")).lower()
        identifier = str(action.get("id", ""))
        if kind == "remove" and identifier in current:
            current.pop(identifier)
            removed.append(identifier)
        elif kind == "modify" and identifier in current:
            candidate = current[identifier].model_dump()
            for field in ("label", "score", "bbox_xywh", "polygon", "text"):
                if field in action:
                    candidate[field] = action[field]
            if "bbox_xywh" in action and "polygon" not in action:
                candidate["polygon"] = None
            updated = _annotation_from_payload(candidate, image_size, identifier)
            if updated is not None:
                current[identifier] = updated
        elif kind == "add":
            added = _annotation_from_payload(action, image_size)
            if added is not None:
                current[added.id] = added
    return list(current.values()), removed, str(payload.get("notes", ""))


def _annotation_from_payload(
    payload: dict[str, Any], image_size: list[int], fallback_id: str | None = None
) -> Annotation | None:
    try:
        x, y, width, height = (float(value) for value in payload["bbox_xywh"])
        image_width, image_height = (float(value) for value in image_size[:2])
        x = min(max(0.0, x), image_width)
        y = min(max(0.0, y), image_height)
        width = min(max(0.0, width), image_width - x)
        height = min(max(0.0, height), image_height - y)
        if width <= 0.0 or height <= 0.0:
            return None
        polygon = _clamp_polygon(payload.get("polygon"), image_width, image_height)
        return Annotation(
            id=str(payload.get("id") or fallback_id or f"llm-{uuid.uuid4().hex[:12]}"),
            label=str(payload.get("label") or "text_line"),
            score=min(1.0, max(0.0, float(payload.get("score", 0.5)))),
            bbox_xywh=[x, y, width, height],
            polygon=polygon,
            text=payload.get("text"),
            attributes={"source": "lm_studio"},
            provider="llm_review",
        )
    except (KeyError, TypeError, ValueError, ValidationError):
        return None


def _clamp_polygon(
    polygon: Any, image_width: float, image_height: float
) -> list[list[float]] | None:
    if polygon is None:
        return None
    if isinstance(polygon, list) and len(polygon) == 1 and isinstance(polygon[0], list):
        polygon = polygon[0]
    if isinstance(polygon, list) and polygon and all(isinstance(v, (int, float)) for v in polygon):
        if len(polygon) % 2:
            return None
        polygon = [polygon[i:i + 2] for i in range(0, len(polygon), 2)]
    if not isinstance(polygon, list) or len(polygon) < 3:
        return None
    points: list[list[float]] = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        x = min(max(0.0, float(point[0])), image_width)
        y = min(max(0.0, float(point[1])), image_height)
        points.append([x, y])
    return points


def _parse_json(content: str) -> dict[str, Any] | None:
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    # Never salvage a nested action from a truncated outer JSON object.
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if not any(
        isinstance(value.get(key), list)
        for key in ("actions", "annotations", "final_annotations", "pick", "fuse", "drop", "uncertain", "edit", "add")
    ):
        return None
    for key in ("pick", "fuse", "drop", "uncertain", "edit", "add"):
        if key in value and not isinstance(value[key], list):
            return None
    return value


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
    api_key: str = "",
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise LMStudioError(f"LM Studio HTTP {error.code}: {error.read().decode('utf-8', 'ignore')[-1000:]}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise LMStudioError(f"LM Studio unavailable at {url}: {error}") from error
