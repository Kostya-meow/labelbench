"""LM Studio client and deterministic application of VLM annotation edits."""

from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from labelbench.cancellation import current_cancellation, request_with_cancellation
from labelbench.contracts import Annotation, LLMMessage, RunResult
from labelbench.llm_batches import collect_reviews
from labelbench.llm_protocol import apply_review, compact_candidates
from labelbench.llm_schema import STRUCTURED_PROMPT, normalize_decisions, response_format
from labelbench.llm_validation import decision_errors
from labelbench.progress import Progress, silent_progress

SYSTEM_PROMPT = STRUCTURED_PROMPT


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
    *, backend: str = "lm_studio", request_mode: str = "batched",
    max_output_tokens: int = 4000,
    progress: Progress = silent_progress,
) -> dict[str, Any]:
    """Send the image and compact detections to LM Studio, then apply its actions."""

    progress("Подготавливаю полигоны и группы кандидатов")
    detections, aliases, groups = compact_candidates(run, providers)
    if not detections:
        raise LMStudioError("Нет сегментационных полигонов для проверки VLM.")
    collected = collect_reviews(detections, lambda batch: _review_batch(
        base_url, api_key, timeout, model, prompt, image_path, batch,
        structured=backend == "lm_studio", request_mode=request_mode,
        max_output_tokens=max_output_tokens,
        progress=progress,
    ), request_mode=request_mode, progress=progress)
    progress("Собираю решения и проверяю итоговую геометрию")
    decisions = apply_review(collected["decisions"], aliases, groups, run.image_size)
    decisions["invalid_decisions"] += collected["ignored_decisions"]
    return {
        **collected, **decisions, "model": model,
        "backend": backend, "request_mode": request_mode,
        "content": json.dumps(collected["decisions"], ensure_ascii=False, separators=(",", ":")) if collected["parsed"] else "",
        "context_chars": sum(item.get("context_chars", 0) for item in collected["batches"]),
        "batch_count": len(collected["batches"]),
        "compact_review": {
            "coordinates": "normalized_0_1", "candidates": detections,
            "decisions": collected["decisions"], "complete": collected["complete"],
            "backend": backend, "request_mode": request_mode,
        },
    }


def _review_batch(
    base_url: str, api_key: str, timeout: float, model: str, prompt: str,
    image_path: Path, detections: list[Any],
    *, structured: bool = True, request_mode: str = "batched", max_output_tokens: int = 4000,
    progress: Progress = silent_progress,
) -> dict[str, Any]:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    raw_image = image_path.read_bytes()
    if mime not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        from PIL import Image

        with Image.open(image_path) as image:
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            raw_image, mime = buffer.getvalue(), "image/png"
    image_bytes = base64.b64encode(raw_image).decode("ascii")
    system_prompt = SYSTEM_PROMPT
    if request_mode == "all":
        system_prompt = system_prompt.replace(
            "This is one batch of a larger page review. Objects absent from this batch may exist in other batches.",
            "This request contains ALL selected groups for the page.",
        )
    user_text = (
        f"User instruction:\n{prompt}\n\n"
        f"Machine detections JSON:\n{json.dumps(detections, ensure_ascii=False, separators=(',', ':'))}"
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
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
        "max_tokens": max_output_tokens,
    }
    if structured:
        request["response_format"] = response_format(detections)
    progress("Ожидаю ответ модели (API возвращает ответ целиком)")
    response = _request_json(
        "POST", f"{base_url.rstrip('/')}/chat/completions", request, timeout, api_key
    )
    usage = {key: response.get("usage", {}).get(key, 0)
             for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    content, truncated = _completion_content(response)
    progress(f"Ответ получен; токенов: {usage.get('total_tokens', 0)}. Проверяю JSON и решения")
    parsed = None if truncated else _parse_json(content)
    errors = decision_errors(parsed, detections) if parsed is not None else []
    retried = parsed is None or bool(errors)
    if retried:
        progress("Ответ неполный или некорректный: выполняю один повтор")
        # Regenerate from the image, never apply guessed closures or incomplete actions.
        retry_request = {
            **request,
            "messages": [
                {"role": "system", "content": system_prompt + (
                    "\nPrevious output failed validation. Start over. "
                    "Resolve every supplied group with exactly ONE decision in groups. "
                    "Follow the required JSON format. Compact complete JSON only. "
                    + " ".join(errors[:8])
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
        errors = decision_errors(parsed, detections) if parsed is not None else []
    if errors:
        parsed = None
    notes = str(parsed.get("notes", "")) if parsed else ""
    if parsed is None:
        notes = (
            "Ответ VLM оборван по лимиту токенов. Увеличьте лимит ответа или выберите режим по частям."
            if truncated else "VLM не вернул корректные решения после повтора. " + " ".join(errors[:3])
        )
    return {
        "model": model,
        "content": content,
        "parsed": parsed is not None,
        "retried": retried,
        "decisions": parsed,
        "notes": notes,
        "usage": usage,
        "context_chars": len(user_text),
    }


def _completion_content(response: dict[str, Any]) -> tuple[str, bool]:
    try:
        choice = response["choices"][0]
        content = choice["message"].get("content")
        return content if isinstance(content, str) else "", choice.get("finish_reason") == "length"
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


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(content: str) -> dict[str, Any] | None:
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    # Never salvage a nested action from a truncated outer JSON object.
    try:
        value = json.loads(candidate, object_pairs_hook=_unique_object)
    except ValueError:
        return None
    if not isinstance(value, dict):
        return None
    try:
        value = normalize_decisions(value)
    except (ValueError, TypeError):
        return None
    if not any(
        isinstance(value.get(key), list)
        for key in ("pick", "fuse", "drop", "uncertain", "edit", "add")
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
        cancellation = current_cancellation.get()
        if cancellation is not None:
            status, data = request_with_cancellation(method, url, body, headers, timeout, cancellation)
            if status >= 400:
                raise urllib.error.HTTPError(url, status, "VLM API error", {}, BytesIO(data))
            return json.loads(data.decode("utf-8"))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 402:
            raise LMStudioError("Недостаточно средств на балансе API (HTTP 402). Пополните баланс RouterAI или выберите LM Studio. Ответ модели не получен.") from error
        detail = error.read().decode('utf-8', 'ignore')
        if api_key:
            detail = detail.replace(api_key, "[redacted]")
        raise LMStudioError(f"VLM API HTTP {error.code}: {detail[-1000:]}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, httpx.HTTPError) as error:
        raise LMStudioError(f"VLM API unavailable at {url}: {error}") from error
