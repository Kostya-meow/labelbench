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

SYSTEM_PROMPT = """You review historical document segmentation.
The image and machine detections use one coordinate system: pixel coordinates, origin at the
top-left corner, bbox_xywh=[x,y,width,height], polygons are [[x,y], ...].
Return JSON only, with this shape:
{"actions":[
  {"action":"keep","id":"existing-id"},
  {"action":"remove","id":"existing-id"},
  {"action":"modify","id":"existing-id","label":"text_line","bbox_xywh":[x,y,w,h],"polygon":[[x,y],...],"score":0.9},
  {"action":"add","label":"text_line","bbox_xywh":[x,y,w,h],"polygon":[[x,y],...],"score":0.8}
],"notes":"short explanation"}
Use keep for correct detections. Use remove for false positives. Use modify only when the
coordinates or label should change. Add missing lines only when visible in the image. Keep all
coordinates inside the image size. Do not invent OCR text. Never return markdown fences."""


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

    detections = _detection_context(run, providers)
    image_bytes = base64.b64encode(image_path.read_bytes()).decode("ascii")
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    user_text = (
        f"User instruction:\n{prompt}\n\n"
        f"Image name: {run.image_name}\nImage size: {run.image_size}\n"
        f"Machine detections JSON:\n{json.dumps(detections, ensure_ascii=False, separators=(',', ':'))}"
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(message.model_dump() for message in history[-20:])
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
        "response_format": {"type": "json_object"},
    }
    try:
        response = _request_json(
            "POST", f"{base_url.rstrip('/')}/chat/completions", request, timeout, api_key
        )
    except LMStudioError as error:
        if "400" not in str(error):
            raise
        request.pop("response_format")
        response = _request_json(
            "POST", f"{base_url.rstrip('/')}/chat/completions", request, timeout, api_key
        )
    try:
        content = str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as error:
        raise LMStudioError("LM Studio returned no assistant content") from error
    parsed = _parse_json(content)
    base_annotations = [annotation for provider in providers for annotation in run.providers[provider].annotations]
    refined, removed_ids, notes = apply_actions(parsed, base_annotations, run.image_size)
    return {
        "model": model,
        "content": content,
        "parsed": parsed is not None,
        "notes": notes,
        "removed_ids": removed_ids,
        "refined_annotations": refined,
    }


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
            updated = _annotation_from_payload(candidate, image_size, identifier)
            if updated is not None:
                current[identifier] = updated
        elif kind == "add":
            added = _annotation_from_payload(action, image_size)
            if added is not None:
                current[added.id] = added
    return list(current.values()), removed, str(payload.get("notes", ""))


def _detection_context(run: RunResult, providers: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "provider": provider,
            "model": run.providers[provider].model,
            "annotations": [
                annotation.model_dump(exclude={"mask_rle", "text"})
                for annotation in run.providers[provider].annotations
            ],
        }
        for provider in providers
    ]


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
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


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
