"""Compact normalized VLM candidates and auditable review decisions."""
from __future__ import annotations

import math
from typing import Any

from labelbench.contracts import Annotation, RunResult

REVIEW_PROMPT = '''Review document regions using the image. Coordinates are normalized 0..1.
Input groups: [provider, [[id,label,confidence,polygon],...]]. Polygons have 3..8 [x,y] points.
Return one compact JSON object with keys keep, remove, uncertain, edit, merge, add, notes.
The first six fields are arrays; notes is a short string. Populate them with your actual decisions.
keep/remove/uncertain contain IDs only. keep=correct, remove=definitely wrong or duplicate,
uncertain=needs human review. Unmentioned IDs are uncertain, never implicitly accepted.
Actually inspect and classify the supplied IDs; do not copy the empty template.
Assign every supplied ID to a decision. Use uncertain when visual evidence is insufficient.
edit entries: [id,polygon]. merge entries: [[ids],polygon] or [[ids]] to average their bounds.
Prefer one correct candidate via keep and remove duplicates; merge only the same physical region.
add entries: [label,polygon], only visibly missing regions. Never repeat unchanged geometry.
Use each ID once. At most 8 points per polygon and 8 geometry edits per reply.
No OCR, markdown, explanations per object. Short notes only. Close the JSON and stop.'''


def compact_candidates(run: RunResult, providers: list[str]) -> tuple[list[Any], dict[str, Annotation]]:
    groups: list[Any] = []
    aliases: dict[str, Annotation] = {}
    index = 0
    width, height = run.image_size
    for provider, result in run.providers.items():
        rows = []
        for annotation in result.annotations:
            identifier = f"a{index:x}"
            index += 1
            if provider not in providers:
                continue
            aliases[identifier] = annotation
            x, y, w, h = annotation.bbox_xywh
            points = annotation.polygon or [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
            # Area-based simplification preserves corners and contour order.
            points = [list(p) for p in points]
            if len(points) > 64:
                points = [points[i * len(points) // 64] for i in range(64)]
            while len(points) > 8:
                areas = []
                for i, b in enumerate(points):
                    a, c = points[i-1], points[(i+1) % len(points)]
                    areas.append(abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])))
                points.pop(areas.index(min(areas)))
            polygon = [[round(min(1, max(0, px/width)), 3),
                        round(min(1, max(0, py/height)), 3)] for px, py in points]
            rows.append([identifier, annotation.label, round(annotation.score, 2), polygon])
        if rows:
            groups.append([provider, rows])
    return groups, aliases


def pixel_polygon(value: Any, size: list[int]) -> list[list[float]]:
    if not isinstance(value, list) or not 3 <= len(value) <= 8:
        raise ValueError("Polygon needs 3..8 points")
    result = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Expected [x,y]")
        x, y = map(float, point)
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
            raise ValueError("Expected normalized coordinates")
        result.append([x * size[0], y * size[1]])
    return result


def marked(annotation: Annotation, status: str) -> Annotation:
    return annotation.model_copy(update={
        "attributes": {**annotation.attributes, "review_status": status},
    })


def with_geometry(base: Annotation, polygon: list[list[float]], status: str) -> Annotation:
    xs, ys = zip(*polygon)
    box = [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError("Empty region")
    return marked(base.model_copy(update={"polygon": polygon, "bbox_xywh": box,
                                         "provider": "llm_review", "mask_rle": None}), status)


def apply_review(payload: dict[str, Any], aliases: dict[str, Annotation], size: list[int]) -> dict[str, Any]:
    current = {key: marked(value, "uncertain") for key, value in aliases.items()}
    used: set[str] = set()
    issues = 0
    for key, status in (("keep", "kept"), ("remove", "removed"), ("uncertain", "uncertain")):
        for identifier in payload.get(key, []):
            if not isinstance(identifier, str) or identifier not in current or identifier in used:
                issues += 1
                if isinstance(identifier, str) and identifier in current:
                    current[identifier] = marked(current[identifier], "uncertain")
                continue
            used.add(identifier)
            current[identifier] = marked(current[identifier], status)
    for key in ("edit", "merge", "add"):
        for entry in payload.get(key, []):
            try:
                if key == "add":
                    label, coords = entry
                    base = Annotation(id=f"vlm-add-{len(current)}", label=str(label), score=0.5,
                                      bbox_xywh=[0, 0, 1, 1], provider="llm_review")
                    current[base.id] = with_geometry(base, pixel_polygon(coords, size), "added")
                    continue
                ids = [entry[0]] if key == "edit" else entry[0]
                if not isinstance(ids, list) or not ids or (key == "merge" and len(ids) < 2):
                    raise ValueError("Invalid IDs")
                for identifier in ids:
                    if isinstance(identifier, str) and identifier in used:
                        current[identifier] = marked(current[identifier], "uncertain")
                if any(not isinstance(i, str) or i not in aliases or i in used for i in ids):
                    raise ValueError("Unknown or repeated ID")
                if len(set(ids)) != len(ids):
                    raise ValueError("Repeated ID")
                if len(entry) > 1:
                    polygon = pixel_polygon(entry[1], size)
                elif key == "merge":
                    boxes = [aliases[i].bbox_xywh for i in ids]
                    x, y, w, h = [sum(b[j] for b in boxes)/len(boxes) for j in range(4)]
                    polygon = [[x,y], [x+w,y], [x+w,y+h], [x,y+h]]
                else:
                    raise ValueError("Missing geometry")
                updated = with_geometry(aliases[ids[0]], polygon, "modified" if key == "edit" else "merged")
                used.update(ids)
                for identifier in ids:
                    current[identifier] = marked(aliases[identifier], "removed")
                current[ids[0]] = updated
            except (ValueError, TypeError, IndexError, KeyError):
                issues += 1
    review = list(current.values())
    notes = str(payload.get("notes", ""))
    if aliases and not used:
        notes = "Модель не оценила ни одного исходного кандидата; они оставлены сомнительными. " + notes
    return {
        "refined_annotations": [a for a in review if a.attributes["review_status"] not in ("removed", "uncertain")],
        "review_annotations": review,
        "removed_ids": [a.id for a in review if a.attributes["review_status"] == "removed"],
        "notes": notes,
        "invalid_decisions": issues,
    }
