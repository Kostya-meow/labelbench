"""Compact polygon-only VLM candidates and auditable review decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from labelbench.contracts import Annotation, RunResult

MAX_POLYGON_POINTS = 50

REVIEW_PROMPT = '''Review document segmentation using the image. Coordinates are normalized 0..1.
Input is [group_id, [[provider,label,confidence,polygon],...]].
Each polygon has 3..50 points. Candidates in one group overlap and describe the same probable object.
Return compact JSON with: pick, fuse, drop, uncertain, edit, add, notes.
pick entries are [group_id,candidate_index]: select one zero-based option and reject its peers.
fuse entries are [group_id,polygon]: draw one best boundary and reject all group candidates.
drop and uncertain contain group IDs. edit entries are [group_id,candidate_index,polygon].
add entries are [label,polygon] for visibly missing objects only.
Resolve every multi-candidate group with exactly one of pick/fuse/drop/uncertain.
Single-candidate groups are kept by default; list one in drop/uncertain only if visibly wrong.
Never include a single-candidate group in pick, fuse, or edit; omit valid singletons entirely.
Prefer pick when one candidate is accurate. Use fuse only when a new boundary is materially better.
Never return bounding boxes, unchanged polygons, OCR text, markdown, or per-object explanations.
Use at most 50 points per output polygon. Keep notes short. Close the JSON and stop.'''


@dataclass(frozen=True)
class Candidate:
    alias: int
    provider: str
    annotation: Annotation
    polygon: list[list[float]]
    bounds: tuple[float, float, float, float]


def compact_candidates(
    run: RunResult, providers: list[str]
) -> tuple[list[Any], dict[int, Annotation], dict[int, list[int]]]:
    """Group spatial duplicates and serialize only normalized segmentation polygons."""

    width, height = run.image_size
    selected: list[Candidate] = []
    for provider, result in run.providers.items():
        if provider not in providers:
            continue
        for annotation in result.annotations:
            if not annotation.polygon or len(annotation.polygon) < 3:
                continue
            points = _simplify_polygon(annotation.polygon, MAX_POLYGON_POINTS)
            alias = len(selected)
            selected.append(Candidate(alias, provider, annotation, points, _bounds(points)))

    groups = _group_candidates(selected)
    aliases = {item.alias: item.annotation for item in selected}
    group_map: dict[int, list[int]] = {}
    payload: list[Any] = []
    for index, group in enumerate(groups):
        group_id = index
        group_map[group_id] = [item.alias for item in group]
        rows = []
        for item in group:
            polygon = [
                [
                    round(min(1.0, max(0.0, x / width)), 4),
                    round(min(1.0, max(0.0, y / height)), 4),
                ]
                for x, y in item.polygon
            ]
            rows.append([
                item.provider,
                item.annotation.label,
                round(item.annotation.score, 2),
                polygon,
            ])
        payload.append([group_id, rows])
    return payload, aliases, group_map


def _simplify_polygon(polygon: list[list[float]], limit: int) -> list[list[float]]:
    points = [[float(point[0]), float(point[1])] for point in polygon]
    points = [point for index, point in enumerate(points) if not index or point != points[index - 1]]
    if len(points) > 200:
        points = [points[index * len(points) // 200] for index in range(200)]
    while len(points) > limit:
        areas = []
        for index, point in enumerate(points):
            previous = points[index - 1]
            following = points[(index + 1) % len(points)]
            areas.append(abs(
                (point[0] - previous[0]) * (following[1] - previous[1])
                - (point[1] - previous[1]) * (following[0] - previous[0])
            ))
        points.pop(areas.index(min(areas)))
    return points


def _bounds(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs, ys = zip(*points, strict=True)
    return min(xs), min(ys), max(xs), max(ys)


def _family(label: str) -> str:
    return "text" if "text" in label.lower() else label.lower()


def _same_object(left: Candidate, right: Candidate) -> bool:
    if _family(left.annotation.label) != _family(right.annotation.label):
        return False
    lx1, ly1, lx2, ly2 = left.bounds
    rx1, ry1, rx2, ry2 = right.bounds
    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    left_area = max(1.0, (lx2 - lx1) * (ly2 - ly1))
    right_area = max(1.0, (rx2 - rx1) * (ry2 - ry1))
    union = left_area + right_area - intersection
    area_ratio = min(left_area, right_area) / max(left_area, right_area)
    iou = intersection / union if union else 0.0
    overlap_small = intersection / min(left_area, right_area)
    vertical = intersection_height / max(1.0, min(ly2 - ly1, ry2 - ry1))
    horizontal = intersection_width / max(1.0, min(lx2 - lx1, rx2 - rx1))
    if left.provider == right.provider:
        return iou >= 0.78 and area_ratio >= 0.7
    if _family(left.annotation.label) == "text":
        return area_ratio >= 0.2 and vertical >= 0.65 and horizontal >= 0.45 and (
            iou >= 0.2 or overlap_small >= 0.65
        )
    return iou >= 0.45


def _group_candidates(candidates: list[Candidate]) -> list[list[Candidate]]:
    remaining = sorted(
        candidates,
        key=lambda item: (
            (item.bounds[1] + item.bounds[3]) / 2,
            (item.bounds[0] + item.bounds[2]) / 2,
            -item.annotation.score,
        ),
    )
    groups: list[list[Candidate]] = []
    while remaining:
        anchor = remaining.pop(0)
        group = [anchor]
        peers = [candidate for candidate in remaining if _same_object(anchor, candidate)]
        group.extend(peers)
        peer_aliases = {candidate.alias for candidate in peers}
        remaining = [candidate for candidate in remaining if candidate.alias not in peer_aliases]
        groups.append(group)
    return groups


def pixel_polygon(value: Any, size: list[int]) -> list[list[float]]:
    if not isinstance(value, list) or not 3 <= len(value) <= MAX_POLYGON_POINTS:
        raise ValueError("Polygon needs 3..50 points")
    result = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Expected [x,y]")
        x, y = map(float, point)
        if not all(math.isfinite(coordinate) and 0 <= coordinate <= 1 for coordinate in (x, y)):
            raise ValueError("Expected normalized coordinates")
        result.append([x * size[0], y * size[1]])
    area = sum(
        point[0] * result[(i + 1) % len(result)][1]
        - result[(i + 1) % len(result)][0] * point[1]
        for i, point in enumerate(result)
    )
    if abs(area) <= 1e-12:
        raise ValueError("Polygon has zero area")
    return result


def marked(annotation: Annotation, status: str, group_id: str | int = "") -> Annotation:
    return annotation.model_copy(update={
        "attributes": {
            **annotation.attributes,
            "review_status": status,
            "review_group": group_id,
        },
    })


def with_geometry(
    base: Annotation,
    polygon: list[list[float]],
    status: str,
    group_id: str | int,
    identifier: str | None = None,
) -> Annotation:
    x1, y1, x2, y2 = _bounds(polygon)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Empty region")
    return marked(base.model_copy(update={
        "id": identifier or base.id,
        "polygon": polygon,
        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
        "provider": "llm_review",
        "mask_rle": None,
    }), status, group_id)


def apply_review(
    payload: dict[str, Any],
    aliases: dict[int, Annotation],
    groups: dict[int, list[int]],
    size: list[int],
) -> dict[str, Any]:
    """Enforce one accepted result per overlap group and retain rejected candidates for display."""

    alias_group = {alias: group_id for group_id, members in groups.items() for alias in members}
    current = {}
    for alias, annotation in aliases.items():
        group_id = alias_group[alias]
        status = "kept" if len(groups[group_id]) == 1 else "uncertain"
        current[alias] = marked(annotation, status, group_id)

    decided_groups: set[int] = set()
    issues = 0

    def available(group_id: Any) -> list[int]:
        if type(group_id) is not int or group_id not in groups or group_id in decided_groups:
            raise ValueError("Unknown or repeated group")
        return groups[group_id]

    for entry in payload.get("pick", []):
        try:
            group_id, candidate_index = entry
            members = available(group_id)
            if (
                not members
                or type(candidate_index) is not int
                or not 0 <= candidate_index < len(members)
            ):
                raise ValueError("Candidate index is outside group")
            alias = members[candidate_index]
            decided_groups.add(group_id)
            for member in members:
                status = "kept" if member == alias else "removed"
                current[member] = marked(aliases[member], status, group_id)
        except (TypeError, ValueError):
            issues += 1

    for key, status in (("drop", "removed"), ("uncertain", "uncertain")):
        for group_id in payload.get(key, []):
            try:
                members = available(group_id)
                decided_groups.add(group_id)
                for member in members:
                    current[member] = marked(aliases[member], status, group_id)
            except ValueError:
                issues += 1

    for entry in payload.get("edit", []):
        try:
            group_id, candidate_index, coordinates = entry
            members = available(group_id)
            if (
                not members
                or type(candidate_index) is not int
                or not 0 <= candidate_index < len(members)
            ):
                raise ValueError("Candidate index is outside group")
            alias = members[candidate_index]
            updated = with_geometry(
                aliases[alias], pixel_polygon(coordinates, size), "modified", group_id
            )
            decided_groups.add(group_id)
            for member in members:
                current[member] = marked(aliases[member], "removed", group_id)
            current[alias] = updated
        except (TypeError, ValueError):
            issues += 1

    for entry in payload.get("fuse", []):
        try:
            group_id, coordinates = entry
            members = available(group_id)
            if not members:
                raise ValueError("Unknown group")
            synthetic_id = f"vlm-merge-{group_id}"
            updated = with_geometry(
                aliases[members[0]],
                pixel_polygon(coordinates, size),
                "merged",
                group_id,
                synthetic_id,
            )
            decided_groups.add(group_id)
            for member in members:
                current[member] = marked(aliases[member], "removed", group_id)
            current[synthetic_id] = updated
        except (TypeError, ValueError):
            issues += 1

    for entry in payload.get("add", []):
        try:
            label, coordinates = entry
            synthetic_id = f"vlm-add-{len(current)}"
            base = Annotation(
                id=synthetic_id,
                label=str(label),
                score=0.5,
                bbox_xywh=[0, 0, 1, 1],
                provider="llm_review",
            )
            current[synthetic_id] = with_geometry(
                base, pixel_polygon(coordinates, size), "added", "added", synthetic_id
            )
        except (TypeError, ValueError):
            issues += 1

    unresolved = [
        group_id
        for group_id, members in groups.items()
        if len(members) > 1 and group_id not in decided_groups
    ]
    review = list(current.values())
    notes = str(payload.get("notes", ""))
    if unresolved:
        notes = f"Не решено групп с несколькими кандидатами: {len(unresolved)}. " + notes
    accepted_statuses = {"kept", "modified", "merged", "added"}
    return {
        "refined_annotations": [
            annotation
            for annotation in review
            if annotation.attributes["review_status"] in accepted_statuses
        ],
        "review_annotations": review,
        "removed_ids": [
            annotation.id
            for annotation in review
            if annotation.attributes["review_status"] == "removed"
        ],
        "notes": notes,
        "invalid_decisions": issues,
        "unresolved_groups": unresolved,
    }
