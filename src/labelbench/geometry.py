"""Polygon geometry shared by evaluation and reproducible fusion baselines."""

from __future__ import annotations

import numpy as np
from shapely import make_valid
from shapely.geometry import Polygon

from labelbench.contracts import Annotation


def shape(points: list[list[float]] | None) -> Polygon | None:
    if not points or len(points) < 3:
        return None
    try:
        array = np.asarray(points, dtype=float)
    except (ValueError, TypeError):
        return None
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        return None
    result = Polygon(array)
    return result if result.is_valid and result.area > 0 else None


def polygon_iou(first: Polygon | None, second: Polygon | None) -> float:
    if first is None or second is None or not first.intersects(second):
        return 0.0
    intersection = first.intersection(second).area
    return float(intersection / (first.area + second.area - intersection))


def overlap_matrix(first: list[Annotation], second: list[Annotation]) -> np.ndarray:
    """Use exact planar polygon areas, never rectangle IoU."""
    a = [shape(item.polygon) for item in first]
    b = [shape(item.polygon) for item in second]
    return np.asarray([[polygon_iou(x, y) for y in b] for x in a]).reshape(len(a), len(b))


def mask_polygon(mask: np.ndarray) -> list[list[float]] | None:
    """Expose the largest external contour; retain RLE for the full mask."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return clean_polygon(contour.astype(float).tolist()) if len(contour) >= 3 else None


def clean_polygon(points: list[list[float]]) -> list[list[float]] | None:
    """Repair contour self-touching bridges and expose the largest valid component."""
    if len(points) < 3:
        return None
    polygon = make_valid(Polygon(points))
    if polygon.geom_type != "Polygon":
        parts = []
        pending = list(getattr(polygon, "geoms", []))
        while pending:
            part = pending.pop()
            if part.geom_type == "Polygon":
                parts.append(part)
            else:
                pending.extend(getattr(part, "geoms", []))
        if not parts:
            return None
        polygon = max(parts, key=lambda part: part.area)
    return [[float(x), float(y)] for x, y in polygon.exterior.coords[:-1]] if polygon.area else None
