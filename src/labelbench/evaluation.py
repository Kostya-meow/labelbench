"""Class-agnostic instance evaluation against explicit ground-truth polygons."""

from __future__ import annotations

from typing import Any

from scipy.optimize import linear_sum_assignment

from labelbench.contracts import Annotation
from labelbench.geometry import overlap_matrix, shape


def metrics(tp: int, fp: int, fn: int, iou_sum: float) -> dict[str, Any]:
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "mean_matched_iou": iou_sum / tp if tp else None,
        "iou_sum": iou_sum,
    }


def evaluate(predictions: list[Annotation], truth: list[Annotation],
             threshold: float = 0.5) -> dict[str, Any]:
    """Maximum-cardinality matching, then maximum IoU. Duplicates count as FP."""
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    if any(shape(item.polygon) is None for item in truth):
        raise ValueError("Ground truth contains invalid polygons; fix it before evaluation")
    overlaps = overlap_matrix(predictions, truth)
    # Cardinality bonus exceeds any possible total IoU improvement.
    reward = (overlaps >= threshold) * (min(overlaps.shape) + 1 + overlaps)
    rows, cols = linear_sum_assignment(reward, maximize=True)
    matched = [(int(a), int(b)) for a, b in zip(rows, cols) if overlaps[a, b] >= threshold]
    pred_ids = {a for a, _ in matched}
    truth_ids = {b for _, b in matched}
    result = metrics(len(matched), len(predictions) - len(matched), len(truth) - len(matched),
                     sum(float(overlaps[a, b]) for a, b in matched))
    result.update({
        "iou_threshold": threshold,
        "matching": "polygon_max_cardinality_then_iou_v1",
        "matches": [{"prediction": predictions[a].id, "truth": truth[b].id,
                     "iou": float(overlaps[a, b])} for a, b in matched],
        "false_positives": [x.id for i, x in enumerate(predictions) if i not in pred_ids],
        "false_negatives": [x.id for i, x in enumerate(truth) if i not in truth_ids],
        "invalid_predictions": sum(shape(x.polygon) is None for x in predictions),
        "empty_page": not predictions and not truth,
    })
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = metrics(sum(x["tp"] for x in rows), sum(x["fp"] for x in rows),
                     sum(x["fn"] for x in rows), sum(x["iou_sum"] for x in rows))
    result["pages"] = len(rows)
    result["invalid_predictions"] = sum(x["invalid_predictions"] for x in rows)
    return result
