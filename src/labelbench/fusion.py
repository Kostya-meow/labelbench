"""Explicit polygon baselines. Scores from different models are not calibrated."""

from __future__ import annotations

from labelbench.contracts import Annotation
from labelbench.geometry import polygon_iou, shape


def fuse(candidates: list[Annotation], method: str, iou: float = 0.5,
         min_votes: int = 2) -> list[Annotation]:
    if method == "union":
        return list(candidates)
    if method not in {"nms", "consensus"}:
        raise ValueError(f"Unknown fusion method: {method}")
    remaining = sorted(candidates, key=lambda x: (-x.score, x.provider, x.id))
    result = []
    while remaining:
        anchor, *remaining = remaining
        boundary = shape(anchor.polygon)
        cluster = [anchor]
        rest = []
        for item in remaining:
            # Compare to the anchor, never transitive connected-component chaining.
            label_match = item.label == anchor.label or (
                item.label in {"text", "text_line"} and anchor.label in {"text", "text_line"})
            if label_match and polygon_iou(boundary, shape(item.polygon)) >= iou:
                cluster.append(item)
            else:
                rest.append(item)
        remaining = rest
        votes = len({x.provider for x in cluster})
        if method == "consensus" and votes < min_votes:
            continue
        chosen = anchor
        if method == "consensus":
            # One best peer per provider: duplicate detections cannot buy extra votes.
            chosen = max(cluster, key=lambda x: sum(
                max((polygon_iou(shape(x.polygon), shape(y.polygon)) for y in cluster
                     if y.provider == provider), default=0)
                for provider in {y.provider for y in cluster if y.provider != x.provider}
            ))
        result.append(chosen.model_copy(update={"attributes": {
            **chosen.attributes, "fusion_method": method, "votes": votes,
            "source_ids": [x.id for x in cluster],
        }}))
    return result
