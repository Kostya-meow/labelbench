"""Family-balanced polygon evidence; no ground truth enters candidate selection."""

from collections import Counter, defaultdict

import numpy as np
from shapely.strtree import STRtree

from labelbench.contracts import Annotation
from labelbench.fusion import fuse
from labelbench.geometry import mask_polygon, polygon_iou, shape
from labelbench.robust_config import RobustConfig, family


def vote_polygon(rows: list[Annotation], limit: int) -> list[list[float]] | None:
    import cv2

    polygons = [shape(a.polygon) for a in rows]
    bounds = np.asarray([p.bounds for p in polygons])
    x0, y0 = bounds[:, :2].min(axis=0)
    x1, y1 = bounds[:, 2:].max(axis=0)
    scale = min(1., (limit-4) / max(x1-x0, y1-y0, 1))
    width, height = int(np.ceil((x1-x0)*scale))+5, int(np.ceil((y1-y0)*scale))+5
    votes = np.zeros((height, width), dtype=np.float32)
    counts = Counter(family(a.provider) for a in rows)
    for row in rows:
        mask = np.zeros_like(votes, dtype=np.uint8)
        points = np.rint((np.asarray(row.polygon) - [x0, y0]) * scale + 2).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
        votes += mask / (len(counts) * counts[family(row.provider)])
    polygon = mask_polygon(votes > .5)
    return ((np.asarray(polygon)-2) / scale + [x0, y0]).tolist() if polygon else None


def _medoid(rows: list[Annotation]) -> tuple[Annotation, float]:
    counts = Counter(family(a.provider) for a in rows)
    weights = np.asarray([1 / (len(counts) * counts[family(a.provider)]) for a in rows])
    polygons = [shape(a.polygon) for a in rows]
    overlaps = np.asarray([[polygon_iou(a, b) for b in polygons] for a in polygons])
    scores = overlaps @ weights
    index = int(np.argmax(scores))
    return rows[index], float(scores[index])


def combine(candidates: list[Annotation], cfg: RobustConfig, *, include_tta: bool = True) -> tuple[dict[str, list[Annotation]], list[dict]]:
    clean = [a for a in candidates if a.attributes["view"].startswith("clean:")]
    outputs = {f"model:{p}": [a for a in clean if a.provider == p] for p in cfg.providers}
    outputs.update({"union": clean, "nms": fuse(clean, "nms", cfg.group_iou, 1),
                    "consensus_clean": [], "joint_medoid": [], "joint_stable": [], "mask_vote": []})
    outputs.update({f"tta:{p}": [] for p in cfg.providers})
    outputs.update({f"trust:{threshold:.1f}": [] for threshold in [.2, .4, .6, .8]})
    valid = [a for a in candidates if shape(a.polygon) is not None]
    polygons = [shape(a.polygon) for a in valid]
    tree = STRtree(polygons)
    remaining = set(range(len(valid)))
    # Clean observations anchor groups; no transitive overlap chaining across adjacent lines.
    order = sorted(remaining, key=lambda i: (not valid[i].attributes["view"].startswith("clean:"),
                                            polygons[i].bounds[1], polygons[i].bounds[0], valid[i].id))
    total_families = len({family(p) for p in cfg.providers})
    evidence = []
    for index in order:
        if index not in remaining:
            continue
        neighbours = [int(j) for j in tree.query(polygons[index], predicate="intersects")
                      if j in remaining and polygon_iou(polygons[index], polygons[j]) >= cfg.group_iou]
        remaining.difference_update(neighbours)
        # At most one observation per provider/view inside a group.
        unique = {}
        for j in neighbours:
            row = valid[j]
            key = (row.provider, row.attributes["view"])
            value = polygon_iou(polygons[index], polygons[j])
            if key not in unique or value > unique[key][0]:
                unique[key] = (value, row)
        rows = [row for _, row in unique.values()]
        medoid, agreement = _medoid(rows)
        groups = defaultdict(set)
        for row in rows:
            groups[family(row.provider)].add(row.attributes["view"])
        stability = float(np.mean([len(views)/len(cfg.views) for views in groups.values()]))
        support = len(groups)/total_families
        trust = float(agreement * (support + stability) / 2)
        identifier = f"group-{len(evidence)}"
        attrs = {"group": identifier, "trust": trust, "stability": stability,
                 "families": len(groups), "agreement": agreement, "selected_source": medoid.id}
        selected = medoid.model_copy(update={"id": identifier, "score": trust, "attributes": attrs})
        outputs["joint_medoid"].append(selected)
        stable = len(groups) >= cfg.min_families and stability >= cfg.stable_fraction
        if stable:
            outputs["joint_stable"].append(selected)
        for threshold in [.2, .4, .6, .8]:
            if trust >= threshold:
                outputs[f"trust:{threshold:.1f}"].append(selected)
        clean_rows = [a for a in rows if a.attributes["view"].startswith("clean:")]
        if len({family(a.provider) for a in clean_rows}) >= cfg.min_families:
            outputs["consensus_clean"].append(_medoid(clean_rows)[0])
        if stable and include_tta:
            polygon = vote_polygon(rows, cfg.vote_resolution)
            geometry = shape(polygon)
            if geometry is not None:
                x1, y1, x2, y2 = geometry.bounds
                outputs["mask_vote"].append(selected.model_copy(update={"polygon": polygon,
                    "bbox_xywh": [x1, y1, x2-x1, y2-y1], "attributes": {**attrs, "geometry": "family_majority"}}))
        evidence.append({**attrs, "status": "stable" if stable else "uncertain",
                         "members": [a.id for a in rows], "discarded_duplicates": len(neighbours)-len(rows)})
    if include_tta:
        for provider in cfg.providers:
            own_config = cfg.model_copy(update={"providers": [provider], "min_families": 1})
            own, _ = combine([a for a in candidates if a.provider == provider], own_config, include_tta=False)
            outputs[f"tta:{provider}"] = own["joint_stable"]
        for excluded in sorted({family(p) for p in cfg.providers}):
            remaining_providers = [p for p in cfg.providers if family(p) != excluded]
            if len({family(p) for p in remaining_providers}) >= cfg.min_families:
                subset_cfg = cfg.model_copy(update={"providers": remaining_providers})
                subset, _ = combine([a for a in candidates if family(a.provider) != excluded], subset_cfg, include_tta=False)
                outputs[f"without:{excluded}"] = subset["joint_stable"]
    return outputs, evidence
