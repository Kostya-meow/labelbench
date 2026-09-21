"""Object and boundary metrics, including explicit missing-detection penalties."""

import math

from shapely.ops import unary_union

from labelbench.contracts import Annotation
from labelbench.evaluation import evaluate
from labelbench.geometry import shape


def measure(predictions: list[Annotation], truth: list[Annotation], size: tuple[int, int],
            boundary_tolerance: float) -> dict:
    loose = evaluate(predictions, truth, .5)
    strict = evaluate(predictions, truth, .75)
    lookup = {a.id: shape(a.polygon) for a in predictions}
    gt = {a.id: shape(a.polygon) for a in truth}
    tolerance = math.hypot(*size) * boundary_tolerance
    boundary_sum = boundary_iou_sum = 0.
    for match in loose["matches"]:
        a, b = lookup[match["prediction"]], gt[match["truth"]]
        precision = a.boundary.intersection(b.boundary.buffer(tolerance)).length / max(a.length, 1e-9)
        recall = b.boundary.intersection(a.boundary.buffer(tolerance)).length / max(b.length, 1e-9)
        boundary_sum += 2*precision*recall/(precision+recall) if precision+recall else 0
        # Continuous polygon analogue of the interior boundary bands; not raster COCO BIoU.
        band_a = a.difference(a.buffer(-tolerance))
        band_b = b.difference(b.buffer(-tolerance))
        boundary_iou_sum += band_a.intersection(band_b).area / max(band_a.union(band_b).area, 1e-9)
    pred_union = unary_union([p for p in lookup.values() if p is not None])
    gt_union = unary_union(list(gt.values()))
    intersection = pred_union.intersection(gt_union).area
    union = pred_union.area + gt_union.area - intersection
    result = {k: loose[k] for k in ("tp", "fp", "fn", "iou_sum", "precision", "recall", "f1")}
    result.update(tp75=strict["tp"], fp75=strict["fp"], fn75=strict["fn"], f1_75=strict["f1"],
                  boundary_sum=boundary_sum, boundary_iou_sum=boundary_iou_sum,
                  boundary_f1=2*boundary_sum / max(2*loose["tp"]+loose["fp"]+loose["fn"], 1),
                  region_intersection=intersection, region_union=union,
                  region_iou=intersection/union if union else 1.,
                  region_dice=2*intersection/(pred_union.area+gt_union.area) if pred_union.area+gt_union.area else 1.,
                  excess_area=pred_union.area-intersection, missed_area=gt_union.area-intersection,
                  matched_ids=[m["prediction"] for m in loose["matches"]])
    return result


def summarize(rows: list[dict]) -> dict:
    summed = {key: sum(r.get(key, 0) for r in rows) for key in [
        "tp", "fp", "fn", "tp75", "fp75", "fn75", "iou_sum", "boundary_sum", "boundary_iou_sum",
        "region_intersection", "region_union", "excess_area", "missed_area"]}
    tp, fp, fn = (summed[k] for k in ("tp", "fp", "fn"))
    denominator = 2*tp+fp+fn
    summed.update(pages=len(rows), precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                  f1=2*tp/max(denominator, 1),
                  f1_75=2*summed["tp75"]/max(2*summed["tp75"]+summed["fp75"]+summed["fn75"], 1),
                  boundary_f1=2*summed["boundary_sum"]/max(denominator, 1),
                  matched_boundary_iou=summed["boundary_iou_sum"]/tp if tp else None,
                  matched_iou=summed["iou_sum"]/tp if tp else None,
                  region_iou=summed["region_intersection"]/max(summed["region_union"], 1e-9),
                  region_dice=2*summed["region_intersection"]/max(summed["region_union"]+summed["region_intersection"], 1e-9))
    return summed
