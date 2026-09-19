#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart Detection Fusion v4
=========================

Три экспериментальных метода для ансамблирования text-line/text-region детекций:

1) Anisotropic Gaussian Consensus
   - polygon/bbox -> ориентированное вытянутое Gaussian-ridge поле;
   - внутри одного provider используется MAX, а не SUM;
   - provider reliability оценивается по согласию с остальными;
   - fusion через positive log-odds evidence.

2) Persistent Components + Watershed
   - sweep по многим threshold;
   - отслеживаем рождение/слияние connected components;
   - устойчивые высокоуровневые компоненты становятся seeds;
   - watershed разрезает слипшиеся области на отдельные строки.

3) Persistent Graph Assignment
   - persistent regions = вершины слева;
   - исходные detections = вершины справа;
   - edge score учитывает confidence, reliability, overlap, geometry;
   - большой штраф, если один bbox накрывает несколько persistent regions;
   - используется Hungarian matching, если установлен scipy, иначе greedy matching.

Все результаты сохраняются В ОДНУ ПАПКУ.

Зависимости:
    pip install numpy pillow opencv-python-headless tqdm

Опционально для точного глобального assignment:
    pip install scipy

Запуск:
    python smart_detection_fusion_v4.py "D:\\json_folder" --images-dir "D:\\images"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

CFG = {
    # ---------- score calibration proxy ----------
    "min_raw_score": 0.20,
    "raw_score_weight": 0.45,
    "rank_score_weight": 0.55,
    "provider_weights": {},

    # ---------- working resolution ----------
    "mask_scale": 0.50,

    # ---------- anisotropic gaussian ----------
    # sigma across the text line, relative to detected line thickness
    "gauss_sigma_perp_ratio": 0.30,
    # gaussian tail outside the long axis flat top
    "gauss_sigma_parallel_tail_ratio": 0.08,
    # central flat part along the long axis
    "gauss_flat_half_ratio": 0.46,
    "gauss_min_sigma": 1.5,
    "gauss_cutoff": 3.0,

    # ---------- provider reliability ----------
    "reliability_min": 0.65,
    "reliability_max": 1.35,
    "reliability_binary_threshold": 0.25,

    # ---------- log-odds fusion ----------
    "logodds_prior": 0.035,
    "support_threshold": 0.25,
    "fusion_support_weight": 0.18,

    # ---------- method 1: fixed gaussian components ----------
    "gaussian_component_threshold": 0.38,
    "gaussian_component_min_area": 20,

    # ---------- persistence sweep ----------
    "persistence_thresholds": [
        0.76, 0.70, 0.64, 0.58, 0.52, 0.46,
        0.40, 0.35, 0.30, 0.25, 0.20, 0.16, 0.12,
    ],
    "persistence_min_area": 14,
    "persistence_min_value": 0.10,
    "persistence_min_birth": 0.30,
    "persistence_overlap_threshold": 0.08,

    # merge near-duplicate seeds before watershed
    "seed_merge_y_ratio": 0.50,
    "seed_merge_gap_height_ratio": 2.0,
    "seed_merge_min_valley": 0.44,

    # ---------- watershed ----------
    "watershed_low_threshold": 0.12,
    "watershed_min_area": 20,
    "watershed_min_width": 5,
    "watershed_min_height": 5,
    "watershed_min_support_mean": 0.70,
    "watershed_min_fused_q75": 0.20,

    # Split a long persistent region at a deep low-consensus valley.
    # This is especially useful for two-page spreads / separate text columns.
    "valley_split_min_aspect": 4.0,
    "valley_split_min_gap_height_ratio": 1.8,
    "valley_split_signal_threshold": 0.18,
    "valley_split_edge_margin_height_ratio": 1.0,

    # ---------- graph assignment ----------
    "assignment_min_region_overlap": 0.08,
    "assignment_min_edge_score": 0.16,
    "assignment_merge_overlap": 0.10,
    "assignment_merge_penalty": 0.20,
    "assignment_large_box_penalty": 0.10,

    # ---------- output de-duplication ----------
    "final_duplicate_affinity": 0.60,
    "final_duplicate_iomin": 0.78,

    # ---------- visualization ----------
    "draw_width": 3,
    "draw_labels": True,
}


MODEL_COLORS = [
    (255, 50, 50),
    (50, 160, 255),
    (50, 210, 70),
    (255, 160, 30),
    (210, 70, 255),
    (30, 210, 210),
    (255, 100, 180),
    (170, 170, 30),
]


# ============================================================
# BASIC GEOMETRY
# ============================================================


def sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def xywh_to_xyxy(box):
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, x2 - x1, y2 - y1]


def bbox_from_polygon(poly):
    arr = np.asarray(poly, dtype=np.float32)
    return [
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        float(arr[:, 0].max()),
        float(arr[:, 1].max()),
    ]


def rect_polygon(box):
    x1, y1, x2, y2 = map(float, box)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b):
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def io_min(a, b):
    inter = intersection_area(a, b)
    den = min(box_area(a), box_area(b))
    return inter / den if den > 0 else 0.0


def overlap_1d(a1, a2, b1, b2):
    inter = max(0.0, min(a2, b2) - max(a1, b1))
    den = min(max(1e-6, a2-a1), max(1e-6, b2-b1))
    return inter / den


def line_affinity(a, b):
    A = a["bbox"]
    B = b["bbox"]

    w1 = max(1.0, A[2]-A[0])
    h1 = max(1.0, A[3]-A[1])
    w2 = max(1.0, B[2]-B[0])
    h2 = max(1.0, B[3]-B[1])

    iv = iou(A, B)
    iom = io_min(A, B)
    xov = overlap_1d(A[0], A[2], B[0], B[2])
    yov = overlap_1d(A[1], A[3], B[1], B[3])

    cx1 = (A[0] + A[2]) * 0.5
    cy1 = (A[1] + A[3]) * 0.5
    cx2 = (B[0] + B[2]) * 0.5
    cy2 = (B[1] + B[3]) * 0.5

    xcenter = math.exp(-abs(cx1-cx2) / max(w1, w2))
    ycenter = math.exp(-abs(cy1-cy2) / max(h1, h2))

    wr = min(w1,w2) / max(w1,w2)
    hr = min(h1,h2) / max(h1,h2)

    horiz = max(w1/h1, w2/h2) >= 1.25

    if horiz:
        return (
            0.17*iv + 0.21*iom + 0.09*xov + 0.22*yov +
            0.15*ycenter + 0.08*wr + 0.08*hr
        )
    else:
        return (
            0.17*iv + 0.21*iom + 0.22*xov + 0.09*yov +
            0.15*xcenter + 0.08*wr + 0.08*hr
        )


# ============================================================
# INPUT NORMALIZATION
# ============================================================


def provider_weight(name):
    return float(CFG["provider_weights"].get(name, 1.0))


def percentile_ranks(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= 1:
        return np.ones_like(values)

    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks / float(len(values)-1)


def flatten_predictions(data):
    detections = []

    for provider, block in data.get("providers", {}).items():
        annotations = block.get("annotations", [])
        raw_scores = [float(a.get("score", 0.0)) for a in annotations]
        ranks = percentile_ranks(raw_scores)

        for ann, rank in zip(annotations, ranks):
            raw = float(ann.get("score", 0.0))
            if raw < CFG["min_raw_score"]:
                continue

            poly = ann.get("polygon")
            bbox_xywh = ann.get("bbox_xywh")

            if poly:
                bbox = bbox_from_polygon(poly)
            elif bbox_xywh:
                bbox = xywh_to_xyxy(bbox_xywh)
                poly = rect_polygon(bbox)
            else:
                continue

            score = (
                CFG["raw_score_weight"] * raw +
                CFG["rank_score_weight"] * float(rank)
            ) * provider_weight(provider)

            score = float(np.clip(score, 0.0, 1.0))

            detections.append({
                "id": ann.get("id"),
                "provider": provider,
                "model": block.get("model", provider),
                "label": ann.get("label", "text"),
                "raw_score": raw,
                "rank_score": float(rank),
                "score": score,
                "bbox": list(map(float, bbox)),
                "polygon": [[float(x), float(y)] for x, y in poly],
                "mask_rle": ann.get("mask_rle"),
            })

    return detections


# ============================================================
# UNICODE-SAFE IMAGE LOAD
# ============================================================


def cv2_imread_unicode(path: Path):
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def candidate_image_paths(image_name, json_path, input_dir, images_dir):
    filename = Path(image_name).name if image_name else ""
    stem = Path(filename).stem if filename else ""

    direct = []
    if images_dir:
        direct += [images_dir / image_name, images_dir / filename]
    direct += [
        json_path.parent / image_name,
        json_path.parent / filename,
        input_dir / image_name,
        input_dir / filename,
        input_dir / "images" / filename,
        input_dir.parent / "images" / filename,
    ]

    seen = set()
    for p in direct:
        if p and p.exists() and p.is_file():
            s = str(p)
            if s not in seen:
                seen.add(s)
                yield p

    roots = []
    if images_dir and images_dir.exists():
        roots.append(images_dir)
    roots += [input_dir, input_dir.parent]

    for root in roots:
        if not root.exists():
            continue

        if filename:
            for p in root.rglob(filename):
                s = str(p)
                if p.is_file() and s not in seen:
                    seen.add(s)
                    yield p

        if stem:
            for p in root.rglob("*"):
                if (
                    p.is_file() and
                    p.stem == stem and
                    p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
                ):
                    s = str(p)
                    if s not in seen:
                        seen.add(s)
                        yield p


def load_source_image(data, json_path, input_dir, images_dir):
    image_name = data.get("image_name", "")

    for p in candidate_image_paths(image_name, json_path, input_dir, images_dir):
        img = cv2_imread_unicode(p)
        if img is not None:
            print(f"\nIMAGE FOUND: {p}")
            return img, p

    print(f"\nIMAGE NOT FOUND: {image_name}")
    w, h = data.get("image_size", [1440, 1172])
    return np.full((int(h), int(w), 3), 255, dtype=np.uint8), None


# ============================================================
# POLYGON MASK
# ============================================================


def polygon_mask(det, H, W, scale):
    mh = int(round(H * scale))
    mw = int(round(W * scale))
    mask = np.zeros((mh, mw), dtype=np.uint8)

    pts = np.asarray(
        [[round(x*scale), round(y*scale)] for x, y in det["polygon"]],
        dtype=np.int32,
    )

    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)

    return mask


# ============================================================
# METHOD 1: ANISOTROPIC GAUSSIAN CONSENSUS
# ============================================================


def oriented_rect_params(det, scale):
    pts = np.asarray(det["polygon"], dtype=np.float32) * float(scale)

    if len(pts) >= 3:
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    else:
        x1, y1, x2, y2 = det["bbox"]
        cx = 0.5*(x1+x2)*scale
        cy = 0.5*(y1+y2)*scale
        w = (x2-x1)*scale
        h = (y2-y1)*scale
        angle = 0.0

    w = max(float(w), 2.0)
    h = max(float(h), 2.0)

    if w >= h:
        major = w
        minor = h
        theta = math.radians(angle)
    else:
        major = h
        minor = w
        theta = math.radians(angle + 90.0)

    return float(cx), float(cy), float(major), float(minor), float(theta)


def add_gaussian_ridge(target, det, scale):
    """
    Flat-top elongated Gaussian ridge:
      - вдоль строки почти плоское плато;
      - по высоте Gaussian;
      - за концами строки Gaussian tail.

    Внутри одного provider target обновляется через MAX.
    """
    H, W = target.shape
    cx, cy, major, minor, theta = oriented_rect_params(det, scale)

    sigma_perp = max(CFG["gauss_min_sigma"], minor * CFG["gauss_sigma_perp_ratio"])
    sigma_tail = max(CFG["gauss_min_sigma"], major * CFG["gauss_sigma_parallel_tail_ratio"])
    flat_half = major * CFG["gauss_flat_half_ratio"]

    pad_u = flat_half + CFG["gauss_cutoff"] * sigma_tail
    pad_v = CFG["gauss_cutoff"] * sigma_perp

    c = abs(math.cos(theta))
    s = abs(math.sin(theta))
    rx = int(math.ceil(c*pad_u + s*pad_v)) + 2
    ry = int(math.ceil(s*pad_u + c*pad_v)) + 2

    x1 = max(0, int(math.floor(cx-rx)))
    x2 = min(W, int(math.ceil(cx+rx+1)))
    y1 = max(0, int(math.floor(cy-ry)))
    y2 = min(H, int(math.ceil(cy+ry+1)))

    if x2 <= x1 or y2 <= y1:
        return

    xs = np.arange(x1, x2, dtype=np.float32) - cx
    ys = np.arange(y1, y2, dtype=np.float32) - cy
    xx, yy = np.meshgrid(xs, ys)

    ct = math.cos(theta)
    st = math.sin(theta)

    u = xx*ct + yy*st
    v = -xx*st + yy*ct

    du = np.maximum(np.abs(u) - flat_half, 0.0)

    field = np.exp(
        -0.5 * (du / sigma_tail)**2
        -0.5 * (v / sigma_perp)**2
    ).astype(np.float32)

    field *= float(det["score"])

    patch = target[y1:y2, x1:x2]
    np.maximum(patch, field, out=patch)


def build_provider_gaussian_maps(detections, H, W):
    scale = float(CFG["mask_scale"])
    mh = max(32, int(round(H*scale)))
    mw = max(32, int(round(W*scale)))

    providers = sorted(set(d["provider"] for d in detections))
    maps = {p: np.zeros((mh, mw), dtype=np.float32) for p in providers}

    for det in detections:
        add_gaussian_ridge(maps[det["provider"]], det, scale)

    return providers, maps, (mh, mw)


def soft_dice(a, b):
    num = 2.0 * float(np.minimum(a,b).sum())
    den = float(a.sum() + b.sum()) + 1e-6
    return num / den


def estimate_provider_reliability(providers, maps):
    """
    Page-adaptive reliability без GT.
    Модель получает больший вес, если её spatial evidence
    согласуется с коллективом остальных моделей.
    """
    raw = {}

    for p in providers:
        own = maps[p]
        others = [maps[q] for q in providers if q != p]

        if not others:
            raw[p] = 1.0
            continue

        other = np.median(np.stack(others, axis=0), axis=0)
        dice = soft_dice(own, other)

        mask = own >= CFG["reliability_binary_threshold"]
        if mask.any():
            precision_like = float(other[mask].mean())
        else:
            precision_like = 0.0

        raw[p] = 0.65*dice + 0.35*precision_like

    vals = np.asarray(list(raw.values()), dtype=np.float32)
    med = float(np.median(vals)) if len(vals) else 1.0
    med = max(med, 1e-4)

    reliabilities = {}
    for p in providers:
        r = raw[p] / med
        r *= provider_weight(p)
        r = float(np.clip(r, CFG["reliability_min"], CFG["reliability_max"]))
        reliabilities[p] = r

    return reliabilities, raw


def fuse_logodds(providers, maps, reliabilities):
    """
    Reliability-weighted fusion, but with a crucial anti-dominance term:
    the SECOND strongest provider is the main signal.

    Thus one detector cannot paint a confident bridge between nearby lines.
    """
    prior = float(CFG["logodds_prior"])
    base_logit = float(logit(prior))

    weighted_stack = []
    evidence_sum = np.zeros_like(next(iter(maps.values())), dtype=np.float32)

    for p in providers:
        m = np.clip(maps[p], 0.0, 1.0)
        r = float(reliabilities[p])
        weighted_stack.append(np.clip(m * r, 0.0, 1.0))

        prob = prior + (1.0-prior) * m
        evidence = np.maximum(logit(prob) - base_logit, 0.0)
        evidence_sum += r * evidence.astype(np.float32)

    stack = np.stack(weighted_stack, axis=0).astype(np.float32)
    support = (stack >= CFG["support_threshold"]).sum(axis=0).astype(np.float32)
    support_ratio = support / max(1.0, float(len(providers)))

    ordered = np.sort(stack, axis=0)
    top1 = ordered[-1]
    top2 = ordered[-2] if len(providers) >= 2 else np.zeros_like(top1)
    top3 = ordered[-3] if len(providers) >= 3 else np.zeros_like(top1)

    logodds_map = sigmoid(base_logit + evidence_sum).astype(np.float32)

    # Log-odds can be very confident from a single provider.
    # Gate it so it matters mostly when >= 2 providers agree.
    multi_gate = np.clip(support - 1.0, 0.0, 1.0)

    fused = (
        0.10 * top1 +
        0.50 * top2 +
        0.12 * top3 +
        0.10 * support_ratio +
        0.18 * logodds_map * multi_gate
    )

    return np.clip(fused, 0.0, 1.0).astype(np.float32), support, support_ratio


# ============================================================
# COMPONENT HELPERS
# ============================================================


def components_from_binary(binary, min_area=1):
    num, labels, stats, cents = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    out = []

    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])

        mask = (labels == lab).astype(np.uint8)

        out.append({
            "label": lab,
            "mask": mask,
            "area": area,
            "bbox": [x, y, x+w, y+h],
            "centroid": [float(cents[lab][0]), float(cents[lab][1])],
        })

    return out


def mask_polygon(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    eps = max(1.0, 0.004*cv2.arcLength(cnt, True))
    approx = cv2.approxPolyDP(cnt, eps, True)
    poly = [[float(p[0][0]), float(p[0][1])] for p in approx]
    return poly if len(poly) >= 3 else None


def upscale_bbox(bbox, inv_scale):
    return [float(v*inv_scale) for v in bbox]


def upscale_poly(poly, inv_scale):
    return [[float(x*inv_scale), float(y*inv_scale)] for x,y in poly]


def component_regions_fixed(fused, support, threshold, inv_scale):
    binary = fused >= threshold
    comps = components_from_binary(binary, CFG["gaussian_component_min_area"])
    regions = []

    for i, c in enumerate(comps):
        pix = c["mask"] > 0
        q75 = float(np.quantile(fused[pix], 0.75)) if pix.any() else 0.0
        support_mean = float(support[pix].mean()) if pix.any() else 0.0

        poly = mask_polygon(c["mask"])
        if poly is None:
            poly = rect_polygon(c["bbox"])

        bbox_full = upscale_bbox(c["bbox"], inv_scale)
        poly_full = upscale_poly(poly, inv_scale)

        regions.append({
            "id": f"gauss_{i:04d}",
            "provider": "gaussian_consensus",
            "label": "text",
            "score": q75,
            "final_score": q75,
            "support_models": int(round(support_mean)),
            "bbox": bbox_full,
            "polygon": poly_full,
        })

    return regions


# ============================================================
# METHOD 2: PERSISTENCE TRACKING
# ============================================================


def mask_overlap_fraction(a, b):
    inter = int(np.logical_and(a > 0, b > 0).sum())
    if inter == 0:
        return 0.0
    den = min(int((a>0).sum()), int((b>0).sum()))
    return inter / max(1, den)


def persistent_tracks(fused):
    thresholds = sorted(CFG["persistence_thresholds"], reverse=True)

    tracks = {}
    active = []
    next_id = 0
    finished = []

    for ti, thr in enumerate(thresholds):
        binary = (fused >= thr).astype(np.uint8)
        comps = components_from_binary(binary, CFG["persistence_min_area"])

        if ti == 0:
            for comp in comps:
                tr = {
                    "id": next_id,
                    "birth": float(thr),
                    "last_threshold": float(thr),
                    "last_mask": comp["mask"],
                    "last_bbox": comp["bbox"],
                    "last_centroid": comp["centroid"],
                }
                tracks[next_id] = tr
                active.append(next_id)
                next_id += 1
            continue

        # For each current component, determine parent active tracks.
        comp_parents = [[] for _ in comps]
        track_child = {tid: None for tid in active}

        for tid in active:
            tr = tracks[tid]
            best_j = None
            best_ov = 0.0

            for j, comp in enumerate(comps):
                ov = mask_overlap_fraction(tr["last_mask"], comp["mask"])
                if ov > best_ov:
                    best_ov = ov
                    best_j = j

            if best_j is not None and best_ov >= CFG["persistence_overlap_threshold"]:
                comp_parents[best_j].append(tid)
                track_child[tid] = best_j
            else:
                tr["death"] = float(thr)
                tr["persistence"] = float(tr["birth"] - thr)
                finished.append(tr.copy())

        new_active = []
        consumed = set()

        for j, comp in enumerate(comps):
            parents = comp_parents[j]

            if len(parents) == 1:
                tid = parents[0]
                tr = tracks[tid]
                tr["last_threshold"] = float(thr)
                tr["last_mask"] = comp["mask"]
                tr["last_bbox"] = comp["bbox"]
                tr["last_centroid"] = comp["centroid"]
                new_active.append(tid)
                consumed.add(tid)

            elif len(parents) >= 2:
                # Merge event: finalize all parent structures just BEFORE merge.
                for tid in parents:
                    tr = tracks[tid]
                    tr["death"] = float(thr)
                    tr["persistence"] = float(tr["birth"] - thr)
                    finished.append(tr.copy())
                    consumed.add(tid)

                # Spawn a new low-level merged track.
                tr = {
                    "id": next_id,
                    "birth": float(thr),
                    "last_threshold": float(thr),
                    "last_mask": comp["mask"],
                    "last_bbox": comp["bbox"],
                    "last_centroid": comp["centroid"],
                    "merged_from": list(parents),
                }
                tracks[next_id] = tr
                new_active.append(next_id)
                next_id += 1

            else:
                # New component appears at lower threshold.
                tr = {
                    "id": next_id,
                    "birth": float(thr),
                    "last_threshold": float(thr),
                    "last_mask": comp["mask"],
                    "last_bbox": comp["bbox"],
                    "last_centroid": comp["centroid"],
                }
                tracks[next_id] = tr
                new_active.append(next_id)
                next_id += 1

        active = new_active

    lowest = float(thresholds[-1])
    for tid in active:
        tr = tracks[tid]
        tr["death"] = lowest
        tr["persistence"] = float(tr["birth"] - lowest)
        finished.append(tr.copy())

    # Filter + deduplicate nearly identical tracks.
    keep = []
    for tr in finished:
        if tr["birth"] < CFG["persistence_min_birth"]:
            continue
        if tr["persistence"] < CFG["persistence_min_value"]:
            continue
        if int((tr["last_mask"] > 0).sum()) < CFG["persistence_min_area"]:
            continue
        keep.append(tr)

    keep = sorted(
        keep,
        key=lambda t: (t["persistence"], t["birth"], int(t["last_mask"].sum())),
        reverse=True,
    )

    unique = []
    for tr in keep:
        dup = False
        for u in unique:
            ov = mask_overlap_fraction(tr["last_mask"], u["last_mask"])
            if ov >= 0.82:
                dup = True
                break
        if not dup:
            unique.append(tr)

    return unique


def bbox_gap_and_row_similarity(a, b):
    A = a["last_bbox"]
    B = b["last_bbox"]

    h1 = max(1, A[3]-A[1])
    h2 = max(1, B[3]-B[1])
    cy1 = 0.5*(A[1]+A[3])
    cy2 = 0.5*(B[1]+B[3])

    y_close = abs(cy1-cy2) <= CFG["seed_merge_y_ratio"] * max(h1,h2)

    gap = max(0, max(A[0],B[0]) - min(A[2],B[2]))
    max_gap = CFG["seed_merge_gap_height_ratio"] * max(h1,h2)

    return y_close, gap, max_gap


def sample_line_min(field, p1, p2, n=40):
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    xs = np.clip(np.round(xs).astype(int), 0, field.shape[1]-1)
    ys = np.clip(np.round(ys).astype(int), 0, field.shape[0]-1)
    return float(field[ys, xs].min())


def merge_same_line_seeds(tracks, fused):
    """
    Иногда одна реальная строка рождается как 2-3 high-threshold fragments.
    Если они лежат в одной строке и между ними нет глубокой valley,
    объединяем их BEFORE watershed.
    """
    tracks = [dict(t) for t in tracks]
    alive = [True]*len(tracks)

    for i in range(len(tracks)):
        if not alive[i]:
            continue

        for j in range(i+1, len(tracks)):
            if not alive[j]:
                continue

            a = tracks[i]
            b = tracks[j]
            y_close, gap, max_gap = bbox_gap_and_row_similarity(a,b)

            if not y_close or gap > max_gap:
                continue

            ca = a["last_centroid"]
            cb = b["last_centroid"]
            valley = sample_line_min(fused, ca, cb)

            if valley >= CFG["seed_merge_min_valley"]:
                merged_mask = np.maximum(a["last_mask"], b["last_mask"]).astype(np.uint8)
                comps = components_from_binary(merged_mask, 1)
                if comps:
                    # bbox around union
                    ys, xs = np.where(merged_mask > 0)
                    bbox = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
                    cent = [float(xs.mean()), float(ys.mean())]
                else:
                    bbox = a["last_bbox"]
                    cent = a["last_centroid"]

                a["last_mask"] = merged_mask
                a["last_bbox"] = bbox
                a["last_centroid"] = cent
                a["persistence"] = max(a["persistence"], b["persistence"])
                a["birth"] = max(a["birth"], b["birth"])
                alive[j] = False

    return [t for t, ok in zip(tracks, alive) if ok]


def split_horizontal_mask_at_valleys(mask, fused):
    """
    Split a very long horizontal region if there is a wide low-consensus valley.

    This prevents one persistent line from spanning two separate page columns /
    left and right pages through a weak bridge. Word spaces are usually much
    narrower than the required gap (relative to line height), so they are kept.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [mask]

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    w = x2 - x1
    h = y2 - y1

    if h <= 0 or (w / max(1.0, float(h))) < CFG["valley_split_min_aspect"]:
        return [mask]

    roi_mask = mask[y1:y2, x1:x2] > 0
    roi_fused = fused[y1:y2, x1:x2]

    signal = np.ones(w, dtype=np.float32)
    for j in range(w):
        col = roi_mask[:, j]
        if col.any():
            signal[j] = float(roi_fused[:, j][col].mean())
        else:
            signal[j] = 0.0

    # Smooth only lightly; we want a genuine wide valley.
    if w >= 7:
        signal = cv2.GaussianBlur(signal.reshape(1, -1), (7, 1), 0).reshape(-1)

    low = signal < CFG["valley_split_signal_threshold"]
    min_gap = max(3, int(round(CFG["valley_split_min_gap_height_ratio"] * h)))
    edge_margin = max(2, int(round(CFG["valley_split_edge_margin_height_ratio"] * h)))

    runs = []
    start = None
    for j, flag in enumerate(low.tolist()):
        if flag and start is None:
            start = j
        elif not flag and start is not None:
            runs.append((start, j))
            start = None
    if start is not None:
        runs.append((start, w))

    candidates = [
        (a, b) for a, b in runs
        if (b-a) >= min_gap and a >= edge_margin and b <= (w-edge_margin)
    ]

    if not candidates:
        return [mask]

    # Split at the widest/deepest valley first, then recurse.
    candidates.sort(key=lambda ab: ((ab[1]-ab[0]), -float(signal[ab[0]:ab[1]].mean())), reverse=True)
    a, b = candidates[0]
    cut = x1 + (a+b)//2

    left = mask.copy()
    left[:, cut:] = 0
    right = mask.copy()
    right[:, :cut] = 0

    if int(left.sum()) < CFG["watershed_min_area"] or int(right.sum()) < CFG["watershed_min_area"]:
        return [mask]

    result = []
    result.extend(split_horizontal_mask_at_valleys(left, fused))
    result.extend(split_horizontal_mask_at_valleys(right, fused))
    return result


def persistent_watershed(fused, support, tracks):
    H, W = fused.shape

    tracks = merge_same_line_seeds(tracks, fused)

    low_mask = (fused >= CFG["watershed_low_threshold"]).astype(np.uint8)

    # Small closing only to help a line grow through tiny gaps.
    low_mask = cv2.morphologyEx(low_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    markers = np.zeros((H,W), dtype=np.int32)
    markers[low_mask == 0] = 1  # known background

    seed_meta = {}
    label = 2

    for tr in tracks:
        seed = (tr["last_mask"] > 0) & (low_mask > 0)
        if int(seed.sum()) < 3:
            continue

        # Ensure seeds don't overwrite previous seeds.
        seed = seed & (markers == 0)
        if int(seed.sum()) < 3:
            continue

        markers[seed] = label
        seed_meta[label] = tr
        label += 1

    # Watershed image: valleys of fused should become boundaries.
    energy = np.clip((1.0-fused)*255.0, 0, 255).astype(np.uint8)
    energy = cv2.GaussianBlur(energy, (0,0), 1.0)
    ws_img = cv2.cvtColor(energy, cv2.COLOR_GRAY2BGR)

    if label > 2:
        cv2.watershed(ws_img, markers)

    regions = []
    persistence_map = np.zeros_like(fused, dtype=np.float32)

    for lab, tr in seed_meta.items():
        base_mask = ((markers == lab) & (low_mask > 0)).astype(np.uint8)

        # Important post-split: one watershed label may still contain a weak
        # bridge across two text columns/pages. Split it at a wide consensus valley.
        submasks = split_horizontal_mask_at_valleys(base_mask, fused)

        for mask in submasks:
            area = int(mask.sum())

            if area < CFG["watershed_min_area"]:
                continue

            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            x1,y1,x2,y2 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
            if (x2-x1) < CFG["watershed_min_width"] or (y2-y1) < CFG["watershed_min_height"]:
                continue

            pix = mask > 0
            fused_q75 = float(np.quantile(fused[pix], 0.75))
            support_mean = float(support[pix].mean())

            if fused_q75 < CFG["watershed_min_fused_q75"]:
                continue
            if support_mean < CFG["watershed_min_support_mean"]:
                continue

            poly = mask_polygon(mask)
            if poly is None:
                poly = rect_polygon([x1,y1,x2,y2])

            persistence = float(tr["persistence"])
            persistence_map[pix] = np.maximum(persistence_map[pix], persistence)

            regions.append({
                "region_id": len(regions),
                "seed_track_id": tr["id"],
                "birth": float(tr["birth"]),
                "death": float(tr["death"]),
                "persistence": persistence,
                "mask_small": mask,
                "bbox_small": [x1,y1,x2,y2],
                "polygon_small": poly,
                "fused_q75": fused_q75,
                "support_mean": support_mean,
                "centroid_small": [float(xs.mean()), float(ys.mean())],
            })

    return regions, persistence_map, markers, tracks


# ============================================================
# METHOD 3: GRAPH / BIPARTITE ASSIGNMENT
# ============================================================


def detection_small_masks(detections, H, W, scale):
    return [polygon_mask(d, H, W, scale) for d in detections]


def region_overlap(det_mask, region_mask):
    inter = int(np.logical_and(det_mask > 0, region_mask > 0).sum())
    if inter == 0:
        return 0.0, 0.0

    region_area = max(1, int((region_mask>0).sum()))
    det_area = max(1, int((det_mask>0).sum()))

    return inter/region_area, inter/det_area


def region_full_geometry(region, inv_scale):
    bbox = upscale_bbox(region["bbox_small"], inv_scale)
    poly = upscale_poly(region["polygon_small"], inv_scale)
    return bbox, poly


def candidate_region_counts(det_masks, regions):
    counts = []

    for dm in det_masks:
        k = 0
        overlaps = []
        for region in regions:
            ro, do = region_overlap(dm, region["mask_small"])
            overlaps.append(ro)
            if ro >= CFG["assignment_merge_overlap"]:
                k += 1
        counts.append((k, overlaps))

    return counts


def edge_score(det, det_mask, det_reliability, region, region_bbox_full, merge_count):
    region_cov, det_cov = region_overlap(det_mask, region["mask_small"])

    if region_cov < CFG["assignment_min_region_overlap"]:
        return None

    aff = line_affinity(
        det,
        {"bbox": region_bbox_full},
    )

    # Shape compatibility: avoid giant boxes swallowing tiny persistent lines.
    det_area = max(1.0, box_area(det["bbox"]))
    reg_area = max(1.0, box_area(region_bbox_full))
    area_ratio = min(det_area, reg_area) / max(det_area, reg_area)

    merge_penalty = CFG["assignment_merge_penalty"] * max(0, merge_count-1)

    large_penalty = 0.0
    if det_area > 2.5*reg_area:
        large_penalty = CFG["assignment_large_box_penalty"] * min(2.0, det_area/reg_area - 2.5)

    persistence_norm = min(1.0, region["persistence"] / 0.35)

    score = (
        0.24*det["score"] +
        0.15*float(det_reliability) +
        0.23*region_cov +
        0.10*det_cov +
        0.12*aff +
        0.08*area_ratio +
        0.08*persistence_norm -
        merge_penalty -
        large_penalty
    )

    return float(score)


def solve_assignment(score_matrix):
    R, D = score_matrix.shape

    try:
        from scipy.optimize import linear_sum_assignment

        # Add dummy columns for "no assignment".
        dummy = np.zeros((R, R), dtype=np.float32)
        aug = np.concatenate([score_matrix, dummy], axis=1)

        cost = -aug
        rows, cols = linear_sum_assignment(cost)

        result = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            if c < D:
                result.append((r,c,float(score_matrix[r,c])))
        return result, "hungarian"

    except Exception:
        # Greedy maximum-weight matching fallback.
        edges = []
        for r in range(R):
            for d in range(D):
                s = float(score_matrix[r,d])
                if s > -1e5:
                    edges.append((s,r,d))

        edges.sort(reverse=True)
        used_r = set()
        used_d = set()
        result = []

        for s,r,d in edges:
            if r in used_r or d in used_d:
                continue
            used_r.add(r)
            used_d.add(d)
            result.append((r,d,s))

        return result, "greedy"


def graph_assignment(regions, detections, det_masks, reliabilities, inv_scale):
    if not regions:
        return [], "none", np.zeros((0, len(detections)), dtype=np.float32)

    counts = candidate_region_counts(det_masks, regions)

    R = len(regions)
    D = len(detections)
    scores = np.full((R,D), -1e6, dtype=np.float32)

    region_geoms = [region_full_geometry(r, inv_scale) for r in regions]

    for ri, region in enumerate(regions):
        rbbox, rpoly = region_geoms[ri]

        for di, det in enumerate(detections):
            merge_count = counts[di][0]
            s = edge_score(
                det,
                det_masks[di],
                reliabilities.get(det["provider"], 1.0),
                region,
                rbbox,
                merge_count,
            )

            if s is not None and s >= CFG["assignment_min_edge_score"]:
                scores[ri,di] = s

    matching, solver = solve_assignment(scores)
    matched_regions = set()
    final = []

    for ri, di, s in matching:
        if s < CFG["assignment_min_edge_score"]:
            continue

        region = regions[ri]
        det = detections[di]
        rbbox, rpoly = region_geoms[ri]

        item = dict(det)
        item.update({
            "region_id": ri,
            "persistence": region["persistence"],
            "birth": region["birth"],
            "death": region["death"],
            "region_fused_q75": region["fused_q75"],
            "region_support_mean": region["support_mean"],
            "region_bbox": rbbox,
            "region_polygon": rpoly,
            "provider_reliability": reliabilities.get(det["provider"], 1.0),
            "merge_count": counts[di][0],
            "assignment_score": float(s),
            "final_score": float(s),
            "selected_mode": "graph_assignment",
        })
        final.append(item)
        matched_regions.add(ri)

    # Fallback: if a strong persistent region had no suitable source box,
    # keep its own rectangular region instead of losing it entirely.
    for ri, region in enumerate(regions):
        if ri in matched_regions:
            continue

        rbbox, rpoly = region_geoms[ri]
        fallback_score = (
            0.48*region["fused_q75"] +
            0.28*min(1.0, region["persistence"]/0.35) +
            0.24*min(1.0, region["support_mean"]/3.0)
        )

        final.append({
            "id": f"persistent_region_{ri:04d}",
            "provider": "persistent_region",
            "model": "persistent_region",
            "label": "text",
            "raw_score": float(fallback_score),
            "rank_score": float(fallback_score),
            "score": float(fallback_score),
            "bbox": rbbox,
            "polygon": rect_polygon(rbbox),
            "region_id": ri,
            "persistence": region["persistence"],
            "birth": region["birth"],
            "death": region["death"],
            "region_fused_q75": region["fused_q75"],
            "region_support_mean": region["support_mean"],
            "region_bbox": rbbox,
            "region_polygon": rpoly,
            "provider_reliability": 1.0,
            "merge_count": 1,
            "assignment_score": float(fallback_score),
            "final_score": float(fallback_score),
            "selected_mode": "persistent_region_fallback",
        })

    final = dedupe_final(final)
    return final, solver, scores


def dedupe_final(regions):
    ordered = sorted(regions, key=lambda d: d.get("final_score", d.get("score",0.0)), reverse=True)
    out = []

    for cand in ordered:
        duplicate = False

        for kept in out:
            aff = line_affinity(cand, kept)
            iom = io_min(cand["bbox"], kept["bbox"])

            if (
                aff >= CFG["final_duplicate_affinity"] or
                (iom >= CFG["final_duplicate_iomin"] and aff >= 0.42)
            ):
                duplicate = True
                break

        if not duplicate:
            out.append(cand)

    return out


# ============================================================
# VISUALIZATION
# ============================================================


def provider_color_map(providers):
    return {p: MODEL_COLORS[i % len(MODEL_COLORS)] for i,p in enumerate(providers)}


def draw_regions(image_bgr, regions, title, mode="polygon", color=(255,0,0), color_by_provider=False):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)

    providers = sorted(set(r.get("provider","?") for r in regions)) if regions else []
    cmap = provider_color_map(providers)

    for idx, r in enumerate(regions):
        c = cmap.get(r.get("provider","?"), color) if color_by_provider else color

        if mode == "box":
            x1,y1,x2,y2 = [int(round(v)) for v in r["bbox"]]
            draw.rectangle([x1,y1,x2,y2], outline=c, width=CFG["draw_width"])
            anchor = (x1,y1)
        else:
            poly = [(int(round(x)), int(round(y))) for x,y in r["polygon"]]
            if not poly:
                continue
            draw.line(poly+[poly[0]], fill=c, width=CFG["draw_width"])
            anchor = (min(x for x,_ in poly), min(y for _,y in poly))

        if CFG["draw_labels"]:
            score = r.get("final_score", r.get("score",0.0))
            pers = r.get("persistence")
            suffix = f" p={pers:.2f}" if pers is not None else ""
            txt = f"{idx} {r.get('provider','?')} {score:.2f}{suffix}"
            x,y = anchor
            draw.text((x+2,max(2,y-13)), txt, fill=c)

    draw.rectangle([0,0,min(canvas.width,1200),24], fill=(255,255,255))
    draw.text((5,5), f"{title} | detections={len(regions)}", fill=(0,0,0))
    return np.asarray(canvas)


def heatmap_overlay(image_bgr, field, title):
    H,W = image_bgr.shape[:2]
    big = cv2.resize(field, (W,H), interpolation=cv2.INTER_LINEAR)
    u8 = np.clip(big*255.0,0,255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(image_bgr,0.68,colored,0.32,0)

    cv2.rectangle(overlay,(0,0),(min(W-1,1200),24),(255,255,255),-1)
    cv2.putText(overlay,title,(5,17),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,0,0),1,cv2.LINE_AA)
    return cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)


def persistent_regions_visual(regions, image_bgr, inv_scale, title):
    draw_regions_list = []
    for r in regions:
        bbox, poly = region_full_geometry(r, inv_scale)
        draw_regions_list.append({
            "provider": "persistent",
            "score": r["fused_q75"],
            "final_score": r["fused_q75"],
            "persistence": r["persistence"],
            "bbox": bbox,
            "polygon": poly,
        })
    return draw_regions(image_bgr, draw_regions_list, title, mode="polygon", color=(0,150,255))


def make_grid(images, labels, target_width=720):
    tiles = []

    for img,label in zip(images,labels):
        pil = Image.fromarray(img)
        ratio = target_width / pil.width
        h = max(1,int(round(pil.height*ratio)))
        pil = pil.resize((target_width,h), Image.Resampling.LANCZOS)

        tile = Image.new("RGB", (target_width,h+34), (255,255,255))
        tile.paste(pil,(0,34))
        draw = ImageDraw.Draw(tile)
        draw.text((8,10),label,fill=(0,0,0))
        tiles.append(tile)

    cols = 2
    rows = math.ceil(len(tiles)/cols)
    cell_w = max(t.width for t in tiles)
    cell_h = max(t.height for t in tiles)

    grid = Image.new("RGB",(cols*cell_w,rows*cell_h),(235,235,235))

    for i,tile in enumerate(tiles):
        x=(i%cols)*cell_w
        y=(i//cols)*cell_h
        grid.paste(tile,(x,y))

    return np.asarray(grid)


# ============================================================
# JSON EXPORT
# ============================================================


def export_regions_json(path, data, method, regions):
    payload = {
        "run_id": data.get("run_id"),
        "image_name": data.get("image_name"),
        "image_size": data.get("image_size"),
        "fusion_method": method,
        "annotations": [],
    }

    for i,r in enumerate(regions):
        payload["annotations"].append({
            "id": r.get("id", f"{method}_{i:04d}"),
            "label": r.get("label","text"),
            "score": float(r.get("final_score",r.get("score",0.0))),
            "bbox_xywh": xyxy_to_xywh(r["bbox"]),
            "polygon": r["polygon"],
            "provider": r.get("provider"),
            "persistence": r.get("persistence"),
            "assignment_score": r.get("assignment_score"),
            "merge_count": r.get("merge_count"),
            "selected_mode": r.get("selected_mode"),
        })

    with open(path,"w",encoding="utf-8") as f:
        json.dump(payload,f,ensure_ascii=False,indent=2)


# ============================================================
# ONE JSON
# ============================================================


def safe_base_name(json_path, image_name):
    image_stem = Path(image_name).stem if image_name else json_path.stem
    return image_stem if image_stem == json_path.stem else f"{json_path.stem}__{image_stem}"


def process_json(json_path, input_dir, output_dir, images_dir):
    with open(json_path,"r",encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data.get("providers"),dict):
        return None

    detections = flatten_predictions(data)
    if not detections:
        return None

    image_bgr, image_path = load_source_image(data,json_path,input_dir,images_dir)
    H,W = image_bgr.shape[:2]

    scale = float(CFG["mask_scale"])
    inv_scale = 1.0/scale

    # ---------- METHOD 1 ----------
    providers, provider_maps, small_hw = build_provider_gaussian_maps(detections,H,W)
    reliabilities, reliability_raw = estimate_provider_reliability(providers,provider_maps)
    fused,support,support_ratio = fuse_logodds(providers,provider_maps,reliabilities)

    method1 = component_regions_fixed(
        fused,
        support,
        CFG["gaussian_component_threshold"],
        inv_scale,
    )

    # ---------- METHOD 2 ----------
    tracks = persistent_tracks(fused)
    persistent_regions,persistence_map,ws_markers,merged_tracks = persistent_watershed(
        fused,support,tracks
    )

    method2 = []
    for i,r in enumerate(persistent_regions):
        bbox,poly = region_full_geometry(r,inv_scale)
        method2.append({
            "id": f"persistent_{i:04d}",
            "provider": "persistent_watershed",
            "model": "persistent_watershed",
            "label": "text",
            "score": r["fused_q75"],
            "final_score": r["fused_q75"],
            "persistence": r["persistence"],
            "bbox": bbox,
            "polygon": poly,
        })

    # ---------- METHOD 3 ----------
    det_masks = detection_small_masks(detections,H,W,scale)
    method3,solver,score_matrix = graph_assignment(
        persistent_regions,
        detections,
        det_masks,
        reliabilities,
        inv_scale,
    )

    base = safe_base_name(json_path,data.get("image_name",""))

    # original
    real_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    Image.fromarray(real_rgb).save(output_dir/f"{base}__00_real.jpg",quality=95)

    # all models
    all_img = draw_regions(
        image_bgr,detections,"01_all_models_on_real",
        mode="polygon",color_by_provider=True,
    )
    Image.fromarray(all_img).save(output_dir/f"{base}__01_all_models_on_real.jpg",quality=94)

    # anisotropic heatmap
    gauss_heat = heatmap_overlay(
        image_bgr,fused,
        "02_anisotropic_logodds_heatmap",
    )
    Image.fromarray(gauss_heat).save(output_dir/f"{base}__02_anisotropic_heatmap.jpg",quality=94)

    # method1 regions
    m1_img = draw_regions(
        image_bgr,method1,
        "03_anisotropic_gaussian_components",
        mode="box",color=(255,0,0),
    )
    Image.fromarray(m1_img).save(output_dir/f"{base}__03_anisotropic_components_on_real.jpg",quality=94)

    # persistence map
    pmax = max([t["persistence"] for t in tracks], default=1.0)
    pnorm = persistence_map / max(pmax,1e-6)
    pers_heat = heatmap_overlay(
        image_bgr,pnorm,
        "04_persistence_map",
    )
    Image.fromarray(pers_heat).save(output_dir/f"{base}__04_persistence_map.jpg",quality=94)

    # method2 persistent regions
    m2_img = draw_regions(
        image_bgr,method2,
        "05_persistent_components_watershed",
        mode="box",color=(0,140,255),
    )
    Image.fromarray(m2_img).save(output_dir/f"{base}__05_persistent_boxes_on_real.jpg",quality=94)

    # graph final boxes
    m3_boxes = draw_regions(
        image_bgr,method3,
        f"06_FINAL_graph_boxes_on_real [{solver}]",
        mode="box",color=(255,0,0),
    )
    Image.fromarray(m3_boxes).save(output_dir/f"{base}__06_FINAL_graph_boxes_on_real.jpg",quality=95)

    # graph final polygons
    m3_poly = draw_regions(
        image_bgr,method3,
        f"07_FINAL_graph_polygons_on_real [{solver}]",
        mode="polygon",color=(255,0,0),
    )
    Image.fromarray(m3_poly).save(output_dir/f"{base}__07_FINAL_graph_polygons_on_real.jpg",quality=95)

    # white background final boxes
    white = np.full_like(image_bgr,255)
    m3_white = draw_regions(
        white,method3,
        "08_FINAL_graph_boxes_on_white",
        mode="box",color=(255,0,0),
    )
    Image.fromarray(m3_white).save(output_dir/f"{base}__08_FINAL_graph_boxes_on_white.jpg",quality=95)

    compare = make_grid(
        [m3_boxes,m2_img,m1_img,gauss_heat,pers_heat,all_img],
        [
            f"06 FINAL graph: {len(method3)}",
            f"05 persistent watershed: {len(method2)}",
            f"03 anisotropic fixed: {len(method1)}",
            "02 anisotropic heatmap",
            "04 persistence map",
            f"01 all models: {len(detections)}",
        ],
        target_width=720,
    )
    Image.fromarray(compare).save(output_dir/f"{base}__99_COMPARE.jpg",quality=92)

    # JSONs
    export_regions_json(output_dir/f"{base}__method1_anisotropic.json",data,"anisotropic_gaussian_consensus",method1)
    export_regions_json(output_dir/f"{base}__method2_persistent.json",data,"persistent_components_watershed",method2)
    export_regions_json(output_dir/f"{base}__FINAL_graph.json",data,"persistent_graph_assignment",method3)

    # Reliability report
    with open(output_dir/f"{base}__provider_reliability.json","w",encoding="utf-8") as f:
        json.dump({
            "solver": solver,
            "providers": providers,
            "reliability": reliabilities,
            "raw_agreement": reliability_raw,
            "persistent_tracks": len(tracks),
            "persistent_regions": len(persistent_regions),
            "final_regions": len(method3),
            "image_found": str(image_path) if image_path else "",
        },f,ensure_ascii=False,indent=2)

    # Persistent diagnostics
    with open(output_dir/f"{base}__persistent_diagnostics.csv","w",newline="",encoding="utf-8-sig") as f:
        fields = [
            "region_id","seed_track_id","birth","death","persistence",
            "fused_q75","support_mean","bbox_x1","bbox_y1","bbox_x2","bbox_y2",
        ]
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for r in persistent_regions:
            b,_ = region_full_geometry(r,inv_scale)
            writer.writerow({
                "region_id": r["region_id"],
                "seed_track_id": r["seed_track_id"],
                "birth": r["birth"],
                "death": r["death"],
                "persistence": r["persistence"],
                "fused_q75": r["fused_q75"],
                "support_mean": r["support_mean"],
                "bbox_x1": b[0],"bbox_y1": b[1],"bbox_x2": b[2],"bbox_y2": b[3],
            })

    return {
        "json": str(json_path),
        "image_found": str(image_path) if image_path else "",
        "raw_detections": len(detections),
        "method1_anisotropic": len(method1),
        "persistent_tracks": len(tracks),
        "method2_persistent": len(method2),
        "method3_graph": len(method3),
        "solver": solver,
    }


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",type=Path,help="Папка с JSON")
    parser.add_argument("--images-dir",type=Path,default=None,help="Папка с исходными изображениями")
    parser.add_argument("--output-dir",type=Path,default=None,help="Папка результата")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    images_dir = args.images_dir.resolve() if args.images_dir else None

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else input_dir.parent / f"{input_dir.name}_fusion_v4"
    )
    output_dir.mkdir(parents=True,exist_ok=True)

    json_files = sorted(input_dir.rglob("*.json"))

    print(f"JSON:   {len(json_files)}")
    print(f"OUTPUT: {output_dir}")
    if images_dir:
        print(f"IMAGES: {images_dir}")
    print()

    rows = []

    for path in tqdm(json_files,desc="Fusion v4"):
        try:
            row = process_json(path,input_dir,output_dir,images_dir)
            if row:
                rows.append(row)
        except Exception as exc:
            print(f"\nERROR: {path}")
            print(f"{type(exc).__name__}: {exc}")

    if rows:
        with open(output_dir/"summary.csv","w",newline="",encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with open(output_dir/"config.json","w",encoding="utf-8") as f:
        json.dump(CFG,f,ensure_ascii=False,indent=2)

    print()
    print("DONE")
    print(f"RESULT: {output_dir}")


if __name__ == "__main__":
    main()
