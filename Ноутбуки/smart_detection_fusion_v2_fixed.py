#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart detection fusion v2
-------------------------

Главная идея:
1) строим consensus heatmap от разных моделей;
2) извлекаем итоговые ОБЛАСТИ как connected components из consensus map;
3) для каждой итоговой области выбираем лучшую исходную рамку
   ИЛИ создаём bbox/polygon из самой consensus области;
4) сохраняем итог именно НА РЕАЛЬНОМ ФОТО.

Выход для каждого JSON:
    <base>__00_real.jpg
    <base>__01_all_models_on_real.jpg
    <base>__02_vote_heatmap.jpg
    <base>__03_staple_heatmap.jpg
    <base>__04_component_mask.jpg
    <base>__05_final_on_real.jpg
    <base>__06_final_on_white.jpg
    <base>__07_final_components_only.jpg
    <base>__99_compare.jpg
    <base>__final.json
    <base>__diagnostics.csv

Запуск:
    pip install numpy pillow opencv-python-headless tqdm

    python smart_detection_fusion_v2.py "D:\\path\\json_folder" ^
        --images-dir "D:\\path\\images"
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


CFG = {
    # -------------------- scoring --------------------
    "min_raw_score": 0.20,
    "raw_score_weight": 0.45,
    "rank_score_weight": 0.55,
    "provider_weights": {},

    # -------------------- mask resolution --------------------
    "mask_scale": 0.50,

    # -------------------- provider voting --------------------
    "pixel_presence_threshold": 0.38,

    # -------------------- component extraction --------------------
    # высокая и низкая граница hysteresis thresholding
    "high_thresh": 0.46,
    "low_thresh": 0.26,

    # минимальное число моделей в среднем / локально
    "min_component_support_mean": 1.30,
    "min_component_support_max": 2.0,

    # минимальный размер итоговой области
    "min_component_area": 24,
    "min_component_w": 6,
    "min_component_h": 6,

    # штраф за голые линии таблицы
    "max_rule_density_for_weak": 0.22,

    # textness
    "min_textness_general": 0.08,
    "min_textness_singleton": 0.28,

    # если у области одна модель
    "singleton_score_threshold": 0.91,

    # overlap для выбора лучшей исходной рамки под компонент
    "min_match_score": 0.16,

    # threshold для бинаризации provider>foreground
    "provider_fg_thr": 0.38,

    # отрисовка
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
# GEOMETRY
# ============================================================

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
    den = min(max(1e-6, a2 - a1), max(1e-6, b2 - b1))
    return inter / den


def line_affinity(a, b):
    A = a["bbox"]
    B = b["bbox"]

    w1 = max(1.0, A[2] - A[0])
    h1 = max(1.0, A[3] - A[1])
    w2 = max(1.0, B[2] - B[0])
    h2 = max(1.0, B[3] - B[1])

    i = iou(A, B)
    iom = io_min(A, B)
    xov = overlap_1d(A[0], A[2], B[0], B[2])
    yov = overlap_1d(A[1], A[3], B[1], B[3])

    cx1 = (A[0] + A[2]) / 2.0
    cy1 = (A[1] + A[3]) / 2.0
    cx2 = (B[0] + B[2]) / 2.0
    cy2 = (B[1] + B[3]) / 2.0

    xcenter = math.exp(-abs(cx1 - cx2) / max(w1, w2))
    ycenter = math.exp(-abs(cy1 - cy2) / max(h1, h2))

    wr = min(w1, w2) / max(w1, w2)
    hr = min(h1, h2) / max(h1, h2)

    horiz = max(w1 / h1, w2 / h2) >= 1.25

    if horiz:
        return (
            0.16 * i +
            0.20 * iom +
            0.10 * xov +
            0.22 * yov +
            0.15 * ycenter +
            0.09 * wr +
            0.08 * hr
        )
    else:
        return (
            0.16 * i +
            0.20 * iom +
            0.22 * xov +
            0.10 * yov +
            0.15 * xcenter +
            0.08 * wr +
            0.09 * hr
        )


# ============================================================
# PREDICTIONS
# ============================================================

def provider_weight(provider):
    return float(CFG["provider_weights"].get(provider, 1.0))


def percentile_ranks(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= 1:
        return np.ones_like(values)

    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks / float(len(values) - 1)


def flatten_predictions(data):
    out = []

    for provider, block in data.get("providers", {}).items():
        anns = block.get("annotations", [])
        raw_scores = [float(a.get("score", 0.0)) for a in anns]
        ranks = percentile_ranks(raw_scores)

        for ann, rank in zip(anns, ranks):
            raw = float(ann.get("score", 0.0))
            if raw < CFG["min_raw_score"]:
                continue

            polygon = ann.get("polygon")
            bbox_xywh = ann.get("bbox_xywh")

            if polygon:
                bbox = bbox_from_polygon(polygon)
            elif bbox_xywh:
                bbox = xywh_to_xyxy(bbox_xywh)
                polygon = rect_polygon(bbox)
            else:
                continue

            calibrated = (
                CFG["raw_score_weight"] * raw +
                CFG["rank_score_weight"] * float(rank)
            )
            calibrated *= provider_weight(provider)
            calibrated = float(np.clip(calibrated, 0.0, 1.0))

            out.append({
                "id": ann.get("id"),
                "provider": provider,
                "model": block.get("model", provider),
                "label": ann.get("label", "text"),
                "raw_score": raw,
                "rank_score": float(rank),
                "score": calibrated,
                "bbox": bbox,
                "polygon": [[float(x), float(y)] for x, y in polygon],
                "mask_rle": ann.get("mask_rle"),
            })

    return out


# ============================================================
# RLE / MASKS
# ============================================================

def decode_coco_rle(rle):
    if not rle:
        return None

    size = rle.get("size")
    counts = rle.get("counts")

    if not size or counts is None:
        return None

    h, w = map(int, size)

    if isinstance(counts, list):
        flat = np.zeros(h * w, dtype=np.uint8)
        pos = 0
        value = 0

        for run in counts:
            run = int(run)
            end = min(flat.size, pos + run)
            if value == 1:
                flat[pos:end] = 1
            pos = end
            value = 1 - value
            if pos >= flat.size:
                break

        return flat.reshape((h, w), order="F")

    try:
        from pycocotools import mask as mask_utils
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]
        return (decoded > 0).astype(np.uint8)
    except Exception:
        return None


def detection_mask(det, out_h, out_w, scale):
    rle_mask = decode_coco_rle(det.get("mask_rle"))
    if rle_mask is not None:
        return cv2.resize(
            rle_mask,
            (out_w, out_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)

    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    pts = np.asarray(
        [[round(x * scale), round(y * scale)] for x, y in det["polygon"]],
        dtype=np.int32,
    )

    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)

    return mask


# ============================================================
# IMAGE FINDING
# ============================================================

def candidate_image_paths(image_name, json_path, input_dir, images_dir):
    filename = Path(image_name).name if image_name else ""
    stem = Path(filename).stem if filename else ""

    paths = []

    def add(p):
        if p is not None:
            paths.append(Path(p))

    if images_dir:
        add(images_dir / image_name)
        add(images_dir / filename)
    add(json_path.parent / image_name)
    add(json_path.parent / filename)
    add(input_dir / image_name)
    add(input_dir / filename)
    add(input_dir / "images" / filename)
    add(input_dir.parent / "images" / filename)

    # exact hits
    for p in paths:
        if p and p.exists() and p.is_file():
            yield p

    # recursive exact filename
    search_roots = []
    if images_dir and Path(images_dir).exists():
        search_roots.append(Path(images_dir))
    search_roots.append(Path(input_dir))
    search_roots.append(Path(input_dir).parent)

    seen = set()
    for root in search_roots:
        if not root.exists():
            continue

        if filename:
            for p in root.rglob(filename):
                if p.is_file() and str(p) not in seen:
                    seen.add(str(p))
                    yield p

        # поиск по stem, если расширение/путь не совпали
        if stem:
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"]:
                p = root / f"{stem}{ext}"
                if p.exists() and p.is_file() and str(p) not in seen:
                    seen.add(str(p))
                    yield p

            for p in root.rglob("*"):
                if p.is_file() and p.stem == stem and p.suffix.lower() in {".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp"}:
                    if str(p) not in seen:
                        seen.add(str(p))
                        yield p


def cv2_imread_unicode(path: Path):
    """
    Unicode-safe чтение изображения на Windows.
    cv2.imread() часто ломается на путях с кириллицей.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None

        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def load_source_image(data, json_path, input_dir, images_dir):
    image_name = data.get("image_name", "")

    checked = []

    for p in candidate_image_paths(
        image_name,
        json_path,
        input_dir,
        images_dir,
    ):
        checked.append(str(p))

        img = cv2_imread_unicode(p)

        if img is not None:
            print(f"\nIMAGE FOUND: {p}")
            return img, p

    # Дополнительный fallback через PIL — тоже хорошо работает
    # с кириллицей на Windows.
    for p_str in checked:
        p = Path(p_str)

        try:
            pil = Image.open(p).convert("RGB")
            rgb = np.asarray(pil)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            print(f"\nIMAGE FOUND VIA PIL: {p}")
            return bgr, p

        except Exception:
            pass

    print()
    print("IMAGE NOT FOUND")
    print(f"image_name from JSON: {image_name}")
    print(f"images_dir: {images_dir}")

    if checked:
        print("Checked candidates:")
        for p in checked[:20]:
            print(f"  {p}")

    # fallback white canvas только если реально ничего не нашли
    w, h = data.get("image_size", [1440, 1172])

    canvas = np.full(
        (int(h), int(w), 3),
        255,
        dtype=np.uint8,
    )

    return canvas, None


# ============================================================
# PROVIDER MAPS / HEATMAPS
# ============================================================

def build_provider_maps(detections, H, W):
    scale = float(CFG["mask_scale"])
    mh = max(32, int(round(H * scale)))
    mw = max(32, int(round(W * scale)))

    providers = sorted(set(d["provider"] for d in detections))
    maps = {p: np.zeros((mh, mw), dtype=np.float32) for p in providers}
    masks = {}

    for idx, det in enumerate(detections):
        mask = detection_mask(det, mh, mw, scale)
        masks[idx] = mask

        pmap = maps[det["provider"]]
        np.maximum(pmap, mask.astype(np.float32) * float(det["score"]), out=pmap)

    return providers, maps, masks, (mh, mw)


def make_consensus_maps(providers, provider_maps):
    stack = np.stack([provider_maps[p] for p in providers], axis=0).astype(np.float32)
    R = stack.shape[0]
    weights = np.asarray([provider_weight(p) for p in providers], dtype=np.float32)
    weights = weights / max(weights.sum(), 1e-6)

    weighted_mean = np.tensordot(weights, stack, axes=(0, 0)).astype(np.float32)

    binary = stack >= CFG["provider_fg_thr"]
    support_count = binary.sum(axis=0).astype(np.float32)
    support_ratio = support_count / max(1.0, float(R))

    if R >= 2:
        part = np.partition(stack, kth=R - 2, axis=0)
        top2 = part[-2:, :, :].mean(axis=0)
    else:
        top2 = stack[0].copy()

    vote_map = (
        0.46 * weighted_mean +
        0.34 * top2 +
        0.20 * support_ratio
    )

    return {
        "stack": stack,
        "binary": binary,
        "support_count": support_count,
        "support_ratio": support_ratio,
        "weighted_mean": weighted_mean,
        "top2": top2,
        "vote_map": np.clip(vote_map, 0.0, 1.0),
    }


def staple_binary(binary_stack, max_iter=30, eps=1e-5):
    D = binary_stack.astype(np.float32)
    R, H, W = D.shape

    roi = D.any(axis=0)
    prob = np.zeros((H, W), dtype=np.float32)

    if roi.sum() == 0:
        return prob, {"sensitivity": [0.5] * R, "specificity": [0.5] * R, "prior": 0.5}

    X = D[:, roi]
    p = np.linspace(0.88, 0.94, R).astype(np.float64)
    q = np.linspace(0.96, 0.99, R).astype(np.float64)
    prior = float(np.clip(X.mean(), 0.03, 0.70))

    for _ in range(max_iter):
        p_old = p.copy()
        q_old = q.copy()

        pp = np.clip(p, 1e-4, 1.0 - 1e-4)[:, None]
        qq = np.clip(q, 1e-4, 1.0 - 1e-4)[:, None]

        log_fg = (
            math.log(max(prior, 1e-6)) +
            (X * np.log(pp) + (1.0 - X) * np.log(1.0 - pp)).sum(axis=0)
        )
        log_bg = (
            math.log(max(1.0 - prior, 1e-6)) +
            ((1.0 - X) * np.log(qq) + X * np.log(1.0 - qq)).sum(axis=0)
        )

        z = np.clip(log_bg - log_fg, -50.0, 50.0)
        Wfg = 1.0 / (1.0 + np.exp(z))

        sum_fg = max(Wfg.sum(), 1e-6)
        sum_bg = max((1.0 - Wfg).sum(), 1e-6)

        p = (X * Wfg[None, :]).sum(axis=1) / sum_fg
        q = ((1.0 - X) * (1.0 - Wfg)[None, :]).sum(axis=1) / sum_bg

        p = np.clip(0.10 + 0.80 * p, 0.05, 0.995)
        q = np.clip(0.10 + 0.80 * q, 0.05, 0.999)
        prior = float(np.clip(Wfg.mean(), 0.02, 0.80))

        delta = max(np.max(np.abs(p - p_old)), np.max(np.abs(q - q_old)))
        if delta < eps:
            break

    prob[roi] = Wfg.astype(np.float32)
    return prob, {
        "sensitivity": p.tolist(),
        "specificity": q.tolist(),
        "prior": prior,
    }


# ============================================================
# IMAGE EVIDENCE
# ============================================================

def build_image_evidence(image_bgr, target_hw):
    mh, mw = target_hw
    small = cv2.resize(image_bgr, (mw, mh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (0, 0), 7)
    norm = cv2.divide(gray, blur, scale=255)

    ink = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )

    hk = max(20, mw // 24)
    vk = max(20, mh // 24)

    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    )

    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    )

    rules = cv2.max(horizontal, vertical)
    rules = cv2.dilate(rules, np.ones((3, 3), np.uint8), iterations=1)

    residual = cv2.bitwise_and(ink, cv2.bitwise_not(rules))

    residual_soft = cv2.GaussianBlur((residual > 0).astype(np.float32), (0, 0), 1.2)

    return {
        "gray": gray,
        "ink": (ink > 0).astype(np.uint8),
        "rules": (rules > 0).astype(np.uint8),
        "residual": (residual > 0).astype(np.uint8),
        "residual_soft": np.clip(residual_soft, 0.0, 1.0).astype(np.float32),
    }


def image_textness(mask, evidence):
    pixels = mask > 0
    area = int(pixels.sum())
    if area < 5:
        return {
            "textness": 0.0,
            "ink_density": 0.0,
            "residual_density": 0.0,
            "rule_density": 0.0,
            "residual_share": 0.0,
        }

    ink = evidence["ink"][pixels].astype(np.float32)
    residual = evidence["residual"][pixels].astype(np.float32)
    rules = evidence["rules"][pixels].astype(np.float32)

    ink_density = float(ink.mean())
    residual_density = float(residual.mean())
    rule_density = float(rules.mean())

    ink_sum = float(ink.sum())
    residual_sum = float(residual.sum())

    residual_share = residual_sum / max(1.0, ink_sum)
    occupancy = min(1.0, residual_density / 0.035)

    textness = (0.55 * occupancy + 0.45 * residual_share) * math.exp(-2.6 * rule_density)

    return {
        "textness": float(np.clip(textness, 0.0, 1.0)),
        "ink_density": ink_density,
        "residual_density": residual_density,
        "rule_density": rule_density,
        "residual_share": residual_share,
    }


# ============================================================
# COMPONENT EXTRACTION FROM CONSENSUS MAP
# ============================================================

def hysteresis_components(score_map, seed_mask, grow_mask):
    """
    Берём компоненты из grow_mask, оставляем только те,
    которые содержат хотя бы один seed pixel.
    """
    grow_u8 = (grow_mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(grow_u8, connectivity=8)

    keep = np.zeros_like(grow_u8)

    for lab in range(1, num):
        comp = labels == lab
        if not comp.any():
            continue
        if np.any(seed_mask[comp] > 0):
            keep[comp] = 1

    return keep, labels


def component_to_polygon(component_mask):
    ys, xs = np.where(component_mask > 0)
    if len(xs) == 0:
        return None, None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    bbox = [x1, y1, x2, y2]

    cnts, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return rect_polygon(bbox), bbox

    cnt = max(cnts, key=cv2.contourArea)
    epsilon = max(1.0, 0.005 * cv2.arcLength(cnt, True))
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    poly = [[float(p[0][0]), float(p[0][1])] for p in approx]
    if len(poly) < 3:
        poly = rect_polygon(bbox)

    return poly, bbox


def extract_consensus_components(consensus, staple_prob, evidence):
    vote_map = consensus["vote_map"]
    support_count = consensus["support_count"]
    support_ratio = consensus["support_ratio"]

    fusion_map = (
        0.42 * vote_map +
        0.26 * staple_prob +
        0.20 * evidence["residual_soft"] +
        0.12 * support_ratio
    )
    fusion_map = np.clip(fusion_map, 0.0, 1.0)

    # Seeds: уверенные пиксели, поддержанные моделями и не являющиеся чистой линией
    seed = (
        (fusion_map >= CFG["high_thresh"]) &
        ((support_count >= 2) | (staple_prob >= 0.48))
    )

    # Grow: можно дорастать слабее, если там есть хоть какое-то consensus signal
    grow = (
        (fusion_map >= CFG["low_thresh"]) &
        (
            (support_count >= 1) |
            (evidence["residual_soft"] >= 0.10)
        )
    )

    # Небольшая морфология, чтобы соединить чуть разорванные строки
    seed = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    grow = cv2.morphologyEx(grow.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    keep_mask, labels = hysteresis_components(fusion_map, seed, grow)

    num, comp_labels, stats, _ = cv2.connectedComponentsWithStats(keep_mask.astype(np.uint8), connectivity=8)

    components = []

    for lab in range(1, num):
        comp = comp_labels == lab
        area = int(comp.sum())

        if area < CFG["min_component_area"]:
            continue

        ys, xs = np.where(comp)
        if len(xs) == 0:
            continue

        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1

        w = x2 - x1
        h = y2 - y1

        if w < CFG["min_component_w"] or h < CFG["min_component_h"]:
            continue

        txt = image_textness(comp.astype(np.uint8), evidence)

        support_vals = support_count[comp]
        support_mean = float(support_vals.mean()) if support_vals.size else 0.0
        support_max = float(support_vals.max()) if support_vals.size else 0.0

        vote_vals = vote_map[comp]
        fusion_vals = fusion_map[comp]
        staple_vals = staple_prob[comp]

        vote_mean = float(vote_vals.mean()) if vote_vals.size else 0.0
        vote_q75 = float(np.quantile(vote_vals, 0.75)) if vote_vals.size else 0.0
        fusion_mean = float(fusion_vals.mean()) if fusion_vals.size else 0.0
        fusion_q75 = float(np.quantile(fusion_vals, 0.75)) if fusion_vals.size else 0.0
        staple_q75 = float(np.quantile(staple_vals, 0.75)) if staple_vals.size else 0.0

        # фильтрация слабых областей
        strong_singleton_like = (
            fusion_q75 >= 0.55 and
            txt["textness"] >= CFG["min_textness_singleton"]
        )

        enough_support = (
            support_mean >= CFG["min_component_support_mean"] or
            support_max >= CFG["min_component_support_max"]
        )

        if not enough_support and not strong_singleton_like:
            continue

        if txt["textness"] < CFG["min_textness_general"] and not strong_singleton_like:
            continue

        if txt["rule_density"] > CFG["max_rule_density_for_weak"] and support_max < 3:
            continue

        poly_small, bbox_small = component_to_polygon(comp.astype(np.uint8))
        if poly_small is None:
            continue

        components.append({
            "component_id": len(components),
            "mask_small": comp.astype(np.uint8),
            "bbox_small": bbox_small,
            "polygon_small": poly_small,
            "area_small": area,
            "support_mean": support_mean,
            "support_max": support_max,
            "vote_mean": vote_mean,
            "vote_q75": vote_q75,
            "fusion_mean": fusion_mean,
            "fusion_q75": fusion_q75,
            "staple_q75": staple_q75,
            **txt,
        })

    return components, fusion_map, keep_mask


# ============================================================
# MATCH COMPONENTS TO ORIGINAL DETECTIONS
# ============================================================

def upscale_polygon(poly, inv_scale):
    return [[float(x * inv_scale), float(y * inv_scale)] for x, y in poly]


def upscale_bbox(bbox, inv_scale):
    return [float(v * inv_scale) for v in bbox]


def component_support_by_provider(component_mask, providers, provider_maps):
    supported = []
    overlaps = {}

    for p in providers:
        pm = provider_maps[p]
        vals = pm[component_mask > 0]

        if vals.size == 0:
            overlaps[p] = 0.0
            continue

        q75 = float(np.quantile(vals, 0.75))
        overlaps[p] = q75

        if q75 >= CFG["provider_fg_thr"] * 0.9:
            supported.append(p)

    return supported, overlaps


def match_detection_to_component(det, component_bbox_full, component_poly_full):
    det_box = det["bbox"]
    comp_box = component_bbox_full

    aff = line_affinity(
        {"bbox": det_box},
        {"bbox": comp_box},
    )

    i = iou(det_box, comp_box)
    iom = io_min(det_box, comp_box)

    # bbox of component approximates consensus area
    score = (
        0.36 * aff +
        0.26 * iom +
        0.18 * i +
        0.20 * det["score"]
    )
    return float(score)


def choose_final_regions(components, detections, providers, provider_maps, mask_scale):
    inv_scale = 1.0 / float(mask_scale)
    finals = []
    used_detection_keys = set()

    # Компоненты сортируем по качеству consensus
    components = sorted(
        components,
        key=lambda c: (
            c["support_max"],
            c["fusion_q75"],
            c["textness"],
            c["area_small"],
        ),
        reverse=True,
    )

    for comp in components:
        supported, provider_overlap = component_support_by_provider(
            comp["mask_small"], providers, provider_maps
        )

        comp_bbox_full = upscale_bbox(comp["bbox_small"], inv_scale)
        comp_poly_full = upscale_polygon(comp["polygon_small"], inv_scale)

        best = None
        best_score = -1.0

        for det in detections:
            key = (det["provider"], det.get("id"), tuple(map(int, det["bbox"])))
            if key in used_detection_keys:
                continue

            s = match_detection_to_component(det, comp_bbox_full, comp_poly_full)

            # Бонусы за то, что рамка пришла от поддержанного provider
            if det["provider"] in supported:
                s += 0.06

            # Бонус если сам детектор уверенный
            s += 0.10 * det["score"]

            if s > best_score:
                best_score = s
                best = det

        use_original = best is not None and best_score >= CFG["min_match_score"]

        region = {
            "component_id": comp["component_id"],
            "support_models": len(supported),
            "support_providers": supported,
            "provider_overlap": provider_overlap,
            "component_score": float(
                0.32 * min(1.0, comp["support_max"] / max(1, len(providers))) +
                0.24 * comp["fusion_q75"] +
                0.18 * comp["vote_q75"] +
                0.14 * comp["staple_q75"] +
                0.12 * comp["textness"]
            ),
            "textness": comp["textness"],
            "ink_density": comp["ink_density"],
            "residual_density": comp["residual_density"],
            "rule_density": comp["rule_density"],
            "fusion_q75": comp["fusion_q75"],
            "vote_q75": comp["vote_q75"],
            "staple_q75": comp["staple_q75"],
            "component_bbox": comp_bbox_full,
            "component_polygon": comp_poly_full,
        }

        if use_original:
            key = (best["provider"], best.get("id"), tuple(map(int, best["bbox"])))
            used_detection_keys.add(key)

            item = copy.deepcopy(best)
            item.update(region)
            item["selected_mode"] = "best_original_detection"
            item["match_score"] = float(best_score)
            item["final_score"] = float(
                0.48 * item["score"] +
                0.34 * region["component_score"] +
                0.18 * min(1.0, len(supported) / max(1, len(providers)))
            )
            finals.append(item)
        else:
            item = {
                "id": f"consensus_component_{comp['component_id']:04d}",
                "provider": "consensus_component",
                "model": "consensus_component",
                "label": "text",
                "raw_score": float(region["component_score"]),
                "rank_score": float(region["component_score"]),
                "score": float(region["component_score"]),
                "bbox": comp_bbox_full,
                "polygon": comp_poly_full,
                **region,
                "selected_mode": "component_bbox",
                "match_score": float(best_score),
                "final_score": float(region["component_score"]),
            }
            finals.append(item)

    # Удаление дублей уже на финальном списке
    finals = sorted(finals, key=lambda d: d["final_score"], reverse=True)
    unique = []

    for cand in finals:
        duplicate = False
        for kept in unique:
            aff = line_affinity(cand, kept)
            iom = io_min(cand["bbox"], kept["bbox"])
            if aff >= 0.60 or (iom >= 0.74 and aff >= 0.40):
                duplicate = True
                break
        if not duplicate:
            unique.append(cand)

    return unique


# ============================================================
# VISUALIZATION
# ============================================================

def color_map_for_providers(providers):
    return {p: MODEL_COLORS[i % len(MODEL_COLORS)] for i, p in enumerate(providers)}


def draw_regions(image_bgr, regions, title, color=(255, 0, 0), color_by_provider=False):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)

    providers = sorted(set(d["provider"] for d in regions)) if regions else []
    cmap = color_map_for_providers(providers)

    for idx, d in enumerate(regions):
        poly = [(int(round(x)), int(round(y))) for x, y in d["polygon"]]
        if not poly:
            continue

        c = cmap.get(d["provider"], color) if color_by_provider else color

        if len(poly) >= 2:
            draw.line(poly + [poly[0]], fill=c, width=CFG["draw_width"])

        if CFG["draw_labels"]:
            score = d.get("final_score", d.get("score", 0.0))
            support = d.get("support_models", 1)
            txt = f"{idx} {d['provider']} {score:.2f} [{support}]"
            x, y = poly[0]
            draw.text((x + 2, max(2, y - 13)), txt, fill=c)

    draw.rectangle([0, 0, min(canvas.width, 1100), 24], fill=(255, 255, 255))
    draw.text((5, 5), f"{title} | detections={len(regions)}", fill=(0, 0, 0))
    return np.asarray(canvas)


def draw_component_mask(mask_small, image_bgr):
    H, W = image_bgr.shape[:2]
    mask_big = cv2.resize(mask_small.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST)
    overlay = image_bgr.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    alpha = (mask_big > 0).astype(np.float32) * 0.28
    overlay = (overlay * (1.0 - alpha[..., None]) + red * alpha[..., None]).astype(np.uint8)

    cv2.rectangle(overlay, (0, 0), (min(W - 1, 1100), 24), (255, 255, 255), -1)
    cv2.putText(
        overlay,
        "04_component_mask: consensus-connected areas",
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def heatmap_overlay(image_bgr, heatmap_small, title):
    H, W = image_bgr.shape[:2]
    heat = cv2.resize(heatmap_small, (W, H), interpolation=cv2.INTER_LINEAR)
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(image_bgr, 0.68, colored, 0.32, 0)
    cv2.rectangle(overlay, (0, 0), (min(W - 1, 1100), 24), (255, 255, 255), -1)
    cv2.putText(
        overlay,
        title,
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def make_grid(images, labels, target_width=720):
    tiles = []

    for img, label in zip(images, labels):
        pil = Image.fromarray(img)
        ratio = target_width / pil.width
        h = max(1, int(round(pil.height * ratio)))
        pil = pil.resize((target_width, h), Image.Resampling.LANCZOS)

        tile = Image.new("RGB", (target_width, h + 34), (255, 255, 255))
        tile.paste(pil, (0, 34))

        draw = ImageDraw.Draw(tile)
        draw.text((8, 10), label, fill=(0, 0, 0))
        tiles.append(tile)

    cols = 2
    rows = math.ceil(len(tiles) / cols)
    cell_w = max(t.width for t in tiles)
    cell_h = max(t.height for t in tiles)

    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (235, 235, 235))
    for i, tile in enumerate(tiles):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        grid.paste(tile, (x, y))

    return np.asarray(grid)


# ============================================================
# EXPORT
# ============================================================

def safe_base_name(json_path, image_name):
    image_stem = Path(image_name).stem if image_name else json_path.stem
    stem = json_path.stem
    return image_stem if image_stem == stem else f"{stem}__{image_stem}"


def export_final_json(path, data, regions):
    payload = {
        "run_id": data.get("run_id"),
        "image_name": data.get("image_name"),
        "image_size": data.get("image_size"),
        "fusion_method": "consensus_components_v2",
        "annotations": [],
    }

    for i, d in enumerate(regions):
        payload["annotations"].append({
            "id": d.get("id", f"final-{i:04d}"),
            "label": d.get("label", "text"),
            "score": float(d.get("final_score", d.get("score", 0.0))),
            "raw_score": float(d.get("raw_score", 0.0)),
            "bbox_xywh": xyxy_to_xywh(d["bbox"]),
            "polygon": d["polygon"],
            "selected_mode": d.get("selected_mode"),
            "source_provider": d.get("provider"),
            "support_models": int(d.get("support_models", 1)),
            "support_providers": d.get("support_providers", []),
            "component_score": d.get("component_score"),
            "match_score": d.get("match_score"),
            "textness": d.get("textness"),
            "rule_density": d.get("rule_density"),
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ============================================================
# PROCESS ONE
# ============================================================

def process_json(json_path, input_dir, output_dir, images_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data.get("providers"), dict):
        return None

    detections = flatten_predictions(data)
    if not detections:
        return None

    image_bgr, image_path = load_source_image(
        data,
        json_path,
        input_dir,
        images_dir,
    )

    if image_path is None:
        print(
            f"WARNING: real image not found for JSON: {json_path.name}"
        )

    H, W = image_bgr.shape[:2]

    providers, provider_maps, det_masks, mask_hw = build_provider_maps(detections, H, W)
    consensus = make_consensus_maps(providers, provider_maps)
    staple_prob, staple_info = staple_binary(consensus["binary"])
    evidence = build_image_evidence(image_bgr, mask_hw)

    components, fusion_map, keep_mask = extract_consensus_components(consensus, staple_prob, evidence)
    finals = choose_final_regions(components, detections, providers, provider_maps, CFG["mask_scale"])

    base = safe_base_name(json_path, data.get("image_name", ""))

    # Save original real image
    Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).save(
        output_dir / f"{base}__00_real.jpg", quality=95
    )

    all_on_real = draw_regions(
        image_bgr,
        detections,
        "01_all_models_on_real",
        color_by_provider=True,
    )
    Image.fromarray(all_on_real).save(output_dir / f"{base}__01_all_models_on_real.jpg", quality=94)

    vote_img = heatmap_overlay(image_bgr, consensus["vote_map"], "02_vote_heatmap: provider-level agreement")
    Image.fromarray(vote_img).save(output_dir / f"{base}__02_vote_heatmap.jpg", quality=94)

    staple_img = heatmap_overlay(image_bgr, staple_prob, "03_staple_heatmap: reliability-weighted consensus")
    Image.fromarray(staple_img).save(output_dir / f"{base}__03_staple_heatmap.jpg", quality=94)

    comp_img = draw_component_mask(keep_mask, image_bgr)
    Image.fromarray(comp_img).save(output_dir / f"{base}__04_component_mask.jpg", quality=94)

    final_on_real = draw_regions(
        image_bgr,
        finals,
        "05_final_on_real",
        color=(255, 0, 0),
        color_by_provider=False,
    )
    Image.fromarray(final_on_real).save(output_dir / f"{base}__05_final_on_real.jpg", quality=94)

    white_canvas = np.full_like(image_bgr, 255)
    final_on_white = draw_regions(
        white_canvas,
        finals,
        "06_final_on_white",
        color=(255, 0, 0),
        color_by_provider=False,
    )
    Image.fromarray(final_on_white).save(output_dir / f"{base}__06_final_on_white.jpg", quality=94)

    # components only with component polygons, useful for debug
    comp_regions = []
    inv = 1.0 / float(CFG["mask_scale"])
    for c in components:
        comp_regions.append({
            "id": f"component_{c['component_id']}",
            "provider": "component",
            "score": c["fusion_q75"],
            "final_score": c["fusion_q75"],
            "support_models": int(round(c["support_max"])),
            "bbox": upscale_bbox(c["bbox_small"], inv),
            "polygon": upscale_polygon(c["polygon_small"], inv),
        })

    comp_only = draw_regions(
        image_bgr,
        comp_regions,
        "07_final_components_only",
        color=(0, 140, 255),
        color_by_provider=False,
    )
    Image.fromarray(comp_only).save(output_dir / f"{base}__07_final_components_only.jpg", quality=94)

    compare = make_grid(
        [
            final_on_real,
            all_on_real,
            vote_img,
            staple_img,
            comp_img,
            final_on_white,
        ],
        [
            f"05_final_on_real: {len(finals)}",
            f"01_all_models_on_real: {len(detections)}",
            "02_vote_heatmap",
            "03_staple_heatmap",
            f"04_component_mask: {len(components)} components",
            f"06_final_on_white: {len(finals)}",
        ],
        target_width=720,
    )
    Image.fromarray(compare).save(output_dir / f"{base}__99_compare.jpg", quality=92)

    export_final_json(output_dir / f"{base}__final.json", data, finals)

    # diagnostics
    with open(output_dir / f"{base}__diagnostics.csv", "w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "component_id",
            "support_models",
            "component_score",
            "textness",
            "fusion_q75",
            "vote_q75",
            "staple_q75",
            "rule_density",
            "selected_mode",
            "source_provider",
            "final_score",
            "match_score",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for d in finals:
            writer.writerow({
                "component_id": d.get("component_id"),
                "support_models": d.get("support_models"),
                "component_score": d.get("component_score"),
                "textness": d.get("textness"),
                "fusion_q75": d.get("fusion_q75"),
                "vote_q75": d.get("vote_q75"),
                "staple_q75": d.get("staple_q75"),
                "rule_density": d.get("rule_density"),
                "selected_mode": d.get("selected_mode"),
                "source_provider": d.get("provider"),
                "final_score": d.get("final_score"),
                "match_score": d.get("match_score"),
                "bbox_x1": d["bbox"][0],
                "bbox_y1": d["bbox"][1],
                "bbox_x2": d["bbox"][2],
                "bbox_y2": d["bbox"][3],
            })

    with open(output_dir / f"{base}__staple_reliability.json", "w", encoding="utf-8") as f:
        json.dump({"providers": providers, "staple": staple_info, "image_found": str(image_path) if image_path else ""}, f, ensure_ascii=False, indent=2)

    return {
        "json": str(json_path),
        "image_name": data.get("image_name", ""),
        "image_found": str(image_path) if image_path else "",
        "raw_detections": len(detections),
        "components": len(components),
        "final_regions": len(finals),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="Папка с JSON")
    parser.add_argument("--images-dir", type=Path, default=None, help="Папка с исходными изображениями")
    parser.add_argument("--output-dir", type=Path, default=None, help="Папка результата")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    images_dir = args.images_dir.resolve() if args.images_dir else None

    output_dir = args.output_dir.resolve() if args.output_dir else (input_dir.parent / f"{input_dir.name}_smart_compare_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.rglob("*.json"))

    print(f"JSON:   {len(json_files)}")
    print(f"OUTPUT: {output_dir}")
    if images_dir:
        print(f"IMAGES: {images_dir}")
    print()

    rows = []
    for path in tqdm(json_files, desc="Smart fusion v2"):
        try:
            row = process_json(path, input_dir, output_dir, images_dir)
            if row is not None:
                rows.append(row)
        except Exception as exc:
            print()
            print(f"ERROR: {path}")
            print(f"{type(exc).__name__}: {exc}")

    if rows:
        with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(CFG, f, ensure_ascii=False, indent=2)

    print()
    print("DONE")
    print(f"RESULT: {output_dir}")


if __name__ == "__main__":
    main()
