#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SMART ensemble для text-line detection.

Идея:
1) НЕ считаем число bbox напрямую.
   Каждая модель имеет максимум ОДИН голос на пиксель.
   Это не даёт PP-OCR с сотнями bbox задавить остальные модели.

2) Используем одновременно:
   - confidence;
   - rank-normalized confidence внутри каждой модели;
   - число РАЗНЫХ моделей, поддерживающих область;
   - pixel/mask consensus;
   - STAPLE-подобный reliability-weighted consensus;
   - содержимое исходного изображения:
       * наличие чернил;
       * штраф за длинные прямые линии таблицы.

3) Если в JSON есть mask_rle — используется настоящая mask.
   Если mask_rle нет — polygon превращается в mask.

4) Все результаты для одного изображения сохраняются В ОДНУ ПАПКУ:
   image__00_all_models.jpg
   image__01_source_vote.jpg
   image__02_pixel_vote.jpg
   image__03_staple_vote.jpg
   image__04_ink_aware.jpg
   image__05_hybrid_smart.jpg
   image__06_vote_heatmap.jpg
   image__99_COMPARE.jpg

Запуск:
    pip install numpy pillow opencv-python tqdm

    python smart_detection_fusion.py "D:\\path\\json_folder" --images-dir "D:\\path\\images"

По умолчанию результат:
    <json_folder>_smart_compare
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


# ============================================================
# CONFIG
# ============================================================

CFG = {
    # ---------- score ----------
    "min_raw_score": 0.20,

    # Без GT это proxy-calibration:
    # raw score + percentile score внутри конкретной модели.
    "raw_score_weight": 0.45,
    "rank_score_weight": 0.55,

    # Можно вручную усилить/ослабить модель.
    # Пример:
    # "rfdetr_historical": 1.10
    "provider_weights": {},

    # ---------- mask ----------
    # Маски строятся в уменьшенном разрешении для скорости.
    "mask_scale": 0.50,

    # С какого score считать, что модель "голосует"
    # за пиксель как foreground.
    "pixel_presence_threshold": 0.38,

    # ---------- object matching ----------
    # Насколько bbox/polygon разных моделей должны быть похожи,
    # чтобы считать их голосами за одну область.
    "vote_affinity_threshold": 0.34,

    # После ранжирования оставляем только одну границу
    # из близких кандидатов.
    "duplicate_affinity_threshold": 0.53,

    # ---------- filtering ----------
    # Обычный случай: желательно минимум 2 РАЗНЫХ модели.
    "preferred_support_models": 2,

    # Одиночное предсказание может пройти только если оно
    # очень сильное + есть image evidence.
    "singleton_score_threshold": 0.91,

    # ---------- image evidence ----------
    # Минимальный textness для слабого singleton.
    "singleton_textness_threshold": 0.30,

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

def xywh_to_xyxy(box):
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


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
    """
    Сходство для line-like областей.

    Важное отличие от старого варианта:
    добавлена совместимость размеров, чтобы гигантский bbox,
    содержащий маленькую строку, не считался идеальным совпадением.
    """
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

    cy1 = (A[1] + A[3]) / 2.0
    cy2 = (B[1] + B[3]) / 2.0
    cx1 = (A[0] + A[2]) / 2.0
    cx2 = (B[0] + B[2]) / 2.0

    ycenter = math.exp(-abs(cy1 - cy2) / max(h1, h2))
    xcenter = math.exp(-abs(cx1 - cx2) / max(w1, w2))

    wr = min(w1, w2) / max(w1, w2)
    hr = min(h1, h2) / max(h1, h2)

    # Ориентационно-адаптивно:
    # горизонтальные строки сильнее требуют совпадения по Y,
    # вертикальные — по X.
    horiz = max(w1 / h1, w2 / h2) >= 1.25

    if horiz:
        return (
            0.18 * i +
            0.22 * iom +
            0.08 * xov +
            0.22 * yov +
            0.15 * ycenter +
            0.08 * wr +
            0.07 * hr
        )
    else:
        return (
            0.18 * i +
            0.22 * iom +
            0.22 * xov +
            0.08 * yov +
            0.15 * xcenter +
            0.07 * wr +
            0.08 * hr
        )


# ============================================================
# SCORE NORMALIZATION / CALIBRATION PROXY
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
# COCO RLE + POLYGON MASKS
# ============================================================

def decode_coco_rle(rle):
    """
    Поддерживает uncompressed COCO RLE, где counts = list[int].
    Для compressed-string RLE пытается использовать pycocotools.
    """
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

            if run < 0:
                return None

            end = min(flat.size, pos + run)

            if value == 1:
                flat[pos:end] = 1

            pos = end
            value = 1 - value

            if pos >= flat.size:
                break

        # COCO RLE is column-major.
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
    """
    Возвращает binary mask конкретной детекции
    в рабочем уменьшенном разрешении.
    """
    rle_mask = decode_coco_rle(det.get("mask_rle"))

    if rle_mask is not None:
        return cv2.resize(
            rle_mask,
            (out_w, out_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)

    mask = np.zeros((out_h, out_w), dtype=np.uint8)

    pts = np.asarray(
        [
            [round(x * scale), round(y * scale)]
            for x, y in det["polygon"]
        ],
        dtype=np.int32,
    )

    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 1)

    return mask


# ============================================================
# PROVIDER-LEVEL PIXEL VOTING
# ============================================================

def build_provider_maps(detections, image_h, image_w):
    """
    Главное правило:
        один provider = максимум один голос на пиксель.

    Если одна модель выдала 100 пересекающихся bbox,
    они НЕ суммируются 100 раз.
    Берётся max confidence этого provider в пикселе.
    """
    scale = float(CFG["mask_scale"])
    mh = max(32, int(round(image_h * scale)))
    mw = max(32, int(round(image_w * scale)))

    providers = sorted(set(d["provider"] for d in detections))

    maps = {
        p: np.zeros((mh, mw), dtype=np.float32)
        for p in providers
    }

    det_masks = {}

    for idx, det in enumerate(detections):
        mask = detection_mask(
            det,
            mh,
            mw,
            scale,
        )

        det_masks[idx] = mask

        pmap = maps[det["provider"]]

        # max, а не sum!
        np.maximum(
            pmap,
            mask.astype(np.float32) * float(det["score"]),
            out=pmap,
        )

    return providers, maps, det_masks, (mh, mw)


def make_consensus_maps(providers, provider_maps):
    stack = np.stack(
        [provider_maps[p] for p in providers],
        axis=0,
    ).astype(np.float32)

    R = stack.shape[0]
    presence_thr = CFG["pixel_presence_threshold"]

    binary = stack >= presence_thr
    support_count = binary.sum(axis=0).astype(np.float32)

    weights = np.asarray(
        [provider_weight(p) for p in providers],
        dtype=np.float32,
    )

    weights = weights / max(weights.sum(), 1e-6)

    weighted_mean = np.tensordot(
        weights,
        stack,
        axes=(0, 0),
    ).astype(np.float32)

    # Top-2 consensus:
    # высокий только там, где есть хотя бы две сильные модели.
    if R >= 2:
        part = np.partition(stack, kth=R - 2, axis=0)
        top2 = part[-2:, :, :].mean(axis=0)
    else:
        top2 = stack[0] * 0.5

    support_ratio = support_count / max(1.0, float(R))

    # Усиливаем confidence там, где есть независимые модели.
    vote_map = (
        0.45 * weighted_mean +
        0.35 * top2 +
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


# ============================================================
# STAPLE-LIKE EM
# ============================================================

def staple_binary(binary_stack, max_iter=35, eps=1e-5):
    """
    Lightweight STAPLE-like EM.

    Input:
        [R, H, W] binary masks, один mask на provider.

    Выход:
        probability map и reliability parameters.

    Важно:
        EM запускается только внутри union ROI,
        чтобы огромный background не задавил foreground.
    """
    D = binary_stack.astype(np.float32)
    R, H, W = D.shape

    roi = D.any(axis=0)

    prob = np.zeros((H, W), dtype=np.float32)

    if roi.sum() == 0:
        return prob, {
            "sensitivity": [0.5] * R,
            "specificity": [0.5] * R,
        }

    X = D[:, roi]  # R x N

    # Небольшое различие инициализации помогает не застрять
    # в полностью симметричном решении.
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
            (
                X * np.log(pp) +
                (1.0 - X) * np.log(1.0 - pp)
            ).sum(axis=0)
        )

        log_bg = (
            math.log(max(1.0 - prior, 1e-6)) +
            (
                (1.0 - X) * np.log(qq) +
                X * np.log(1.0 - qq)
            ).sum(axis=0)
        )

        z = np.clip(log_bg - log_fg, -50.0, 50.0)
        Wfg = 1.0 / (1.0 + np.exp(z))

        sum_fg = max(Wfg.sum(), 1e-6)
        sum_bg = max((1.0 - Wfg).sum(), 1e-6)

        p = (X * Wfg[None, :]).sum(axis=1) / sum_fg
        q = (
            (1.0 - X) *
            (1.0 - Wfg)[None, :]
        ).sum(axis=1) / sum_bg

        # Beta-like regularization:
        # не позволяем одной странице дать экстремальные 0/1.
        p = np.clip(0.10 + 0.80 * p, 0.05, 0.995)
        q = np.clip(0.10 + 0.80 * q, 0.05, 0.999)

        prior = float(np.clip(Wfg.mean(), 0.02, 0.80))

        delta = max(
            np.max(np.abs(p - p_old)),
            np.max(np.abs(q - q_old)),
        )

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
    """
    Строим:
        ink       - всё тёмное содержимое;
        rules     - длинные прямые horizontal/vertical линии;
        residual  - ink без длинных прямых линий.

    Это помогает отличать строку текста от голой линии таблицы.
    """
    mh, mw = target_hw

    small = cv2.resize(
        image_bgr,
        (mw, mh),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY,
    )

    # Нормализация неравномерного исторического фона.
    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        7,
    )

    norm = cv2.divide(
        gray,
        blur,
        scale=255,
    )

    ink = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    # Убираем совсем мелкую одиночную пыль.
    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )

    hk = max(20, mw // 24)
    vk = max(20, mh // 24)

    hor_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (hk, 1),
    )
    ver_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vk),
    )

    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        hor_kernel,
    )

    vertical = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        ver_kernel,
    )

    rules = cv2.max(
        horizontal,
        vertical,
    )

    # Чуть расширяем rule mask,
    # чтобы убрать края толстой линии.
    rules_dilated = cv2.dilate(
        rules,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    residual = cv2.bitwise_and(
        ink,
        cv2.bitwise_not(rules_dilated),
    )

    return {
        "gray": gray,
        "ink": (ink > 0).astype(np.uint8),
        "rules": (rules_dilated > 0).astype(np.uint8),
        "residual": (residual > 0).astype(np.uint8),
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

    residual_share = (
        residual_sum / max(1.0, ink_sum)
    )

    # 3% residual ink уже заметный сигнал для широкой line-mask.
    occupancy = min(
        1.0,
        residual_density / 0.035,
    )

    # Прямая линия без букв:
    # residual_share низкий, rule_density высокий.
    textness = (
        0.55 * occupancy +
        0.45 * residual_share
    )

    textness *= math.exp(
        -2.5 * rule_density
    )

    return {
        "textness": float(np.clip(textness, 0.0, 1.0)),
        "ink_density": ink_density,
        "residual_density": residual_density,
        "rule_density": rule_density,
        "residual_share": residual_share,
    }


# ============================================================
# CANDIDATE FEATURES
# ============================================================

def provider_object_votes(candidate, detections, providers):
    """
    Каждый provider даёт максимум один object-level vote.
    """
    best_by_provider = {}

    for other in detections:
        p = other["provider"]
        a = line_affinity(candidate, other)

        current = best_by_provider.get(p)

        value = a * other["score"]

        if current is None or value > current[0]:
            best_by_provider[p] = (
                value,
                a,
                other["score"],
            )

    vote_thr = CFG["vote_affinity_threshold"]

    supported = []

    vote_strengths = []

    for p in providers:
        info = best_by_provider.get(p)

        if info is None:
            continue

        value, affinity, score = info

        if affinity >= vote_thr:
            supported.append(p)
            vote_strengths.append(
                affinity * score
            )

    support_count = len(supported)
    support_ratio = support_count / max(1, len(providers))

    vote_strength = (
        float(np.mean(vote_strengths))
        if vote_strengths
        else 0.0
    )

    return {
        "support_models": support_count,
        "support_ratio": support_ratio,
        "support_providers": supported,
        "object_vote_strength": vote_strength,
    }


def masked_stats(values, mask):
    pix = values[mask > 0]

    if pix.size == 0:
        return {
            "mean": 0.0,
            "q75": 0.0,
            "q90": 0.0,
            "coverage_025": 0.0,
            "coverage_050": 0.0,
        }

    return {
        "mean": float(pix.mean()),
        "q75": float(np.quantile(pix, 0.75)),
        "q90": float(np.quantile(pix, 0.90)),
        "coverage_025": float((pix >= 0.25).mean()),
        "coverage_050": float((pix >= 0.50).mean()),
    }


def compute_candidate_features(
    detections,
    providers,
    det_masks,
    consensus,
    staple_prob,
    evidence,
):
    featured = []

    for idx, det in enumerate(detections):
        d = copy.deepcopy(det)
        mask = det_masks[idx]

        obj = provider_object_votes(
            d,
            detections,
            providers,
        )

        vote_stats = masked_stats(
            consensus["vote_map"],
            mask,
        )

        top2_stats = masked_stats(
            consensus["top2"],
            mask,
        )

        staple_stats = masked_stats(
            staple_prob,
            mask,
        )

        support_pixels = consensus["support_count"][mask > 0]

        if support_pixels.size:
            px_support2 = float(
                (support_pixels >= 2).mean()
            )
            px_support3 = float(
                (support_pixels >= 3).mean()
            )
            px_support_mean = float(
                support_pixels.mean()
            )
        else:
            px_support2 = 0.0
            px_support3 = 0.0
            px_support_mean = 0.0

        text = image_textness(
            mask,
            evidence,
        )

        d.update(obj)

        d.update({
            "mask_vote_mean": vote_stats["mean"],
            "mask_vote_q75": vote_stats["q75"],
            "mask_vote_q90": vote_stats["q90"],

            "top2_mean": top2_stats["mean"],
            "top2_q75": top2_stats["q75"],

            "staple_mean": staple_stats["mean"],
            "staple_q75": staple_stats["q75"],

            "pixel_support2": px_support2,
            "pixel_support3": px_support3,
            "pixel_support_mean": px_support_mean,

            **text,
        })

        featured.append(d)

    return featured


# ============================================================
# SMART SCORES
# ============================================================

def score_source_vote(d):
    """
    Source confidence voting:
    score + distinct provider support + agreement strength.
    """
    return float(np.clip(
        0.30 * d["score"] +
        0.35 * d["support_ratio"] +
        0.35 * d["object_vote_strength"],
        0.0,
        1.0,
    ))


def score_pixel_vote(d):
    """
    Голосование на уровне mask/pixels.
    """
    return float(np.clip(
        0.20 * d["score"] +
        0.20 * d["support_ratio"] +
        0.25 * d["mask_vote_q75"] +
        0.20 * d["top2_q75"] +
        0.15 * d["pixel_support2"],
        0.0,
        1.0,
    ))


def score_staple_vote(d):
    return float(np.clip(
        0.20 * d["score"] +
        0.20 * d["support_ratio"] +
        0.25 * d["object_vote_strength"] +
        0.35 * d["staple_q75"],
        0.0,
        1.0,
    ))


def score_ink_aware(d):
    """
    Здесь image content участвует напрямую.

    Особенно полезно против:
    - длинных пустых рамок;
    - линий таблицы;
    - detector hallucinations в белом поле.
    """
    line_penalty = min(
        0.25,
        0.40 * d["rule_density"]
    )

    score = (
        0.20 * d["score"] +
        0.20 * d["support_ratio"] +
        0.20 * d["object_vote_strength"] +
        0.17 * d["top2_q75"] +
        0.13 * d["pixel_support2"] +
        0.10 * d["textness"] -
        line_penalty
    )

    return float(np.clip(score, 0.0, 1.0))


def score_hybrid(d):
    """
    Итоговый "умный" вариант.

    Не требует, чтобы ВСЯ область совпадала пиксель-в-пиксель:
    достаточно object votes + локально сильного mask consensus.

    При этом:
    - много моделей -> бонус;
    - высокий calibrated score -> бонус;
    - реальные чернила -> бонус;
    - длинная прямая линия -> штраф.
    """
    support_bonus = min(
        1.0,
        d["support_models"] / 3.0,
    )

    line_penalty = min(
        0.22,
        0.35 * d["rule_density"]
    )

    score = (
        0.18 * d["score"] +
        0.18 * d["object_vote_strength"] +
        0.16 * support_bonus +
        0.14 * d["mask_vote_q75"] +
        0.12 * d["top2_q75"] +
        0.08 * d["staple_q75"] +
        0.08 * d["pixel_support2"] +
        0.06 * d["textness"] -
        line_penalty
    )

    return float(np.clip(score, 0.0, 1.0))


# ============================================================
# FILTER + DUPLICATE RESOLUTION
# ============================================================

def candidate_passes(d, method):
    """
    Здесь специально разные режимы, чтобы пользователь
    получил реально разные картинки для сравнения.
    """
    support = d["support_models"]

    strong_singleton = (
        d["score"] >= CFG["singleton_score_threshold"] and
        d["textness"] >= CFG["singleton_textness_threshold"]
    )

    if method == "source":
        return (
            support >= 2 or
            strong_singleton
        )

    if method == "pixel":
        return (
            d["pixel_support2"] >= 0.08 or
            d["top2_q75"] >= 0.28 or
            strong_singleton
        )

    if method == "staple":
        return (
            (
                support >= 2 and
                d["staple_q75"] >= 0.30
            ) or
            strong_singleton
        )

    if method == "ink":
        # Если только одна модель — image evidence должно быть сильным.
        if support <= 1:
            return strong_singleton

        # Сильная прямая линия и почти нет residual ink.
        if (
            d["rule_density"] > 0.20 and
            d["textness"] < 0.18 and
            support < 3
        ):
            return False

        return (
            d["textness"] >= 0.08 or
            support >= 3
        )

    if method == "hybrid":
        if support >= 3:
            return True

        if support == 2:
            return (
                d["pixel_support2"] >= 0.05 or
                d["object_vote_strength"] >= 0.28
            )

        return strong_singleton

    return True


def greedy_unique(candidates, score_fn, method):
    ranked = []

    for d in candidates:
        if not candidate_passes(d, method):
            continue

        item = copy.deepcopy(d)
        item["final_score"] = score_fn(item)
        ranked.append(item)

    ranked.sort(
        key=lambda x: (
            x["final_score"],
            x["support_models"],
            x["score"],
        ),
        reverse=True,
    )

    kept = []
    dup_thr = CFG["duplicate_affinity_threshold"]

    for candidate in ranked:
        duplicate = False

        for selected in kept:
            affinity = line_affinity(
                candidate,
                selected,
            )

            # Дополнительная защита от вложенного bbox.
            contained_overlap = io_min(
                candidate["bbox"],
                selected["bbox"],
            )

            if (
                affinity >= dup_thr or
                (
                    contained_overlap >= 0.72 and
                    affinity >= 0.40
                )
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept


# ============================================================
# IMAGE / PATH HELPERS
# ============================================================

def find_image(image_name, json_path, input_dir, images_dir):
    if not image_name:
        return None

    filename = Path(image_name).name

    candidates = []

    if images_dir:
        candidates += [
            images_dir / image_name,
            images_dir / filename,
        ]

    candidates += [
        json_path.parent / image_name,
        json_path.parent / filename,
        input_dir / image_name,
        input_dir / filename,
        input_dir / "images" / filename,
        input_dir.parent / "images" / filename,
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    if images_dir and images_dir.exists():
        matches = list(images_dir.rglob(filename))

        if matches:
            return matches[0]

    return None


def load_source_image(data, json_path, input_dir, images_dir):
    image_name = data.get("image_name", "")

    found = find_image(
        image_name,
        json_path,
        input_dir,
        images_dir,
    )

    if found:
        bgr = cv2.imread(
            str(found),
            cv2.IMREAD_COLOR,
        )

        if bgr is not None:
            return bgr, found

    w, h = data.get(
        "image_size",
        [1440, 1172],
    )

    return (
        np.full(
            (int(h), int(w), 3),
            255,
            dtype=np.uint8,
        ),
        None,
    )


# ============================================================
# VISUALIZATION
# ============================================================

def color_map_for_providers(providers):
    return {
        p: MODEL_COLORS[i % len(MODEL_COLORS)]
        for i, p in enumerate(providers)
    }


def draw_detections(
    image_bgr,
    detections,
    title,
    *,
    color_by_provider=False,
):
    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)

    providers = sorted(
        set(d["provider"] for d in detections)
    )

    cmap = color_map_for_providers(
        providers
    )

    for idx, d in enumerate(detections):
        poly = [
            (int(round(x)), int(round(y)))
            for x, y in d["polygon"]
        ]

        if not poly:
            continue

        if color_by_provider:
            color = cmap.get(
                d["provider"],
                (255, 0, 0),
            )
        else:
            color = (255, 0, 0)

        if len(poly) >= 2:
            draw.line(
                poly + [poly[0]],
                fill=color,
                width=CFG["draw_width"],
            )

        if CFG["draw_labels"]:
            final_score = d.get(
                "final_score",
                d["score"],
            )

            support = d.get(
                "support_models",
                1,
            )

            text = (
                f"{idx} "
                f"{d['provider']} "
                f"{final_score:.2f} "
                f"[{support}]"
            )

            x, y = poly[0]

            draw.text(
                (x + 2, max(2, y - 13)),
                text,
                fill=color,
            )

    # Header
    header_h = 24

    draw.rectangle(
        [0, 0, min(canvas.width, 1100), header_h],
        fill=(255, 255, 255),
    )

    draw.text(
        (5, 5),
        f"{title} | detections={len(detections)}",
        fill=(0, 0, 0),
    )

    return np.asarray(canvas)


def heatmap_overlay(image_bgr, heatmap_small, title):
    H, W = image_bgr.shape[:2]

    heat = cv2.resize(
        heatmap_small,
        (W, H),
        interpolation=cv2.INTER_LINEAR,
    )

    heat_u8 = np.clip(
        heat * 255.0,
        0,
        255,
    ).astype(np.uint8)

    colored = cv2.applyColorMap(
        heat_u8,
        cv2.COLORMAP_TURBO,
    )

    overlay = cv2.addWeighted(
        image_bgr,
        0.68,
        colored,
        0.32,
        0,
    )

    cv2.rectangle(
        overlay,
        (0, 0),
        (min(W - 1, 1100), 24),
        (255, 255, 255),
        -1,
    )

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

    return cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB,
    )


def make_grid(images, labels, target_width=720):
    """
    2 columns x N rows comparison sheet.
    """
    tiles = []

    for img, label in zip(images, labels):
        pil = Image.fromarray(img)

        ratio = target_width / pil.width
        h = max(1, int(round(pil.height * ratio)))

        pil = pil.resize(
            (target_width, h),
            Image.Resampling.LANCZOS,
        )

        tile = Image.new(
            "RGB",
            (target_width, h + 34),
            (255, 255, 255),
        )

        tile.paste(
            pil,
            (0, 34),
        )

        draw = ImageDraw.Draw(tile)

        draw.text(
            (8, 10),
            label,
            fill=(0, 0, 0),
        )

        tiles.append(tile)

    cols = 2
    rows = math.ceil(len(tiles) / cols)

    cell_w = max(t.width for t in tiles)
    cell_h = max(t.height for t in tiles)

    grid = Image.new(
        "RGB",
        (cols * cell_w, rows * cell_h),
        (235, 235, 235),
    )

    for i, tile in enumerate(tiles):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h

        grid.paste(
            tile,
            (x, y),
        )

    return np.asarray(grid)


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_json(
    json_path,
    input_dir,
    output_dir,
    images_dir,
):
    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(
        data.get("providers"),
        dict,
    ):
        return None

    detections = flatten_predictions(
        data
    )

    if not detections:
        return None

    image_bgr, image_path = load_source_image(
        data,
        json_path,
        input_dir,
        images_dir,
    )

    H, W = image_bgr.shape[:2]

    providers, provider_maps, det_masks, mask_hw = build_provider_maps(
        detections,
        H,
        W,
    )

    consensus = make_consensus_maps(
        providers,
        provider_maps,
    )

    staple_prob, staple_info = staple_binary(
        consensus["binary"],
    )

    evidence = build_image_evidence(
        image_bgr,
        mask_hw,
    )

    featured = compute_candidate_features(
        detections,
        providers,
        det_masks,
        consensus,
        staple_prob,
        evidence,
    )

    methods = {
        "01_source_vote": greedy_unique(
            featured,
            score_source_vote,
            "source",
        ),

        "02_pixel_vote": greedy_unique(
            featured,
            score_pixel_vote,
            "pixel",
        ),

        "03_staple_vote": greedy_unique(
            featured,
            score_staple_vote,
            "staple",
        ),

        "04_ink_aware": greedy_unique(
            featured,
            score_ink_aware,
            "ink",
        ),

        "05_hybrid_smart": greedy_unique(
            featured,
            score_hybrid,
            "hybrid",
        ),
    }

    stem = json_path.stem

    image_stem = (
        Path(data.get("image_name", "")).stem
        or stem
    )

    base = (
        image_stem
        if image_stem == stem
        else f"{stem}__{image_stem}"
    )

    # 00 all models
    all_img = draw_detections(
        image_bgr,
        detections,
        "00_all_models",
        color_by_provider=True,
    )

    Image.fromarray(all_img).save(
        output_dir / f"{base}__00_all_models.jpg",
        quality=94,
    )

    comparison_images = []
    comparison_labels = []

    for name, result in methods.items():
        vis = draw_detections(
            image_bgr,
            result,
            name,
            color_by_provider=False,
        )

        Image.fromarray(vis).save(
            output_dir / f"{base}__{name}.jpg",
            quality=94,
        )

        comparison_images.append(vis)
        comparison_labels.append(
            f"{name}: {len(result)}"
        )

    # Heatmaps
    vote_heat = heatmap_overlay(
        image_bgr,
        consensus["vote_map"],
        "06_vote_heatmap: provider-level score + agreement",
    )

    Image.fromarray(vote_heat).save(
        output_dir / f"{base}__06_vote_heatmap.jpg",
        quality=94,
    )

    staple_heat = heatmap_overlay(
        image_bgr,
        staple_prob,
        "07_staple_heatmap: reliability-weighted mask consensus",
    )

    Image.fromarray(staple_heat).save(
        output_dir / f"{base}__07_staple_heatmap.jpg",
        quality=94,
    )

    # Comparison grid:
    # all fusion methods + heatmap
    grid_imgs = comparison_images + [vote_heat]
    grid_labels = comparison_labels + ["06_vote_heatmap"]

    grid = make_grid(
        grid_imgs,
        grid_labels,
    )

    Image.fromarray(grid).save(
        output_dir / f"{base}__99_COMPARE.jpg",
        quality=92,
    )

    # Candidate diagnostics CSV for this page
    diagnostic_path = (
        output_dir /
        f"{base}__diagnostics.csv"
    )

    diagnostic_fields = [
        "id",
        "provider",
        "raw_score",
        "score",
        "support_models",
        "support_providers",
        "object_vote_strength",
        "mask_vote_mean",
        "mask_vote_q75",
        "top2_q75",
        "staple_q75",
        "pixel_support2",
        "pixel_support3",
        "textness",
        "ink_density",
        "residual_density",
        "rule_density",
        "residual_share",
        "source_score",
        "pixel_score",
        "staple_score",
        "ink_score",
        "hybrid_score",
    ]

    with open(
        diagnostic_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=diagnostic_fields,
        )

        writer.writeheader()

        for d in featured:
            writer.writerow({
                "id": d["id"],
                "provider": d["provider"],
                "raw_score": d["raw_score"],
                "score": d["score"],
                "support_models": d["support_models"],
                "support_providers": ",".join(d["support_providers"]),
                "object_vote_strength": d["object_vote_strength"],
                "mask_vote_mean": d["mask_vote_mean"],
                "mask_vote_q75": d["mask_vote_q75"],
                "top2_q75": d["top2_q75"],
                "staple_q75": d["staple_q75"],
                "pixel_support2": d["pixel_support2"],
                "pixel_support3": d["pixel_support3"],
                "textness": d["textness"],
                "ink_density": d["ink_density"],
                "residual_density": d["residual_density"],
                "rule_density": d["rule_density"],
                "residual_share": d["residual_share"],
                "source_score": score_source_vote(d),
                "pixel_score": score_pixel_vote(d),
                "staple_score": score_staple_vote(d),
                "ink_score": score_ink_aware(d),
                "hybrid_score": score_hybrid(d),
            })

    # Reliability info
    reliability_path = (
        output_dir /
        f"{base}__staple_reliability.json"
    )

    with open(
        reliability_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "providers": providers,
                "staple": staple_info,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "json": str(json_path),
        "image": str(image_path) if image_path else "",
        "raw": len(detections),
        **{
            name: len(result)
            for name, result in methods.items()
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Папка с JSON",
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Папка с исходными изображениями",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Папка результата",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()

    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = (
            input_dir.parent /
            f"{input_dir.name}_smart_compare"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images_dir = (
        args.images_dir.resolve()
        if args.images_dir
        else None
    )

    json_files = sorted(
        input_dir.rglob("*.json")
    )

    print(f"JSON:   {len(json_files)}")
    print(f"OUTPUT: {output_dir}")
    print()

    summary_rows = []

    for path in tqdm(
        json_files,
        desc="Smart fusion",
    ):
        try:
            row = process_json(
                path,
                input_dir,
                output_dir,
                images_dir,
            )

            if row:
                summary_rows.append(row)

        except Exception as exc:
            print()
            print(f"ERROR: {path}")
            print(
                f"{type(exc).__name__}: {exc}"
            )

    if summary_rows:
        fields = list(
            summary_rows[0].keys()
        )

        with open(
            output_dir / "summary.csv",
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(summary_rows)

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            CFG,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("DONE")
    print(f"RESULT: {output_dir}")


if __name__ == "__main__":
    main()
