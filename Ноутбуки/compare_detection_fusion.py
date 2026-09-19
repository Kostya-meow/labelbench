#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Сравнение нескольких способов объединения детекций текста из JSON.

Вход:
    папка с JSON формата:
    {
      "image_name": "...jpg",
      "image_size": [W, H],
      "providers": {
         "model_a": {"annotations": [...]},
         "model_b": {"annotations": [...]}
      }
    }

Запуск:
    python compare_detection_fusion.py /path/to/json_folder

Можно явно указать папку с изображениями:
    python compare_detection_fusion.py /path/to/json_folder --images-dir /path/to/images

По умолчанию рядом с входной папкой будет создано:
    <json_folder>_fusion_compare/

Структура:
    <json_folder>_fusion_compare/
        00_all_models/
        01_score_nms/
        02_soft_nms/
        03_weighted_box_fusion/
        04_consensus_best_polygon/
        05_consensus_medoid/
        06_strict_consensus/
        summary.csv

Зависимости:
    pip install pillow numpy tqdm
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ============================================================
# НАСТРОЙКИ
# ============================================================

CFG = {
    # Предсказания ниже этого raw score игнорируем.
    "min_raw_score": 0.20,

    # Поскольку score разных моделей часто плохо сопоставим,
    # смешиваем raw score и percentile внутри конкретной модели.
    "raw_score_weight": 0.45,
    "rank_score_weight": 0.55,

    # Порог сходства для того, чтобы считать две детекции
    # одной и той же строкой/областью.
    "cluster_affinity_threshold": 0.45,

    # Обычный NMS.
    "nms_iou_threshold": 0.45,

    # Soft-NMS.
    "soft_nms_sigma": 0.50,
    "soft_nms_min_score": 0.25,

    # Strict consensus:
    # область оставляется, если её подтвердили >= N разных моделей.
    "strict_min_models": 2,

    # Одиночная детекция всё же может остаться,
    # если её нормализованный score очень высокий.
    "strict_singleton_score": 0.86,

    # В Consensus Best Polygon итоговая оценка кандидата:
    "candidate_score_weight": 0.45,
    "candidate_support_weight": 0.30,
    "candidate_geometry_weight": 0.25,

    # Дополнительный вес отдельным моделям.
    # Например:
    # "rfdetr_historical": 1.10
    "provider_weights": {},

    # Толщина линий визуализации.
    "draw_width": 3,

    # Показывать подпись около каждого полигона.
    "draw_labels": True,
}


# Цвета нужны только для визуального различия моделей.
# PIL ожидает RGB.
MODEL_COLORS = [
    (255, 0, 0),
    (0, 140, 255),
    (0, 180, 0),
    (220, 120, 0),
    (180, 0, 220),
    (0, 180, 180),
    (255, 80, 140),
    (120, 120, 0),
]


# ============================================================
# БАЗОВАЯ ГЕОМЕТРИЯ
# ============================================================

def xywh_to_xyxy(box):
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, x2 - x1, y2 - y1]


def bbox_from_polygon(poly):
    arr = np.asarray(poly, dtype=float)
    return [
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        float(arr[:, 0].max()),
        float(arr[:, 1].max()),
    ]


def rect_polygon_from_xyxy(box):
    x1, y1, x2, y2 = map(float, box)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


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
    """
    Intersection / minimum area.

    Полезнее обычного IoU, если одна модель дала почти всю строку,
    а другая только её часть.
    """
    inter = intersection_area(a, b)
    denom = min(box_area(a), box_area(b))
    return inter / denom if denom > 0 else 0.0


def axis_overlap_ratio(a1, a2, b1, b2):
    inter = max(0.0, min(a2, b2) - max(a1, b1))
    denom = min(max(1e-9, a2 - a1), max(1e-9, b2 - b1))
    return inter / denom


def line_affinity(a, b):
    """
    Сходство двух детекций текстовой строки.

    Используется смесь:
      - IoU;
      - IoMin;
      - overlap по Y;
      - близость вертикальных центров.

    Это лучше обычного IoU для строк текста разной длины.
    """
    A = a["bbox"]
    B = b["bbox"]

    i = iou(A, B)
    iom = io_min(A, B)
    yov = axis_overlap_ratio(A[1], A[3], B[1], B[3])

    h1 = max(1.0, A[3] - A[1])
    h2 = max(1.0, B[3] - B[1])

    cy1 = (A[1] + A[3]) / 2.0
    cy2 = (B[1] + B[3]) / 2.0

    center_y_sim = math.exp(
        -abs(cy1 - cy2) / max(h1, h2)
    )

    return (
        0.25 * i
        + 0.35 * iom
        + 0.25 * yov
        + 0.15 * center_y_sim
    )


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def provider_weight(name: str) -> float:
    return float(CFG["provider_weights"].get(name, 1.0))


def percentile_ranks(values):
    """
    Преобразуем confidence внутри каждой модели в percentile [0..1].

    Это важно, потому что:
      модель A может считать 0.70 хорошим score,
      модель B почти всегда выдаёт 0.95.
    """
    values = np.asarray(values, dtype=float)

    if len(values) <= 1:
        return np.ones_like(values)

    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)

    return ranks / float(len(values) - 1)


def flatten_predictions(data: Dict[str, Any]):
    detections = []

    providers = data.get("providers", {})

    for provider, block in providers.items():
        annotations = block.get("annotations", [])
        raw_scores = [
            float(a.get("score", 0.0))
            for a in annotations
        ]

        ranks = percentile_ranks(raw_scores)

        for ann, rank in zip(annotations, ranks):
            raw = float(ann.get("score", 0.0))

            if raw < CFG["min_raw_score"]:
                continue

            polygon = ann.get("polygon")
            bbox_xywh = ann.get("bbox_xywh")

            if bbox_xywh is not None:
                bbox = xywh_to_xyxy(bbox_xywh)
            elif polygon:
                bbox = bbox_from_polygon(polygon)
            else:
                continue

            if not polygon:
                polygon = rect_polygon_from_xyxy(bbox)

            calibrated_score = (
                CFG["raw_score_weight"] * raw
                + CFG["rank_score_weight"] * float(rank)
            )

            calibrated_score *= provider_weight(provider)

            detections.append({
                "id": ann.get("id"),
                "provider": provider,
                "model": block.get("model", provider),
                "label": ann.get("label", "text"),
                "raw_score": raw,
                "rank_score": float(rank),
                "score": float(calibrated_score),
                "bbox": list(map(float, bbox)),
                "polygon": [
                    [float(x), float(y)]
                    for x, y in polygon
                ],
            })

    return detections


# ============================================================
# CLUSTERING
# ============================================================

def cluster_detections(detections, threshold=None):
    if threshold is None:
        threshold = CFG["cluster_affinity_threshold"]

    remaining = sorted(
        [copy.deepcopy(d) for d in detections],
        key=lambda d: d["score"],
        reverse=True,
    )

    clusters = []

    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]

        changed = True

        while changed:
            changed = False
            keep = []

            for candidate in remaining:
                best_affinity = max(
                    line_affinity(candidate, member)
                    for member in cluster
                )

                if best_affinity >= threshold:
                    cluster.append(candidate)
                    changed = True
                else:
                    keep.append(candidate)

            remaining = keep

        clusters.append(cluster)

    return clusters


# ============================================================
# МЕТОД 1: SCORE NMS
# ============================================================

def score_nms(detections):
    threshold = CFG["nms_iou_threshold"]

    work = sorted(
        [copy.deepcopy(d) for d in detections],
        key=lambda d: d["score"],
        reverse=True,
    )

    kept = []

    while work:
        best = work.pop(0)
        kept.append(best)

        work = [
            d for d in work
            if iou(best["bbox"], d["bbox"]) < threshold
        ]

    return kept


# ============================================================
# МЕТОД 2: SOFT NMS
# ============================================================

def soft_nms(detections):
    sigma = CFG["soft_nms_sigma"]
    min_score = CFG["soft_nms_min_score"]

    work = [copy.deepcopy(d) for d in detections]
    kept = []

    while work:
        work.sort(
            key=lambda d: d["score"],
            reverse=True,
        )

        best = work.pop(0)
        kept.append(best)

        new_work = []

        for d in work:
            overlap = iou(best["bbox"], d["bbox"])

            d["score"] *= math.exp(
                -(overlap * overlap) / sigma
            )

            if d["score"] >= min_score:
                new_work.append(d)

        work = new_work

    return kept


# ============================================================
# МЕТОД 3: WEIGHTED BOX FUSION
# ============================================================

def weighted_box_fusion(detections):
    clusters = cluster_detections(detections)
    output = []

    for idx, cluster in enumerate(clusters):
        weights = np.asarray(
            [
                max(1e-6, d["score"])
                * provider_weight(d["provider"])
                for d in cluster
            ],
            dtype=float,
        )

        boxes = np.asarray(
            [d["bbox"] for d in cluster],
            dtype=float,
        )

        fused_box = np.average(
            boxes,
            axis=0,
            weights=weights,
        )

        providers = sorted(
            set(d["provider"] for d in cluster)
        )

        output.append({
            "id": f"wbf-{idx:05d}",
            "provider": "WBF",
            "model": "Weighted Box Fusion",
            "label": "text_line",
            "raw_score": float(
                np.average(
                    [d["raw_score"] for d in cluster],
                    weights=weights,
                )
            ),
            "rank_score": max(
                d["rank_score"] for d in cluster
            ),
            "score": max(
                d["score"] for d in cluster
            ),
            "bbox": fused_box.tolist(),
            "polygon": rect_polygon_from_xyxy(
                fused_box.tolist()
            ),
            "support_models": len(providers),
            "support_providers": providers,
            "cluster_size": len(cluster),
        })

    return output


# ============================================================
# CONSENSUS FUNCTIONS
# ============================================================

def candidate_geometry_score(candidate, cluster):
    others = [
        d for d in cluster
        if d["id"] != candidate["id"]
        or d["provider"] != candidate["provider"]
    ]

    if not others:
        return 0.0

    similarities = []
    weights = []

    for other in others:
        w = max(0.05, other["score"])

        similarities.append(
            line_affinity(candidate, other)
        )
        weights.append(w)

    return float(
        np.average(
            similarities,
            weights=weights,
        )
    )


# ============================================================
# МЕТОД 4: CONSENSUS BEST POLYGON
# ============================================================

def consensus_best_polygon(detections):
    clusters = cluster_detections(detections)

    total_providers = max(
        1,
        len(set(d["provider"] for d in detections)),
    )

    output = []

    for idx, cluster in enumerate(clusters):
        support_providers = sorted(
            set(d["provider"] for d in cluster)
        )

        support_ratio = (
            len(support_providers)
            / total_providers
        )

        candidates = []

        for d in cluster:
            geometry_score = candidate_geometry_score(
                d,
                cluster,
            )

            quality = (
                CFG["candidate_score_weight"]
                * d["score"]

                + CFG["candidate_support_weight"]
                * support_ratio

                + CFG["candidate_geometry_weight"]
                * geometry_score
            )

            candidates.append(
                (quality, geometry_score, d)
            )

        quality, geometry_score, best = max(
            candidates,
            key=lambda x: x[0],
        )

        result = copy.deepcopy(best)

        result.update({
            "id": f"consensus-best-{idx:05d}",
            "fusion_method": "consensus_best_polygon",
            "final_quality": float(quality),
            "geometry_score": float(geometry_score),
            "support_models": len(support_providers),
            "support_providers": support_providers,
            "cluster_size": len(cluster),
        })

        output.append(result)

    return output


# ============================================================
# МЕТОД 5: CONSENSUS MEDOID
# ============================================================

def consensus_medoid(detections):
    clusters = cluster_detections(detections)
    output = []

    for idx, cluster in enumerate(clusters):
        support_providers = sorted(
            set(d["provider"] for d in cluster)
        )

        scored = []

        for candidate in cluster:
            geometry_score = candidate_geometry_score(
                candidate,
                cluster,
            )

            # Здесь геометрия важнее score.
            medoid_score = (
                0.72 * geometry_score
                + 0.28 * candidate["score"]
            )

            scored.append(
                (
                    medoid_score,
                    geometry_score,
                    candidate,
                )
            )

        medoid_score, geometry_score, best = max(
            scored,
            key=lambda x: x[0],
        )

        result = copy.deepcopy(best)

        result.update({
            "id": f"medoid-{idx:05d}",
            "fusion_method": "consensus_medoid",
            "final_quality": float(medoid_score),
            "geometry_score": float(geometry_score),
            "support_models": len(support_providers),
            "support_providers": support_providers,
            "cluster_size": len(cluster),
        })

        output.append(result)

    return output


# ============================================================
# МЕТОД 6: STRICT CONSENSUS
# ============================================================

def strict_consensus(detections):
    """
    Основной вариант, если цель:
      одна максимально чистая итоговая разметка.

    1. Берём лучший реальный polygon внутри consensus cluster.
    2. Если область подтверждена >= strict_min_models:
       оставляем.
    3. Если только одна модель:
       оставляем только при очень высоком score.
    """
    candidates = consensus_best_polygon(detections)
    output = []

    for d in candidates:
        support = int(
            d.get("support_models", 1)
        )

        if support >= CFG["strict_min_models"]:
            output.append(d)

        elif d["score"] >= CFG["strict_singleton_score"]:
            output.append(d)

    return output


# ============================================================
# ПОИСК ИЗОБРАЖЕНИЯ
# ============================================================

def find_image(
    image_name: str,
    json_path: Path,
    input_dir: Path,
    images_dir: Optional[Path],
) -> Optional[Path]:

    if not image_name:
        return None

    candidates = []

    if images_dir:
        candidates.extend([
            images_dir / image_name,
            images_dir / Path(image_name).name,
        ])

    candidates.extend([
        json_path.parent / image_name,
        json_path.parent / Path(image_name).name,

        input_dir / image_name,
        input_dir / Path(image_name).name,

        input_dir / "images" / image_name,
        input_dir / "images" / Path(image_name).name,

        input_dir.parent / image_name,
        input_dir.parent / Path(image_name).name,

        input_dir.parent / "images" / image_name,
        input_dir.parent / "images" / Path(image_name).name,
    ])

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    # Последняя попытка: рекурсивно внутри входной папки.
    filename = Path(image_name).name

    matches = list(input_dir.rglob(filename))

    if matches:
        return matches[0]

    if images_dir and images_dir.exists():
        matches = list(images_dir.rglob(filename))

        if matches:
            return matches[0]

    return None


# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================

def provider_color_map(detections):
    providers = sorted(
        set(d["provider"] for d in detections)
    )

    return {
        provider: MODEL_COLORS[i % len(MODEL_COLORS)]
        for i, provider in enumerate(providers)
    }


def load_image_or_canvas(
    data: Dict[str, Any],
    json_path: Path,
    input_dir: Path,
    images_dir: Optional[Path],
):
    image_name = data.get("image_name", "")

    image_path = find_image(
        image_name=image_name,
        json_path=json_path,
        input_dir=input_dir,
        images_dir=images_dir,
    )

    if image_path is not None:
        return Image.open(image_path).convert("RGB"), image_path

    width, height = data.get(
        "image_size",
        [1440, 1172],
    )

    canvas = Image.new(
        "RGB",
        (int(width), int(height)),
        (255, 255, 255),
    )

    return canvas, None


def draw_predictions(
    image: Image.Image,
    predictions,
    *,
    colors_by_provider=False,
    title_text=None,
):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    if colors_by_provider:
        color_map = provider_color_map(predictions)
    else:
        color_map = {}

    for index, d in enumerate(predictions):
        polygon = [
            (float(x), float(y))
            for x, y in d["polygon"]
        ]

        if colors_by_provider:
            color = color_map.get(
                d["provider"],
                (255, 0, 0),
            )
        else:
            color = (255, 0, 0)

        if len(polygon) >= 2:
            draw.line(
                polygon + [polygon[0]],
                fill=color,
                width=CFG["draw_width"],
            )

        if CFG["draw_labels"] and polygon:
            provider = d.get(
                "provider",
                "?",
            )

            score = float(
                d.get("score", 0.0)
            )

            support = int(
                d.get("support_models", 1)
            )

            text = (
                f"{index} "
                f"{provider} "
                f"{score:.2f} "
                f"[{support}]"
            )

            x, y = polygon[0]

            draw.text(
                (
                    max(0, int(x) + 2),
                    max(0, int(y) - 13),
                ),
                text,
                fill=color,
            )

    if title_text:
        # Небольшая белая панель сверху.
        text_height = 22

        draw.rectangle(
            [
                0,
                0,
                min(canvas.width, 900),
                text_height,
            ],
            fill=(255, 255, 255),
        )

        draw.text(
            (5, 4),
            title_text,
            fill=(0, 0, 0),
        )

    return canvas


# ============================================================
# EXPORT
# ============================================================

METHODS = {
    "01_score_nms": score_nms,
    "02_soft_nms": soft_nms,
    "03_weighted_box_fusion": weighted_box_fusion,
    "04_consensus_best_polygon": consensus_best_polygon,
    "05_consensus_medoid": consensus_medoid,
    "06_strict_consensus": strict_consensus,
}


def safe_output_name(
    json_path: Path,
    image_name: str,
):
    image_stem = (
        Path(image_name).stem
        if image_name
        else json_path.stem
    )

    return f"{json_path.stem}__{image_stem}.jpg"


def process_one_json(
    json_path: Path,
    input_dir: Path,
    output_dir: Path,
    images_dir: Optional[Path],
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

    detections = flatten_predictions(data)

    if not detections:
        return None

    base_image, found_image_path = load_image_or_canvas(
        data=data,
        json_path=json_path,
        input_dir=input_dir,
        images_dir=images_dir,
    )

    image_name = data.get(
        "image_name",
        "",
    )

    output_filename = safe_output_name(
        json_path,
        image_name,
    )

    # --------------------------------------------------------
    # 00: все модели одновременно
    # --------------------------------------------------------
    all_dir = output_dir / "00_all_models"
    all_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_image = draw_predictions(
        base_image,
        detections,
        colors_by_provider=True,
        title_text=(
            f"ALL MODELS | "
            f"detections={len(detections)}"
        ),
    )

    all_image.save(
        all_dir / output_filename,
        quality=95,
    )

    method_stats = {}

    # --------------------------------------------------------
    # Fusion методы
    # --------------------------------------------------------
    for method_name, method_fn in METHODS.items():
        predictions = method_fn(
            detections
        )

        method_dir = (
            output_dir / method_name
        )

        method_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        vis = draw_predictions(
            base_image,
            predictions,
            colors_by_provider=False,
            title_text=(
                f"{method_name} | "
                f"detections={len(predictions)}"
            ),
        )

        vis.save(
            method_dir / output_filename,
            quality=95,
        )

        method_stats[method_name] = len(
            predictions
        )

    return {
        "json": str(json_path),
        "image_name": image_name,
        "image_found": (
            str(found_image_path)
            if found_image_path
            else ""
        ),
        "raw_detections": len(
            detections
        ),
        **method_stats,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Папка с result JSON",
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help=(
            "Необязательная папка "
            "с исходными изображениями"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Куда сохранить результат. "
            "По умолчанию создаётся "
            "<input_dir>_fusion_compare рядом."
        ),
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()

    if args.output_dir is None:
        output_dir = (
            input_dir.parent
            / f"{input_dir.name}_fusion_compare"
        )
    else:
        output_dir = (
            args.output_dir.resolve()
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Ищем JSON рекурсивно.
    json_files = sorted(
        input_dir.rglob("*.json")
    )

    print(
        f"JSON files: {len(json_files)}"
    )

    print(
        f"Output: {output_dir}"
    )

    rows = []

    for json_path in tqdm(
        json_files,
        desc="Processing",
    ):
        try:
            row = process_one_json(
                json_path=json_path,
                input_dir=input_dir,
                output_dir=output_dir,
                images_dir=args.images_dir,
            )

            if row is not None:
                rows.append(row)

        except Exception as exc:
            print(
                f"\nERROR: {json_path}\n"
                f"  {type(exc).__name__}: {exc}"
            )

    # summary.csv
    if rows:
        columns = [
            "json",
            "image_name",
            "image_found",
            "raw_detections",
            *METHODS.keys(),
        ]

        with open(
            output_dir / "summary.csv",
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=columns,
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(row)

    # config.json
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
    print(
        f"Processed: {len(rows)} JSON"
    )
    print(
        f"Results:   {output_dir}"
    )


if __name__ == "__main__":
    main()
