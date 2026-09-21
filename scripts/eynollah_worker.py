"""Run the converted Eynollah textline model in its isolated environment."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

INPUT_WIDTH = 672
INPUT_HEIGHT = 448


def _add_torch_cuda_libraries() -> None:
    root = Path(__file__).resolve().parents[1]
    torch_lib = root / ".venv-gpu" / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.is_dir():
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")


def _session(model_path: Path, device: str) -> tuple[Any, str]:
    _add_torch_cuda_libraries()
    import onnxruntime as ort

    requested = "cuda" if device == "auto" else device
    session = ort.InferenceSession(
        str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    active = session.get_providers()
    if requested == "cuda" and "CUDAExecutionProvider" not in active:
        raise RuntimeError(f"CUDAExecutionProvider unavailable; active providers: {active}")
    actual_device = "cuda" if "CUDAExecutionProvider" in active else "cpu"
    return session, actual_device


def _encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    pixels = mask.astype(np.uint8).flatten(order="F")
    counts: list[int] = []
    previous = 0
    run_length = 0
    for pixel in pixels:
        value = int(pixel)
        if value == previous:
            run_length += 1
        else:
            counts.append(run_length)
            previous = value
            run_length = 1
    counts.append(run_length)
    return {"size": list(mask.shape), "counts": counts}


def _annotations(model: Any, image_path: Path, device: str, include_masks: bool = True) -> dict[str, Any]:
    try:
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
    except OSError as error:
        raise RuntimeError(f"Cannot read image: {image_path}") from error
    height, width = rgb.shape[:2]
    resized = cv2.resize(rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    input_name = model.get_inputs()[0].name
    probabilities = np.asarray(model.run(None, {input_name: tensor[None]})[0])[0]
    line_probability = probabilities[:, :, 1]
    small_mask = np.argmax(probabilities, axis=2).astype(np.uint8)
    mask = cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    confidence = cv2.resize(line_probability, (width, height), interpolation=cv2.INTER_LINEAR)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotations: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 4.0:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        polygon = contour.reshape(-1, 2)
        if len(polygon) < 3:
            continue
        if include_masks:
            contour_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 1, thickness=-1)
            score = float(np.mean(confidence[contour_mask.astype(bool)]))
        else:
            contour_mask = np.zeros((box_height, box_width), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour - np.array([x, y])], -1, 1, thickness=-1)
            score = float(np.mean(confidence[y:y+box_height, x:x+box_width][contour_mask.astype(bool)]))
        annotations.append(
            {
                "id": f"eynollah-textline-{uuid.uuid4().hex[:12]}",
                "label": "text_line",
                "score": min(1.0, max(0.0, score)),
                "bbox_xywh": [float(x), float(y), float(box_width), float(box_height)],
                "polygon": [[float(point[0]), float(point[1])] for point in polygon],
                "mask_rle": _encode_binary_mask(contour_mask.astype(bool)) if include_masks else None,
                "attributes": {"class_id": 1, "device": device, "area": int(area)},
                "provider": "eynollah_textline",
            }
        )
    annotations.sort(key=lambda item: (item["bbox_xywh"][1], item["bbox_xywh"][0]))
    return {"device": device, "image_size": [width, height], "annotations": annotations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prefetch", action="store_true")
    args = parser.parse_args()
    session, device = _session(args.model, args.device)
    if args.prefetch:
        print(json.dumps({"device": device}))
        return
    if args.image is None:
        raise ValueError("--image is required unless --prefetch is set")
    print(json.dumps(_annotations(session, args.image, device)))


if __name__ == "__main__":
    main()
