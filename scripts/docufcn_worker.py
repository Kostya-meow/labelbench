"""Isolated Doc-UFCN worker used by the main API process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

MEAN = [194, 185, 160]
STD = [49, 49, 47]
INPUT_SIZE = 768


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--image")
    parser.add_argument("--prefetch", action="store_true")
    arguments = parser.parse_args()

    import torch
    from doc_ufcn.main import DocUFCN

    device = "cuda" if arguments.device == "auto" and torch.cuda.is_available() else arguments.device
    device = "cpu" if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Doc-UFCN worker")
    model = DocUFCN(2, INPUT_SIZE, device)
    model.load(arguments.checkpoint, MEAN, STD)
    if arguments.prefetch:
        print(json.dumps({"device": next(model.net.parameters()).device.type}))
        return
    if not arguments.image:
        raise ValueError("--image is required unless --prefetch is used")
    image = np.asarray(Image.open(Path(arguments.image)).convert("RGB"))
    predictions = model.predict(image)[0]
    annotations: list[dict[str, object]] = []
    for class_id, candidates in predictions.items():
        label = "text_line" if int(class_id) == 1 else str(class_id)
        for index, candidate in enumerate(candidates):
            polygon = [
                [float(point[0]), float(point[1])] for point in candidate["polygon"]
            ]
            if len(polygon) < 3:
                continue
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            annotations.append(
                {
                    "id": f"docufcn-{int(class_id)}-{index}",
                    "label": label,
                    "score": float(candidate["confidence"]),
                    "bbox_xywh": [
                        min(xs),
                        min(ys),
                        max(xs) - min(xs),
                        max(ys) - min(ys),
                    ],
                    "polygon": polygon,
                    "attributes": {"class_id": int(class_id), "device": device},
                    "provider": "docufcn",
                }
            )
    print(json.dumps({"device": device, "annotations": annotations}))


if __name__ == "__main__":
    main()
