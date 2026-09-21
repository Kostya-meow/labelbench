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
    parser.add_argument("--onnx")
    parser.add_argument("--export-onnx")
    arguments = parser.parse_args()

    import torch
    from doc_ufcn.main import DocUFCN

    device = "cuda" if arguments.device == "auto" and torch.cuda.is_available() else arguments.device
    device = "cpu" if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Doc-UFCN worker")
    model = DocUFCN(2, INPUT_SIZE, device)
    if arguments.onnx and not arguments.export_onnx:
        model.mean, model.std = MEAN, STD
    else:
        model.load(arguments.checkpoint, MEAN, STD)
    if arguments.export_onnx:
        class StatelessBatchNorm(torch.nn.Module):
            """Equivalent inference math when training statistics are not tracked."""

            def __init__(self, layer: torch.nn.BatchNorm2d) -> None:
                super().__init__()
                self.weight = layer.weight
                self.bias = layer.bias
                self.eps = layer.eps

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                mean = value.mean(dim=(0, 2, 3), keepdim=True)
                variance = ((value - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
                return (value - mean) / torch.sqrt(variance + self.eps) * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        for parent in list(model.net.modules()):
            for name, layer in list(parent.named_children()):
                if isinstance(layer, torch.nn.BatchNorm2d) and not layer.track_running_stats:
                    setattr(parent, name, StatelessBatchNorm(layer))
        torch.onnx.export(model.net.eval(), (torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device),),
                          arguments.export_onnx, input_names=["image"], output_names=["probability"],
                          dynamic_axes={"image": {2: "height", 3: "width"},
                                        "probability": {2: "height", 3: "width"}},
                          opset_version=17, dynamo=False)
        return
    if arguments.onnx:
        import onnxruntime as ort

        providers = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(arguments.onnx, providers=providers)
        if device == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("Doc-UFCN ONNX CUDA unavailable")

        class Network:
            def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
                return torch.from_numpy(session.run(None, {"image": tensor.detach().cpu().numpy()})[0])

        model.net = Network()
    if arguments.prefetch:
        print(json.dumps({"device": device}))
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
