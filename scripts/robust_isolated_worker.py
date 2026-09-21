"""Persistent local JSON-lines inference worker; no augmented images retained."""

import argparse
import contextlib
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["docufcn", "eynollah"])
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        if args.kind == "eynollah":
            from eynollah_worker import _annotations, _session

            model, device = _session(args.checkpoint, "cuda")

            def predict(path: Path) -> list[dict]:
                return _annotations(model, path, device, include_masks=False)["annotations"]
        else:
            import numpy as np
            import onnxruntime as ort
            import torch
            from doc_ufcn.main import DocUFCN
            from PIL import Image

            session = ort.InferenceSession(str(args.checkpoint), providers=[
                ("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"])
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("CUDA unavailable")

            class Network:
                def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
                    return torch.from_numpy(session.run(None, {"image": tensor.cpu().numpy()})[0])

            model = DocUFCN(2, 768, "cuda")
            model.mean, model.std = [194, 185, 160], [49, 49, 47]
            model.net = Network()

            def predict(path: Path) -> list[dict]:
                with Image.open(path) as image:
                    predictions = model.predict(np.asarray(image.convert("RGB")))[0]
                result = []
                for label, rows in predictions.items():
                    if int(label) != 1:
                        continue
                    for index, row in enumerate(rows):
                        points = np.asarray(row["polygon"], dtype=float)
                        x1, y1 = points.min(axis=0)
                        x2, y2 = points.max(axis=0)
                        result.append({"id": f"docufcn-{index}", "label": "text_line",
                                       "score": float(row["confidence"]), "polygon": points.tolist(),
                                       "bbox_xywh": [x1, y1, x2-x1, y2-y1], "provider": "docufcn_onnx"})
                return result
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                rows = predict(Path(request["image"]))
            reply = {"annotations": rows}
        except Exception as error:  # noqa: BLE001 - transport errors to parent
            reply = {"error": f"{type(error).__name__}: {error}"}
        sys.stdout.write(json.dumps(reply, ensure_ascii=False, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
