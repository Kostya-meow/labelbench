"""Export YOLO26 segmentation and verify CUDA polygon agreement on a real image."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    os.environ["YOLO_AUTOINSTALL"] = "False"
    from labelbench.evaluation import evaluate
    from labelbench.providers.yolo26 import YOLO26Provider
    from labelbench.providers.yolo26_onnx import YOLO26ONNXProvider
    from labelbench.settings import Settings
    from labelbench.storage import write_json
    from ultralytics import YOLO

    cfg = Settings.from_environment()
    destination = cfg.models_dir / "onnx"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "verification.json", {"verified": False})
    original = YOLO(cfg.yolo26_model)
    checkpoint = destination / "yolo26n-seg.pt"
    shutil.copy2(original.ckpt_path, checkpoint)
    exported = YOLO(str(checkpoint)).export(format="onnx", imgsz=640, opset=17,
                                            simplify=False, dynamic=False, device="cpu")
    native = YOLO26Provider(str(checkpoint), "cuda").annotate(args.image)
    onnx = YOLO26ONNXProvider(cfg)
    onnx._device = "cuda"
    converted = onnx.annotate(args.image)
    report = evaluate(converted.annotations, native.annotations, threshold=0.95)
    report.update(image=str(args.image.resolve()), native_count=len(native.annotations),
                  onnx_count=len(converted.annotations), runtime="CUDAExecutionProvider", model=str(exported))
    report["verified"] = bool(native.annotations) and not report["fp"] and not report["fn"]
    write_json(destination / "verification.json", report)
    if not native.annotations or report["fp"] or report["fn"]:
        raise RuntimeError("ONNX parity verification failed or image has no detections; inspect verification.json")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
