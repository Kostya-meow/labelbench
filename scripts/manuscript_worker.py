"""Isolated source-checkout Mask2Former inference using the upstream ONNX pipeline."""

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from PIL import Image


@contextlib.contextmanager
def inference_lock(root: Path) -> Iterator[None]:
    """Serialize this model across API processes to avoid simultaneous allocations."""
    import msvcrt

    path = root / "data/models/manuscript/inference.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if not path.stat().st_size:
            stream.write(b"0")
            stream.flush()
        deadline = time.monotonic()+300
        while True:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Manuscript GPU queue timeout") from None
                time.sleep(.25)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def prepare(root: Path, device: str, download: bool = False, weights_override: Path | None = None) -> object:
    import onnxruntime as ort
    from manuscript.detectors import Mask2Former
    from manuscript.models import Registry

    repository = root / "data/vendor/manuscript-ocr"
    catalog = json.loads((repository / "src/manuscript/models/registry.json").read_text(encoding="utf-8"))
    preset = "mask2former_line_v0_prev"
    directory = root / "data/models/manuscript" / preset
    artifacts = catalog["models"][preset]["artifacts"]
    if download:
        registry = Registry(root=root / "data/models/manuscript", timeout=120)
        # Pin the branch catalog: do not refresh it from main or replace its artifacts.
        directory.mkdir(parents=True, exist_ok=True)
        for spec in artifacts.values():
            if spec is not None:
                registry._artifact(directory, spec)
    for spec in artifacts.values():
        if spec is None:
            continue
        path = directory / spec["filename"]
        if not path.is_file() or path.stat().st_size != spec["size"]:
            raise RuntimeError(f"Missing/incomplete model artifact: {path.name}; run DOWNLOAD_MANUSCRIPT.bat")
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {path.name}")
    if device == "auto":
        device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    if device == "cuda":
        cuda_lib = root / ".venv-gpu/Lib/site-packages/torch/lib"
        if cuda_lib.is_dir():
            os.environ["PATH"] = str(cuda_lib) + os.pathsep + os.environ.get("PATH", "")
            ort.preload_dlls(directory=str(cuda_lib))
    weights = directory / artifacts["weights"]["filename"]
    optimized = directory / "mask2former_line_v0_prev.matmul.onnx"
    verified = directory / "matmul-verification.json"
    if optimized.is_file() and verified.is_file():
        record = json.loads(verified.read_text(encoding="utf-8"))
        if (record.get("verified") and record.get("source_sha256") == artifacts["weights"]["sha256"]
                and record.get("runtime_sha256") == hashlib.sha256((root / "scripts/manuscript_runtime.py").read_bytes()).hexdigest()
                and record.get("optimized_sha256") == hashlib.sha256(optimized.read_bytes()).hexdigest()):
            weights = optimized
    if weights == directory / artifacts["weights"]["filename"] and not download and weights_override is None:
        raise RuntimeError("Memory-efficient graph has not passed verification; run INSTALL_MANUSCRIPT.bat")
    weights = weights_override or weights
    detector_class = Mask2Former
    if weights.name.endswith(".matmul.onnx"):
        from manuscript_runtime import MemoryEfficientMask2Former

        detector_class = MemoryEfficientMask2Former
    detector = detector_class(weights=weights,
                          config=directory / artifacts["config"]["filename"], device=device)
    detector._initialize_session()
    if device == "cuda" and "CUDAExecutionProvider" not in detector.onnx_session.get_providers():
        raise RuntimeError("CUDAExecutionProvider did not load; refusing silent CPU fallback")
    return detector


def predict(detector: object, image: Path, threshold: float | None) -> dict:
    if threshold is not None:
        detector.score_threshold = threshold
    page = detector.predict(str(image))
    with Image.open(image) as source:
        size = list(source.size)
    annotations = []
    for block in page.blocks:
        for line in block.lines:
            for span in line.text_spans:
                polygon = [[float(x), float(y)] for x, y in span.polygon]
                xs, ys = zip(*polygon)
                annotations.append({"id": f"manuscript-mask2former-{len(annotations)}", "label": "text_line",
                                    "score": float(span.detection_confidence), "polygon": polygon,
                                    "bbox_xywh": [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)],
                                    "provider": "manuscript_mask2former_onnx",
                                    "attributes": {"device": detector.device, "backend": "onnxruntime",
                                                   "memory_optimized": str(detector.weights).endswith(".matmul.onnx")}})
    return {"device": detector.device, "image_size": size, "annotations": annotations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--prefetch", action="store_true")
    args = parser.parse_args()
    if args.confidence is not None and not 0 <= args.confidence <= 1:
        parser.error("confidence must be between 0 and 1")
    root = Path(__file__).resolve().parents[1]
    with inference_lock(root), contextlib.redirect_stdout(sys.stderr):
        detector = prepare(root, args.device, download=args.prefetch)
        if args.prefetch:
            report = {"device": detector.device, "verified": True,
                      "providers": detector.onnx_session.get_providers(),
                      "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                               cwd=root / "data/vendor/manuscript-ocr", text=True).strip()}
            (root / "data/models/manuscript/verified.json").write_text(json.dumps(report), encoding="utf-8")
        else:
            if args.image is None:
                parser.error("--image is required")
            report = predict(detector, args.image, args.confidence)
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
