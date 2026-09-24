"""Gate the memory-efficient graph on GPU logits and real polygon parity."""

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from manuscript_worker import inference_lock, predict, prepare
from shapely.geometry import Polygon


class CaptureSession:
    def __init__(self, session: object) -> None:
        self.session = session
        self.outputs = None

    def __getattr__(self, name: str) -> object:
        return getattr(self.session, name)

    def run(self, *args: object) -> object:
        self.outputs = self.session.run(*args)
        return self.outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", type=Path)
    parser.add_argument("--stage", choices=["original", "matmul"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.images is None:
        args.images = [root / "архив 2/109.jpg", root / "архив 2/1090.jpg"]
        if not all(p.is_file() for p in args.images):
            args.images = [root / f"data/vendor/manuscript-ocr/example/images/crop{i}.png" for i in (1, 2)]
    directory = root / "data/models/manuscript/mask2former_line_v0_prev"
    original = directory / "mask2former_line_v0_prev.onnx"
    optimized = directory / "mask2former_line_v0_prev.matmul.onnx"
    if args.stage:
        with contextlib.redirect_stdout(sys.stderr):
            model = prepare(root, "cuda", weights_override=original if args.stage == "original" else optimized)
            model.onnx_session = CaptureSession(model.onnx_session)
            for index, image in enumerate(args.images):
                result = predict(model, image, None)
                np.savez(args.output / f"{args.stage}-{index}.npz", *model.onnx_session.outputs)
                (args.output / f"{args.stage}-{index}.json").write_text(json.dumps(result), encoding="utf-8")
        return
    results = []
    with inference_lock(root), tempfile.TemporaryDirectory(prefix="parity-", dir=directory) as temporary:
        output = Path(temporary)
        for stage in ["original", "matmul"]:
            subprocess.run([sys.executable, __file__, "--stage", stage, "--output", temporary,
                            "--images", *map(str, args.images)], check=True)
        for index, image in enumerate(args.images):
            diffs = []
            with np.load(output / f"original-{index}.npz") as first, np.load(output / f"matmul-{index}.npz") as second:
                for key in first.files:
                    a, b = first[key].astype(np.float32), second[key].astype(np.float32)
                    diffs.append(float(np.max(np.abs(a-b))))
                    if not np.allclose(a, b, rtol=.01, atol=.08):
                        raise RuntimeError(f"GPU logits diverged: {image.name}/{key}, max abs {diffs[-1]}")
            a = json.loads((output / f"original-{index}.json").read_text())["annotations"]
            b = json.loads((output / f"matmul-{index}.json").read_text())["annotations"]
            polygons = [Polygon(row["polygon"]).buffer(0) for row in b]
            matches = []
            for row in a:
                polygon = Polygon(row["polygon"]).buffer(0)
                overlaps = [polygon.intersection(p).area / polygon.union(p).area for p in polygons]
                best = int(np.argmax(overlaps)) if overlaps else -1
                matches.append(overlaps[best] if best >= 0 else 0.)
                if best >= 0:
                    polygons.pop(best)
            if len(a) != len(b) or min(matches, default=1.) < .98:
                raise RuntimeError(f"Polygon parity failed: {image.name}, {len(a)}/{len(b)}, IoU {min(matches, default=0)}")
            results.append({"image": image.name, "polygons": len(a), "min_polygon_iou": min(matches, default=1.),
                            "max_abs_logits": diffs})
    report = {"verified": True, "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
              "runtime_sha256": hashlib.sha256((root / "scripts/manuscript_runtime.py").read_bytes()).hexdigest(),
              "optimized_sha256": hashlib.sha256(optimized.read_bytes()).hexdigest(), "tests": results,
              "rtol": .01, "atol": .08, "minimum_polygon_iou": .98}
    (directory / "matmul-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
