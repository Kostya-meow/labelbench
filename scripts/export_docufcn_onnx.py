"""Export the isolated Doc-UFCN network and compare its final polygons."""

import argparse
import logging
import subprocess
from pathlib import Path

from labelbench.evaluation import evaluate
from labelbench.providers.docufcn import DocUFCNProvider
from labelbench.providers.docufcn_onnx import DocUFCNONNXProvider
from labelbench.settings import Settings
from labelbench.storage import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    cfg = Settings.from_environment()
    converted = DocUFCNONNXProvider(cfg)
    converted._device = "cuda"
    converted._onnx_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    write_json(converted._onnx_checkpoint.with_suffix(".json"), {"verified": False})
    subprocess.run([str(cfg.dla_python), "scripts/docufcn_worker.py", "--checkpoint",
                    str(cfg.docufcn_checkpoint), "--device", "cuda", "--export-onnx",
                    str(converted._onnx_checkpoint)], check=True)
    expected = DocUFCNProvider(cfg.docufcn_checkpoint, cfg.dla_python, "cuda").annotate(args.image)
    actual = converted.annotate(args.image)
    report = evaluate(actual.annotations, expected.annotations, threshold=.95)
    report.update(verified=bool(expected.annotations) and not report["fp"] and not report["fn"],
                  runtime="CUDAExecutionProvider", image=str(args.image.resolve()))
    write_json(converted._onnx_checkpoint.with_suffix(".json"), report)
    if not report["verified"]:
        raise RuntimeError("Doc-UFCN ONNX polygon parity failed")
    logging.getLogger(__name__).warning("Verified %d polygons", len(actual.annotations))


if __name__ == "__main__":
    main()
