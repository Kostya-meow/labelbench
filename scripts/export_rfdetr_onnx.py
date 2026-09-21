"""Export historical RF-DETR and validate final text-line polygons on CUDA."""

import argparse
import logging
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(Path("data/models/huggingface").resolve()))
    os.environ.setdefault("RF_HOME", str(Path("data/models/rfdetr").resolve()))
    import torch
    from rfdetr import RFDETRSegPreview

    from labelbench.evaluation import evaluate
    from labelbench.providers.rfdetr import RFDETRHistoricalProvider
    from labelbench.providers.rfdetr_onnx import RFDETRONNXProvider
    from labelbench.settings import Settings
    from labelbench.storage import write_json

    cfg = Settings.from_environment()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    converted = RFDETRONNXProvider(cfg)
    converted._device = "cuda"
    write_json(converted._onnx_checkpoint.with_suffix(".json"), {"verified": False})
    native = RFDETRHistoricalProvider(cfg.rfdetr_checkpoint, "cuda")
    native._model = RFDETRSegPreview(pretrain_weights=str(cfg.rfdetr_checkpoint), device="cuda", num_classes=2)
    if not args.verify_only:
        native._model.export(output_dir=str(converted._onnx_checkpoint.parent),
                             output_name=converted._onnx_checkpoint.stem, verbose=False)
    native._model.inference(compile=False, inplace=True, dtype="float32")
    expected = native.annotate(args.image)
    actual = converted.annotate(args.image)
    report = evaluate(actual.annotations, expected.annotations, threshold=.95)
    report.update(verified=bool(expected.annotations) and not report["fp"] and not report["fn"],
                  runtime="CUDAExecutionProvider", image=str(args.image.resolve()), native_precision="fp32")
    write_json(converted._onnx_checkpoint.with_suffix(".json"), report)
    if not report["verified"]:
        raise RuntimeError("RF-DETR ONNX polygon parity failed")
    logging.getLogger(__name__).warning("Verified %d polygons", len(actual.annotations))


if __name__ == "__main__":
    main()
