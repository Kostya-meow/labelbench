"""Run in .venv-manuscript: numerical checks for the optional runtime optimizations."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module(name: str) -> object:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_roi_iou_matches_full_masks_exactly() -> None:
    pytest.importorskip("manuscript")
    cls = module("manuscript_runtime").MemoryEfficientMask2Former
    detector = cls.__new__(cls)
    detector._mask_metadata = {}
    rng = np.random.default_rng(42)
    masks = [rng.random((43, 67)) > .85 for _ in range(20)]
    masks.extend([np.zeros((43, 67), dtype=bool), np.ones((43, 67), dtype=bool)])
    for a in masks:
        for b in masks:
            union = np.logical_or(a, b).sum()
            expected = float(np.logical_and(a, b).sum() / union) if union else 0.
            assert detector._mask_iou(a, b) == expected


def test_mask_einsum_rewrite_preserves_dynamic_batch(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx.reference import ReferenceEvaluator

    h = onnx.helper
    graph = h.make_graph([h.make_node("Einsum", ["a", "b"], ["out"], equation="bqc, bchw -> bqhw")], "mask",
                         [h.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [None, 3, 4]),
                          h.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [None, 4, 5, 6])],
                         [h.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [None, 3, 5, 6])])
    source, target = tmp_path / "source.onnx", tmp_path / "target.onnx"
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 18)]), source)
    assert module("optimize_manuscript_onnx").optimize(source, target) == 1
    rng = np.random.default_rng(42)
    for batch in [1, 2]:
        feed = {"a": rng.normal(size=(batch, 3, 4)).astype(np.float32),
                "b": rng.normal(size=(batch, 4, 5, 6)).astype(np.float32)}
        first = ReferenceEvaluator(str(source)).run(None, feed)[0]
        second = ReferenceEvaluator(str(target)).run(None, feed)[0]
        np.testing.assert_allclose(first, second, atol=1e-6, rtol=1e-6)
