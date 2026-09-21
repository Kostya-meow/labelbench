import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from labelbench.contracts import Annotation
from labelbench.geometry import shape
from labelbench.robust_analysis import statistics
from labelbench.robust_config import RobustConfig, View
from labelbench.robust_db import RobustDB
from labelbench.robust_fusion import combine
from labelbench.robust_metrics import measure
from labelbench.robust_transforms import restore, transform


def annotation(provider: str, view: str = "clean:1", shift: float = 0) -> Annotation:
    return Annotation(id=f"{provider}/{view}/{shift}", label="text_line", provider=provider, score=.9,
                      polygon=[[10+shift, 10], [80+shift, 10], [80+shift, 25], [10+shift, 25]],
                      bbox_xywh=[10+shift, 10, 70, 15], attributes={"view": view})


@pytest.mark.parametrize("kind,value", [("clean",1),("scale",.8),("rotate",1.5),("rotate",-1.5)])
def test_inverse_geometry_matches_original(kind: str, value: float) -> None:
    source = Image.new("RGB", (1000, 800), "white")
    view, inverse = transform(source, View(kind=kind, value=value), 512, 42, "page")
    original = annotation("a")
    points = np.asarray(original.polygon)
    forward = np.linalg.inv(inverse)
    transformed = (np.column_stack([points, np.ones(len(points))]) @ forward.T)[:, :2]
    row = original.model_copy(update={"polygon": transformed.tolist()})
    result = restore([row], inverse, source.size, "a", kind)
    assert shape(result[0].polygon).symmetric_difference(shape(original.polygon)).area < .001
    assert max(view.size) > 0


def test_noise_is_reproducible_without_files() -> None:
    source = Image.new("RGB", (100, 100), "grey")
    spec = View(kind="noise", value=4)
    first = transform(source, spec, 512, 42, "same")[0].tobytes()
    assert first == transform(source, spec, 512, 42, "same")[0].tobytes()
    assert hashlib.sha256(first).digest() != hashlib.sha256(transform(source, spec, 512, 42, "different")[0].tobytes()).digest()


def test_family_votes_no_fake_independence_and_one_selected_polygon() -> None:
    views = [View(kind="clean"), View(kind="brightness", value=.8)]
    cfg = RobustConfig(dataset_id="test", providers=["ppocr6", "ppocr6_tiny"], views=views)
    rows = [annotation(p, v.key) for p in cfg.providers for v in views]
    results, evidence = combine(rows, cfg)
    assert len(results["joint_medoid"]) == 1
    assert results["joint_stable"] == []
    assert evidence[0]["families"] == 1
    cfg = cfg.model_copy(update={"providers": ["ppocr6", "docufcn"]})
    rows = [annotation(p, v.key) for p in cfg.providers for v in views]
    results, evidence = combine(rows, cfg)
    assert len(results["joint_stable"]) == 1
    assert len(results["mask_vote"]) == 1
    assert evidence[0]["stability"] == 1


def test_duplicate_penalty_and_boundary_metric() -> None:
    a = annotation("a")
    perfect = measure([a], [a], (100, 100), .002)
    assert perfect["f1"] == perfect["boundary_f1"] == perfect["region_iou"] == 1
    duplicate = measure([a, a.model_copy(update={"id": "duplicate"})], [a], (100, 100), .002)
    assert duplicate["fp"] == 1
    assert duplicate["boundary_f1"] < 1


def test_single_model_tta_is_independent_of_other_models() -> None:
    views = [View(kind="clean"), View(kind="brightness", value=.8)]
    own = [annotation("a", v.key) for v in views]
    cfg = RobustConfig(dataset_id="test", providers=["a", "b"], views=views)
    combined, _ = combine(own + [annotation("b", v.key, shift=15) for v in views], cfg)
    isolated, _ = combine(own, cfg.model_copy(update={"providers": ["a"]}))
    assert [a.polygon for a in combined["tta:a"]] == [a.polygon for a in isolated["tta:a"]]


def test_sqlite_checkpoints_survive_reopening(tmp_path: Path) -> None:
    path = tmp_path / "experiment.sqlite"
    db = RobustDB(path)
    cfg = RobustConfig(dataset_id="test", providers=["a"], views=[View(kind="clean")])
    db.create("run", cfg.model_dump(), {"images": ["a.png"]}, {})
    with pytest.raises(ValueError, match="already active"):
        db.create("duplicate", cfg.model_dump(), {"images": ["a.png"]}, {})
    db.save_task("run", "a.png", "a", "clean:1", 1, {"annotations": []})
    reopened = RobustDB(path)
    assert reopened.get("run")["tasks_done"] == 1
    assert reopened.task("run", "a.png", "a", "clean:1")["result"] == {"annotations": []}
    reopened.update("run", cancel=1)
    assert db.cancelled("run")


def test_paired_statistics_equal_methods_have_zero_effect() -> None:
    rows = [{"has_gt": True, "complete": True, "metrics": {
        method: {"tp": tp, "fp": 1, "fn": 2} for method in ["nms", "joint_stable"]}}
        for tp in [2, 4, 8]]
    result = statistics(rows, 42, 100)
    assert result["contrasts"]["nms"]["delta_f1"] == 0
    assert result["contrasts"]["nms"]["ci95"] == [0., 0.]
    assert result["contrasts"]["nms"]["p_holm"] == 1
