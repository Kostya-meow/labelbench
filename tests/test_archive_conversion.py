import json
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from labelbench.archive_conversion import convert_archive
from labelbench.datasets import DatasetStore


def test_snapshot_preserves_line_polygons_and_excludes_incomplete_pages(tmp_path: Path) -> None:
    archive = tmp_path / "original"
    archive.mkdir()
    for name in ["line.jpg", "bad.jpg"]:
        Image.new("RGB", (200, 100)).save(archive / name)
    polygon = [[.1, .2], [.8, .2], [.8, .4], [.1, .4]]
    with sqlite3.connect(archive / "annotations.db") as connection:
        connection.executescript(
            "CREATE TABLE annotations(id INTEGER,image_name TEXT,polygon_json TEXT,line_id INTEGER);"
            "CREATE TABLE page_view_settings(image_name TEXT,rotation_degrees REAL);"
        )
        connection.executemany("INSERT INTO annotations VALUES(?,?,?,?)", [
            (1, "line.jpg", json.dumps(polygon), 10),
            (2, "line.jpg", json.dumps(polygon), 10),
            (3, "bad.jpg", None, 11),
            (4, "missing.jpg", json.dumps(polygon), 12),
        ])
    before = (archive / "annotations.db").read_bytes()
    report = convert_archive(archive, tmp_path / "converted", tmp_path / "output")
    assert report["images"] == 1
    assert report["polygons"] == 2  # Shared line_id never silently merges objects.
    assert report["excluded_images"] == ["bad.jpg"]
    assert report["missing_images"] == ["missing.jpg"]
    store = DatasetStore(tmp_path / "output")
    truth = store.truth(report["dataset_id"], "line.jpg")
    assert truth[0].polygon == [[20, 20], [160, 20], [160, 40], [20, 40]]
    assert store.image_path(report["dataset_id"], "line.jpg") == archive / "line.jpg"
    assert before == (archive / "annotations.db").read_bytes()
    with pytest.raises(ValueError, match="new destination"):
        convert_archive(archive, tmp_path / "converted", tmp_path / "output")


def test_dataset_versions_do_not_overwrite_old_manifests(tmp_path: Path) -> None:
    Image.new("RGB", (100, 100)).save(tmp_path / "page.jpg")
    source = tmp_path / "dataset.json"
    source.write_text(json.dumps({"page.jpg": {"free_words": []}}))
    store = DatasetStore(tmp_path / "output")
    first = store.register(tmp_path)
    source.write_text(json.dumps({"page.jpg": {"free_words": [], "note": "updated"}}))
    second = store.register(tmp_path)
    assert first["id"] != second["id"]
    assert store.get(first["id"])["annotation_sha256"] == first["annotation_sha256"]
    with pytest.raises(ValueError, match="Ground truth changed"):
        store.truth(first["id"], "page.jpg")
