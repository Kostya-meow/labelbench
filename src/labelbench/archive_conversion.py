"""Snapshot the annotation editor database; retain each stored line polygon."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from PIL import Image

from labelbench.contracts import safe_child_path
from labelbench.datasets import DatasetStore
from labelbench.geometry import shape
from labelbench.storage import write_json


def convert_archive(root: Path, destination: Path, output: Path) -> dict[str, Any]:
    root, destination = root.resolve(), destination.resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Conversion must be outside the original archive")
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = destination / "annotations.snapshot.db"
    if snapshot.exists():
        raise ValueError("Use a new destination to preserve previous snapshots")
    with closing(sqlite3.connect((root / "annotations.db").as_uri() + "?mode=ro", uri=True)) as source, closing(sqlite3.connect(snapshot)) as target:
        source.backup(target)
    pages: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    missing: set[str] = set()
    with closing(sqlite3.connect(snapshot)) as connection:
        rows = connection.execute("SELECT id,image_name,polygon_json,line_id FROM annotations ORDER BY image_name,id")
        for identifier, name, raw, line_id in rows:
            path = safe_child_path(root, name)
            if not path.is_file():
                missing.add(name)
                continue
            if name not in pages:
                with Image.open(path) as image:
                    size = list(image.size)
                pages[name] = {"free_words": [], "image_size": size}
            page = pages[name]
            try:
                points = json.loads(raw)
                if not points or any(not 0 <= v <= 1 for point in points for v in point):
                    raise ValueError("Expected normalized polygon")
                polygon = [[x * page["image_size"][0], y * page["image_size"][1]] for x, y in points]
                if shape(polygon) is None:
                    raise ValueError("Invalid polygon")
            except (ValueError, TypeError) as error:
                issues.append({"image": name, "id": identifier, "error": str(error)})
                continue
            page["free_words"].append({"polygon": polygon, "source_id": identifier,
                                       "source_line_id": line_id, "label": "text_line"})
        rotations = connection.execute(
            "SELECT image_name,rotation_degrees FROM page_view_settings WHERE rotation_degrees != 0"
        ).fetchall()
    # A page with discarded geometry cannot be treated as complete ground truth.
    excluded = {item["image"] for item in issues}
    document = {name: page for name, page in pages.items() if name not in excluded}
    write_json(destination / "dataset.json", document)
    manifest = DatasetStore(output).register(destination, name="Archive DB — line polygons",
                                              image_root=root, unit="stored_line_polygon")
    report = {"dataset_id": manifest["id"], "source": str(root),
              "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              "images": len(manifest["images"]), "polygons": manifest["polygons"],
              "missing_images": sorted(missing), "excluded_images": sorted(excluded),
              "invalid_objects": issues, "view_rotations": rotations,
              "policy": "Each DB polygon retained; normalized coordinates scaled to original image size. No grouping, merging, or view rotation applied."}
    write_json(destination / "conversion-report.json", report)
    return report
