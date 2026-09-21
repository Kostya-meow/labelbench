"""Read-only custom archive datasets: each explicit polygon is one GT instance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any

from PIL import Image

from labelbench.contracts import Annotation, safe_child_path
from labelbench.geometry import shape
from labelbench.storage import write_json


def extract_polygons(value: Any) -> list[list[list[float]]]:
    if isinstance(value, dict):
        result = [value["polygon"]] if "polygon" in value else []
        for key, child in value.items():
            if key != "polygon":
                result.extend(extract_polygons(child))
        return result
    if isinstance(value, list):
        return [polygon for child in value for polygon in extract_polygons(child)]
    return []


class DatasetStore:
    def __init__(self, output: Path) -> None:
        self.directory = output / "datasets"
        self._documents: dict[str, tuple[int, dict[str, Any]]] = {}
        self._lock = RLock()

    def register(self, root: Path, annotation_file: str = "dataset.json",
                 name: str | None = None, *, image_root: Path | None = None,
                 unit: str = "explicit_polygon") -> dict[str, Any]:
        root = root.resolve()
        source = safe_child_path(root, annotation_file)
        image_root = image_root.resolve() if image_root else root
        raw = source.read_bytes()
        document = json.loads(raw)
        if not isinstance(document, dict) or not document:
            raise ValueError("Expected a nonempty map: image name -> blocks/free_lines/free_words")
        identifier = hashlib.sha256(str(source).encode() + raw).hexdigest()[:12]
        images, missing = [], []
        polygon_count = invalid = outside = 0
        for image_name, page in document.items():
            if not isinstance(page, dict) or not any(k in page for k in ("blocks", "free_lines", "free_words")):
                raise ValueError(f"Unsupported page schema: {image_name}")
            path = safe_child_path(image_root, image_name)
            if not path.is_file():
                missing.append(image_name)
                continue
            with Image.open(path) as image:
                width, height = image.size
            polygons = extract_polygons(page)
            bad = sum(shape(points) is None for points in polygons)
            invalid += bad
            outside += sum(any(x < 0 or y < 0 or x > width or y > height for x, y in points)
                           for points in polygons if shape(points) is not None)
            polygon_count += len(polygons)
            images.append({"name": image_name, "size": [width, height], "objects": len(polygons),
                           "invalid": bad})
        manifest = {"id": identifier, "name": name or root.name, "root": str(image_root),
                    "annotation_root": str(root),
                    "annotation_file": annotation_file, "annotation_sha256": hashlib.sha256(raw).hexdigest(),
                    "unit": unit, "images": images, "missing_images": missing,
                    "polygons": polygon_count, "invalid_polygons": invalid, "outside_polygons": outside}
        write_json(self.directory / f"{identifier}.json", manifest)
        with self._lock:
            self._documents[identifier] = (source.stat().st_mtime_ns, document)
        return manifest

    def get(self, identifier: str) -> dict[str, Any]:
        return json.loads(safe_child_path(self.directory, f"{identifier}.json").read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.directory.glob("*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            result.append({**{k: v for k, v in item.items() if k not in {"images", "root"}},
                           "image_count": len(item["images"])})
        return result

    def image_path(self, identifier: str, image_name: str) -> Path:
        manifest = self.get(identifier)
        if image_name not in {item["name"] for item in manifest["images"]}:
            raise ValueError("Image is not in the dataset manifest")
        return safe_child_path(Path(manifest["root"]), image_name)

    def truth(self, identifier: str, image_name: str) -> list[Annotation]:
        manifest = self.get(identifier)
        self.image_path(identifier, image_name)
        source = safe_child_path(Path(manifest.get("annotation_root", manifest["root"])), manifest["annotation_file"])
        with self._lock:
            cached = self._documents.get(identifier)
            if cached is None or cached[0] != source.stat().st_mtime_ns:
                raw = source.read_bytes()
                if hashlib.sha256(raw).hexdigest() != manifest["annotation_sha256"]:
                    raise ValueError("Ground truth changed: re-import dataset to create a new version")
                cached = (source.stat().st_mtime_ns, json.loads(raw))
                self._documents[identifier] = cached
            polygons = extract_polygons(cached[1][image_name])
        result = []
        for index, points in enumerate(polygons):
            geometry = shape(points)
            if geometry is None:
                raise ValueError(f"Invalid GT polygon {index} in {image_name}")
            x1, y1, x2, y2 = geometry.bounds
            label = "text_line" if manifest["unit"] == "stored_line_polygon" else "text"
            result.append(Annotation(id=f"gt-{index}", label=label, score=1,
                                     bbox_xywh=[x1, y1, x2-x1, y2-y1], polygon=points,
                                     provider="ground_truth"))
        return result
