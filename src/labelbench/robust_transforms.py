"""Deterministic in-memory views and inverse polygon geometry."""

import hashlib
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from shapely.geometry import Polygon, box

from labelbench.contracts import Annotation
from labelbench.geometry import clean_polygon
from labelbench.robust_config import View


def transform(source: Image.Image, view: View, max_side: int, seed: int,
              identity: str) -> tuple[Image.Image, np.ndarray]:
    width, height = source.size
    factor = min(1., max_side / max(width, height))
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    image = source.resize(size, Image.Resampling.LANCZOS)
    forward = np.diag([size[0] / width, size[1] / height, 1.])
    if view.kind == "brightness":
        image = ImageEnhance.Brightness(image).enhance(view.value)
    elif view.kind == "contrast":
        image = ImageEnhance.Contrast(image).enhance(view.value)
    elif view.kind == "blur":
        image = image.filter(ImageFilter.GaussianBlur(view.value))
    elif view.kind == "noise":
        digest = hashlib.sha256(f"{seed}:{identity}:{view.key}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        values = np.asarray(image).astype(float)
        image = Image.fromarray(np.clip(values + rng.normal(0, view.value, values.shape), 0, 255).astype(np.uint8))
    elif view.kind == "jpeg":
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=round(view.value))
        stream.seek(0)
        with Image.open(stream) as compressed:
            image = compressed.convert("RGB")
    elif view.kind == "scale":
        new_size = (round(image.width * view.value), round(image.height * view.value))
        forward = np.diag([new_size[0] / image.width, new_size[1] / image.height, 1.]) @ forward
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    elif view.kind == "rotate":
        import cv2

        w, h = image.size
        affine = cv2.getRotationMatrix2D((w / 2, h / 2), view.value, 1.)
        cos, sin = abs(affine[0, 0]), abs(affine[0, 1])
        new_w, new_h = int(np.ceil(w * cos + h * sin)), int(np.ceil(h * cos + w * sin))
        affine[:, 2] += [(new_w - w) / 2, (new_h - h) / 2]
        array = cv2.warpAffine(np.asarray(image), affine, (new_w, new_h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))
        image = Image.fromarray(array)
        forward = np.vstack([affine, [0, 0, 1]]) @ forward
    return image, np.linalg.inv(forward)


def restore(rows: list[Annotation], inverse: np.ndarray, size: tuple[int, int],
            provider: str, view: str) -> list[Annotation]:
    result = []
    for index, item in enumerate(rows):
        if not item.polygon:
            continue
        points = np.asarray(item.polygon)
        restored = (np.column_stack([points, np.ones(len(points))]) @ inverse.T)[:, :2]
        polygon = clean_polygon(restored.tolist())
        if polygon is None:
            continue
        clipped = Polygon(polygon).intersection(box(0, 0, *size))
        if clipped.geom_type != "Polygon" or clipped.area <= 0:
            continue
        polygon = [[round(x, 4), round(y, 4)] for x, y in clipped.exterior.coords[:-1]]
        x1, y1, x2, y2 = clipped.bounds
        result.append(Annotation(id=f"{provider}/{view}/{index}", label="text_line", score=item.score,
                                 polygon=polygon, bbox_xywh=[x1, y1, x2-x1, y2-y1], provider=provider,
                                 attributes={"view": view, "source_id": item.id}))
    return result
