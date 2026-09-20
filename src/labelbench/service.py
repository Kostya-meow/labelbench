"""Run orchestration, persistence, overlay rendering, and consensus heuristic."""

from __future__ import annotations

import uuid
from pathlib import Path
from threading import Lock

from PIL import Image, ImageDraw

from labelbench.annotation_cache import cache_key, load_cached, load_legacy, save_cached
from labelbench.contracts import Annotation, ProviderResult, RunResult, safe_child_path
from labelbench.progress import Progress, silent_progress
from labelbench.providers.base import AnnotationProvider
from labelbench.settings import Settings

PROVIDER_COLORS = {
    "ppocr": "#f36f38",
    "ppocr6": "#16a085",
    "mask2former": "#29b6a6",
    "sam2": "#9b73e8",
    "yolo26": "#e2b93b",
    "rfdetr_historical": "#dc5a8a",
    "docufcn": "#5378d8",
    "eynollah_textline": "#c45a35",
}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class AnnotationService:
    """Coordinates providers while keeping all writes contained in output."""

    def __init__(self, settings: Settings, providers: dict[str, AnnotationProvider]) -> None:
        self.settings = settings
        self.providers = providers
        self._inference_lock = Lock()

    def list_images(self) -> list[str]:
        return sorted(
            file.relative_to(self.settings.images_dir).as_posix()
            for file in self.settings.images_dir.rglob("*")
            if file.is_file() and file.suffix.lower() in ALLOWED_SUFFIXES
        )

    def run(self, image_name: str, provider_names: list[str], force: bool = False,
            progress: Progress = silent_progress) -> RunResult:
        progress("Проверяю изображение и выбранные детекторы")
        image_path = safe_child_path(self.settings.images_dir, image_name)
        if not image_path.is_file() or image_path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise FileNotFoundError("Image not found or unsupported")
        unknown = set(provider_names) - self.providers.keys()
        if unknown:
            raise KeyError(f"Unknown providers: {', '.join(sorted(unknown))}")
        run_id = uuid.uuid4().hex[:12]
        results: dict[str, ProviderResult] = {}
        for name in provider_names:
            provider = self.providers[name]
            progress(f"{name}: ожидаю очередь обработки")
            with self._inference_lock:
                progress(f"{name}: проверяю сохранённую разметку")
                key = cache_key(image_path, provider)
                result = None if force else load_cached(self.settings.output_dir, key)
                if result is None and not force:
                    result = load_legacy(self.settings.output_dir, image_path, image_name, provider)
                if result is None:
                    progress(f"{name}: проверяю окружение, загружаю модель и выполняю inference")
                    status = provider.availability()
                    if not status.available:
                        raise RuntimeError(f"{name} unavailable: {status.detail}")
                    result = provider.annotate(image_path)
                    # First inference may download a checkpoint used by the signature.
                    key = cache_key(image_path, provider)
                else:
                    progress(f"{name}: использую сохранённую разметку")
                result = result.model_copy(update={"image_name": image_name})
                result = save_cached(self.settings.output_dir, key, result)
            self._write_provider_result(run_id, result)
            progress(f"{name}: {len(result.annotations)} сегментов; сохраняю JSON и отрисовку")
            self._draw_overlay(run_id, image_path, result)
            results[name] = result
        with Image.open(image_path) as image:
            image_size = [image.width, image.height]
        merged = RunResult(
            run_id=run_id,
            image_name=image_name,
            image_size=image_size,
            providers=results,
            consensus_score=self._consensus(results),
        )
        combined_dir = self.settings.output_dir / "combined" / run_id
        progress("Сохраняю общий результат и изображение")
        combined_dir.mkdir(parents=True, exist_ok=True)
        self._draw_combined_overlay(run_id, image_path, results)
        (combined_dir / "result.json").write_text(merged.model_dump_json(indent=2), encoding="utf-8")
        return merged

    def load_run(self, run_id: str) -> RunResult:
        path = safe_child_path(self.settings.output_dir / "combined", f"{run_id}/result.json")
        return RunResult.model_validate_json(path.read_text(encoding="utf-8"))

    def _write_provider_result(self, run_id: str, result: ProviderResult) -> None:
        target = self.settings.output_dir / result.provider / run_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")

    def _draw_overlay(self, run_id: str, image_path: Path, result: ProviderResult) -> None:
        with Image.open(image_path).convert("RGBA") as source:
            canvas = source.copy()
        draw = ImageDraw.Draw(canvas, "RGBA")
        color = PROVIDER_COLORS.get(result.provider, "#ffffff")
        for annotation in result.annotations:
            self._draw_annotation(draw, annotation, color)
        target = self.settings.output_dir / result.provider / run_id / "overlay.png"
        canvas.convert("RGB").save(target, quality=92)

    def _draw_combined_overlay(
        self, run_id: str, image_path: Path, results: dict[str, ProviderResult]
    ) -> None:
        with Image.open(image_path).convert("RGBA") as source:
            canvas = source.copy()
        draw = ImageDraw.Draw(canvas, "RGBA")
        for result in results.values():
            color = PROVIDER_COLORS.get(result.provider, "#ffffff")
            for annotation in result.annotations:
                self._draw_annotation(draw, annotation, color)
        target = self.settings.output_dir / "combined" / run_id / "overlay.png"
        canvas.convert("RGB").save(target, quality=92)

    @staticmethod
    def _draw_annotation(draw: ImageDraw.ImageDraw, annotation: Annotation, color: str) -> None:
        x, y, width, height = annotation.bbox_xywh
        if annotation.polygon:
            draw.polygon([tuple(point) for point in annotation.polygon], outline=color, fill=None, width=3)
        else:
            draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
        label = annotation.text or annotation.label
        draw.text((x + 3, max(0, y - 16)), f"{label} {annotation.score:.2f}", fill=color)

    @staticmethod
    def _consensus(results: dict[str, ProviderResult]) -> float:
        boxes = [annotation.bbox_xywh for result in results.values() for annotation in result.annotations]
        if len(boxes) < 2:
            return 0.0
        overlaps = [
            AnnotationService._iou(first, second)
            for index, first in enumerate(boxes)
            for second in boxes[index + 1 :]
        ]
        return round(max(overlaps, default=0.0), 4)

    @staticmethod
    def _iou(first: list[float], second: list[float]) -> float:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        left, top = max(ax, bx), max(ay, by)
        right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = aw * ah + bw * bh - intersection
        return intersection / union if union else 0.0
