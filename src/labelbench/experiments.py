"""Persisted, cancellable dataset experiments with exact polygon evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event, Lock
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from labelbench.contracts import RunResult, safe_child_path
from labelbench.evaluation import aggregate, evaluate
from labelbench.fusion import fuse
from labelbench.inference_options import ProviderOptions
from labelbench.service import AnnotationService
from labelbench.storage import read_json, write_json


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    dataset_id: str
    providers: list[str] = Field(min_length=1, max_length=20)
    options: dict[str, ProviderOptions] = Field(default_factory=dict)
    images: list[str] | None = None
    split: Literal["all", "dev", "test"] = "dev"
    seed: int = 42
    limit: int = Field(default=10, ge=1, le=10000)
    methods: list[Literal["union", "nms", "consensus"]] = Field(default_factory=lambda: ["nms", "consensus"])
    match_iou: float = Field(default=0.5, gt=0, le=1)
    fusion_iou: float = Field(default=0.5, gt=0, le=1)
    min_votes: int = Field(default=2, ge=1, le=20)
    text_only: bool = True
    force: bool = False


class ExperimentStore:
    def __init__(self, service: AnnotationService) -> None:
        self.service = service
        self.directory = service.settings.output_dir / "experiments"
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="experiment")
        self._lock = Lock()
        self._active: dict[str, Event] = {}
        # A previous process cannot still own work in this single-server store.
        for path in self.directory.glob("*/experiment.json"):
            item = read_json(path)
            if item["status"] in {"queued", "running"}:
                item.update(status="interrupted", message="Сервер перезапущен; завершённые страницы сохранены")
                write_json(path, item)

    def get(self, identifier: str) -> dict[str, Any]:
        return read_json(safe_child_path(self.directory, f"{identifier}/experiment.json"))

    def list(self) -> list[dict[str, Any]]:
        return [read_json(path) for path in
                sorted(self.directory.glob("*/experiment.json"), key=lambda p: p.stat().st_mtime, reverse=True)]

    def start(self, request: ExperimentRequest) -> dict[str, Any]:
        if set(request.providers) - self.service.providers.keys():
            raise ValueError("Unknown provider")
        if set(request.options) - set(request.providers):
            raise ValueError("Options refer to unselected providers")
        manifest = self.service.datasets.get(request.dataset_id)
        available = {x["name"] for x in manifest["images"]}
        names = list(dict.fromkeys(request.images)) if request.images is not None else sorted(available)
        if set(names) - available:
            raise ValueError("Unknown dataset image")
        # Stable page split; the selected names are also frozen in the experiment manifest.
        if request.split != "all":
            names = [n for n in names if (int(hashlib.sha256(f"{request.seed}:{n}".encode()).hexdigest()[:8], 16) % 5 == 0)
                     == (request.split == "dev")]
        names = names[:request.limit]
        if not names:
            raise ValueError("No images in the selected split")
        identifier = uuid.uuid4().hex[:12]
        item = {"id": identifier, "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "queued", "config": request.model_dump(), "images": names,
                "dataset_sha256": manifest["annotation_sha256"], "completed": 0,
                "elapsed": 0, "message": "В очереди", "events": [], "summary": {}, "errors": [],
                "environment": self._environment()}
        with self._lock:
            if self._active:
                raise ValueError("An experiment is already running; finish or cancel it first")
            event = Event()
            self._active[identifier] = event
            write_json(self.directory / identifier / "experiment.json", item)
            self._future = self._executor.submit(self._run, item, request, event)
        return item

    def cancel(self, identifier: str) -> bool:
        with self._lock:
            event = self._active.get(identifier)
            if event:
                event.set()
            return event is not None

    def page(self, identifier: str, index: int) -> dict[str, Any]:
        item = self.get(identifier)
        if not 0 <= index < len(item["images"]):
            raise ValueError("Invalid page index")
        return read_json(safe_child_path(self.directory, f"{identifier}/pages/{index}.json"))

    def _run(self, item: dict[str, Any], config: ExperimentRequest, event: Event) -> None:
        started = time.monotonic()
        path = self.directory / item["id"] / "experiment.json"
        collected: dict[str, list[dict[str, Any]]] = {}

        def report(message: str) -> None:
            if event.is_set():
                raise InterruptedError("Остановлено пользователем; текущая модель завершена")
            item.update(status="running", elapsed=round(time.monotonic()-started, 2), message=message)
            item["events"] = (item["events"] + [{"seconds": item["elapsed"], "message": message}])[-100:]
            write_json(path, item)

        try:
            for index, name in enumerate(item["images"]):
                report(f"Страница {index+1}/{len(item['images'])}: {name}")
                try:
                    if self.service.datasets.get(config.dataset_id)["annotation_sha256"] != item["dataset_sha256"]:
                        raise InterruptedError("Версия GT изменилась; создайте новый эксперимент")
                    truth = self.service.datasets.truth(config.dataset_id, name)
                    run = self.service.run(name, config.providers, force=config.force, progress=report,
                                           dataset_id=config.dataset_id, options=config.options,
                                           persist=False, allow_legacy=False)
                    outputs = {p: [a for a in r.annotations if not config.text_only or a.label in {"text", "text_line"}]
                               for p, r in run.providers.items()}
                    candidates = [a for rows in outputs.values() for a in rows]
                    outputs.update({f"fusion:{method}": fuse(candidates, method, config.fusion_iou, config.min_votes)
                                    for method in dict.fromkeys(config.methods)})
                    scores = {method: evaluate(annotations, truth, config.match_iou) for method, annotations in outputs.items()}
                    for method, score in scores.items():
                        collected.setdefault(method, []).append(score)
                    payload = {"run": self._lean_run(run), "metrics": scores,
                               "outputs": {k: [a.model_dump(exclude={"mask_rle"}) for a in v] for k, v in outputs.items()}}
                    write_json(path.parent / "pages" / f"{index}.json", payload)
                    item["completed"] += 1
                    item["summary"] = {key: aggregate(rows) for key, rows in collected.items()}
                except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
                    if isinstance(error, InterruptedError):
                        raise
                    item["errors"].append({"image": name, "error": str(error)[:1500]})
                write_json(path, item)
            item["status"] = "partial" if item["errors"] else "done"
            item["message"] = "Эксперимент завершён"
        except InterruptedError as error:
            item.update(status="cancelled", message=str(error))
        except Exception as error:  # noqa: BLE001 - persist unexpected background job failures
            item.update(status="error", message=f"{type(error).__name__}: {error}")
        finally:
            item["elapsed"] = round(time.monotonic()-started, 2)
            write_json(path, item)
            with self._lock:
                self._active.pop(item["id"], None)

    @staticmethod
    def _lean_run(run: RunResult) -> dict[str, Any]:
        return run.model_dump(mode="json", exclude={"providers": {"__all__": {"annotations": {"__all__": {"mask_rle"}}}}})

    def _environment(self) -> dict[str, Any]:
        packages = {}
        for name in ("torch", "transformers", "shapely", "scipy", "numpy", "onnxruntime-gpu"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.service.settings.root_dir,
                                  capture_output=True, text=True, check=False).stdout.strip()
        digest = hashlib.sha256()
        for source in sorted((self.service.settings.root_dir / "src").rglob("*.py")):
            digest.update(source.relative_to(self.service.settings.root_dir).as_posix().encode())
            digest.update(source.read_bytes())
        return {"python": platform.python_version(), "packages": packages, "git_revision": revision,
                "source_sha256": digest.hexdigest(),
                "device": self.service.settings.device, "evaluation": "polygon_matching_v1",
                "split": "sha256(seed:filename)%5; dev=0, test=1..4"}
