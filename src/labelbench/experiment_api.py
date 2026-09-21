"""Dataset and experiment endpoints; existing inference API remains compatible."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from labelbench.contracts import Annotation, ProviderResult, RunResult
from labelbench.evaluation import evaluate
from labelbench.experiments import ExperimentRequest, ExperimentStore
from labelbench.geometry import shape
from labelbench.service import AnnotationService
from labelbench.storage import write_json


class DatasetRequest(BaseModel):
    root: str
    annotation_file: str = "dataset.json"
    name: str | None = None


class EvaluateRequest(BaseModel):
    iou: float = Field(default=0.5, gt=0, le=1)
    text_only: bool = True


class ExternalRequest(BaseModel):
    dataset_id: str
    image_name: str
    method: str = Field(min_length=1, max_length=120)
    annotations: list[dict[str, Any]] = Field(max_length=20000)
    iou: float = Field(default=0.5, gt=0, le=1)


def experiment_router(service: AnnotationService, store: ExperimentStore) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["experiments"])

    @router.get("/datasets")
    def datasets() -> dict[str, Any]:
        archive = service.settings.root_dir / "архив 2"
        return {"datasets": service.datasets.list(), "suggested_root": str(archive) if archive.is_dir() else ""}

    @router.post("/datasets")
    def register(request: DatasetRequest) -> dict[str, Any]:
        try:
            return service.datasets.register(Path(request.root), request.annotation_file, request.name)
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @router.get("/datasets/{identifier}")
    def dataset(identifier: str) -> dict[str, Any]:
        try:
            item = service.datasets.get(identifier)
            return {k: v for k, v in item.items() if k != "root"}
        except (ValueError, OSError) as error:
            raise HTTPException(404, "Dataset not found") from error

    @router.get("/datasets/{identifier}/image")
    def image(identifier: str, name: str) -> FileResponse:
        try:
            path = service.datasets.image_path(identifier, name)
            if not path.is_file():
                raise FileNotFoundError(name)
            return FileResponse(path)
        except (ValueError, OSError) as error:
            raise HTTPException(404, "Image not found") from error

    @router.get("/datasets/{identifier}/truth")
    def truth(identifier: str, name: str) -> dict[str, Any]:
        try:
            return {"annotations": service.datasets.truth(identifier, name)}
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @router.post("/runs/{run_id}/evaluate")
    def evaluate_run(run_id: str, request: EvaluateRequest) -> dict[str, Any]:
        try:
            run = service.load_run(run_id)
            if run.dataset_id is None:
                raise ValueError("Run has no ground-truth dataset")
            truth = service.datasets.truth(run.dataset_id, run.image_name)
            return {p: evaluate([a for a in r.annotations if not request.text_only or a.label in {"text", "text_line"}],
                                truth, request.iou) for p, r in run.providers.items()}
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @router.get("/experiments")
    def experiments() -> dict[str, Any]:
        return {"experiments": store.list()}

    @router.post("/experiments", status_code=202)
    def start(request: ExperimentRequest) -> dict[str, Any]:
        try:
            return store.start(request)
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @router.get("/experiments/{identifier}")
    def get(identifier: str) -> dict[str, Any]:
        try:
            return store.get(identifier)
        except (ValueError, OSError) as error:
            raise HTTPException(404, "Experiment not found") from error

    @router.post("/experiments/{identifier}/cancel")
    def cancel(identifier: str) -> dict[str, bool]:
        return {"requested": store.cancel(identifier)}

    @router.get("/experiments/{identifier}/pages/{index}")
    def page(identifier: str, index: int) -> dict[str, Any]:
        try:
            return store.page(identifier, index)
        except (ValueError, OSError) as error:
            raise HTTPException(404, "Page result not found") from error

    @router.post("/experiments/{identifier}/pages/{index}/open")
    def open_page(identifier: str, index: int, method: str) -> RunResult:
        try:
            page = store.page(identifier, index)
            run = RunResult.model_validate(page["run"])
            if method not in page["outputs"]:
                raise ValueError("Unknown experiment method")
            base = next(iter(run.providers.values()))
            result = base.model_copy(update={"provider": method, "model": method,
                                            "annotations": [Annotation.model_validate(a) for a in page["outputs"][method]]})
            run = run.model_copy(update={"run_id": uuid.uuid4().hex[:12], "providers": {method: result}})
            write_json(service.settings.output_dir / "combined" / run.run_id / "result.json", run.model_dump(mode="json"))
            return run
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @router.post("/evaluations")
    def external(request: ExternalRequest) -> dict[str, Any]:
        try:
            path = service.datasets.image_path(request.dataset_id, request.image_name)
            from PIL import Image

            with Image.open(path) as source:
                size = list(source.size)
            annotations = []
            for index, row in enumerate(request.annotations):
                geometry = shape(row.get("polygon"))
                if geometry is None:
                    raise ValueError(f"Prediction {index} has no valid polygon (pixel coordinates required)")
                x1, y1, x2, y2 = geometry.bounds
                annotations.append(Annotation(id=f"external-{index}", label=row.get("label", "text"),
                                              score=row.get("score", 1), polygon=row["polygon"],
                                              bbox_xywh=[x1, y1, x2-x1, y2-y1], provider=request.method))
            truth = service.datasets.truth(request.dataset_id, request.image_name)
            score = evaluate(annotations, truth, request.iou)
            result = ProviderResult(provider=request.method, model=request.method, image_name=request.image_name,
                                    image_size=size, annotations=annotations, elapsed_seconds=0)
            run = RunResult(run_id=uuid.uuid4().hex[:12], dataset_id=request.dataset_id, image_name=request.image_name,
                            image_size=size, providers={request.method: result}, consensus_score=0)
            target = service.settings.output_dir / "combined" / run.run_id
            write_json(target / "result.json", run.model_dump(mode="json"))
            write_json(target / "evaluation.json", score)
            return {"run": run, "metrics": {request.method: score}}
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    return router
