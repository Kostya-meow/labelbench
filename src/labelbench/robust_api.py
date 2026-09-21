"""Robustness dashboard API and on-demand original-image overlays."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from labelbench.contracts import safe_child_path
from labelbench.datasets import DatasetStore
from labelbench.robust_config import RobustConfig
from labelbench.robust_manager import RobustManager


class PageRequest(BaseModel):
    image: str
    method: str = "joint_stable"


def robust_router(manager: RobustManager, old_experiments: object) -> APIRouter:
    router = APIRouter(prefix="/api/robustness", tags=["robustness"])

    @router.get("")
    def history() -> dict:
        return {"runs": [{k: r[k] for k in ("id", "status", "created", "pages_done", "tasks_done", "tasks_total", "message")}
                         | {"name": r["config"]["name"]} for r in manager.list()]}

    @router.post("", status_code=202)
    def start(config: RobustConfig) -> dict:
        try:
            if getattr(old_experiments, "_active", {}):
                raise ValueError("An ordinary experiment is already using the GPU")
            return manager.start(config)
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @router.get("/{identifier}")
    def status(identifier: str) -> dict:
        try:
            manager.list()
            item = manager.db.get(identifier)
            item["manifest"] = {k: v for k, v in item["manifest"].items() if k not in {"root", "image_sha256"}}
            return item
        except (ValueError, OSError) as error:
            raise HTTPException(404, str(error)) from error

    @router.post("/{identifier}/cancel")
    def cancel(identifier: str) -> dict:
        try:
            manager.cancel(identifier)
            return {"requested": True}
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    @router.post("/{identifier}/resume")
    def resume(identifier: str) -> dict:
        try:
            return manager.resume(identifier)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @router.get("/{identifier}/page")
    def page(identifier: str, image: str, method: str = "joint_stable") -> dict:
        try:
            item = manager.db.get(identifier)
            if image not in item["manifest"]["images"]:
                raise ValueError("Image is not in experiment")
            payload = manager.db.page(identifier, image)
            if payload is None:
                return {"ready": False, "message": "Страница ещё обрабатывается"}
            truth = DatasetStore(manager.settings.output_dir).truth(item["config"]["dataset_id"], image) if payload["has_gt"] else []
            return {"ready": True, "image": image, "size": payload["size"], "methods": list(payload["outputs"]),
                    "annotations": payload["outputs"].get(method, []), "evidence": payload["evidence"],
                    "truth": [a.model_dump() for a in truth], "has_gt": payload["has_gt"],
                    "metrics": payload["metrics"].get(method), "errors": payload["errors"]}
        except (ValueError, OSError) as error:
            raise HTTPException(400, str(error)) from error

    @router.get("/{identifier}/image")
    def image(identifier: str, name: str) -> FileResponse:
        try:
            item = manager.db.get(identifier)
            if name not in item["manifest"]["images"]:
                raise ValueError("Unknown image")
            return FileResponse(safe_child_path(Path(item["manifest"]["root"]), name))
        except (ValueError, OSError) as error:
            raise HTTPException(404, str(error)) from error

    @router.get("/{identifier}/artifact")
    def artifact(identifier: str, name: str) -> FileResponse:
        allowed = {"analysis.json", "page-metrics.csv", "analysis-report.md", "protocol.json",
                   "figures/method-f1.png", "figures/corruption-f1.png", "figures/trust-risk.png"}
        if name not in allowed:
            raise HTTPException(404, "Unknown artifact")
        target = safe_child_path(manager.directory, f"{identifier}/{name}")
        if not target.is_file():
            raise HTTPException(404, "Artifact will appear after analysis completes")
        return FileResponse(target, filename=target.name)

    return router
