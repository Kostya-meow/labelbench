"""FastAPI application exposing the comparison interface and JSON API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from labelbench.contracts import LLMReviewRequest, RunRequest
from labelbench.llm import LMStudioError, list_models, review_run
from labelbench.registry import default_registry
from labelbench.service import AnnotationService
from labelbench.settings import Settings

settings = Settings.from_environment()
settings.ensure_directories()
service = AnnotationService(settings, default_registry(settings))
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_directories()
    yield


app = FastAPI(title="LabelBench", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/files/images", StaticFiles(directory=settings.images_dir), name="images")
app.mount("/files/output", StaticFiles(directory=settings.output_dir), name="output")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "device": settings.device,
        "providers": {
            name: {"available": item.available, "detail": item.detail}
            for name, provider in service.providers.items()
            for item in [provider.availability()]
        },
    }


@app.get("/api/images")
def images() -> dict[str, list[str]]:
    return {"images": service.list_images()}


@app.get("/api/llm/models")
def llm_models() -> dict[str, object]:
    try:
        models = list_models(settings.lm_studio_url, settings.lm_studio_timeout)
        return {"available": True, "base_url": settings.lm_studio_url, "models": models}
    except LMStudioError as error:
        return {"available": False, "base_url": settings.lm_studio_url, "models": [], "detail": str(error)}


@app.post("/api/llm/review")
def llm_review(request: LLMReviewRequest) -> dict[str, object]:
    try:
        run = service.load_run(request.run_id)
        unknown = set(request.providers) - run.providers.keys()
        if unknown:
            raise HTTPException(status_code=400, detail=f"Providers are not in run: {', '.join(sorted(unknown))}")
        image_path = service.settings.images_dir / run.image_name
        return review_run(
            settings.lm_studio_url,
            settings.lm_studio_api_key,
            settings.lm_studio_timeout,
            request.model,
            request.prompt,
            request.history,
            run,
            image_path,
            request.providers,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Run or image not found") from error
    except LMStudioError as error:
        detail = str(error)
        if "Failed to load model" in detail:
            detail += " Откройте LM Studio, загрузите выбранную модель и дождитесь готовности сервера."
        raise HTTPException(status_code=503, detail=detail) from error


@app.post("/api/runs")
def create_run(request: RunRequest):
    try:
        return service.run(request.image_name, request.providers)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        return service.load_run(run_id)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Run not found") from error


@app.get("/api/runs/{run_id}/overlay/{provider}")
def overlay(run_id: str, provider: str) -> FileResponse:
    if provider not in service.providers:
        raise HTTPException(status_code=404, detail="Unknown provider")
    path = settings.output_dir / provider / run_id / "overlay.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Overlay not found")
    return FileResponse(path)
