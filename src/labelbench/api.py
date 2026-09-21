"""FastAPI application exposing the comparison interface and JSON API."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from labelbench.cancellation import ReviewCancelled, current_cancellation
from labelbench.contracts import LLMReviewRequest, RunRequest
from labelbench.experiment_api import experiment_router
from labelbench.experiments import ExperimentStore
from labelbench.llm import LMStudioError, list_models, review_run
from labelbench.progress import Progress, ProgressJournal, silent_progress
from labelbench.registry import default_registry
from labelbench.review_store import save_review
from labelbench.robust_api import robust_router
from labelbench.robust_manager import RobustManager
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
app.state.progress = ProgressJournal()
app.state.experiments = ExperimentStore(service)
app.state.robustness = RobustManager(settings)
app.include_router(experiment_router(service, app.state.experiments, app.state.robustness.active))
app.include_router(robust_router(app.state.robustness, app.state.experiments))
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
    return tracked_request(request, _llm_review)


def tracked_request(request: RunRequest | LLMReviewRequest,
                    operation: Callable[..., Any]) -> Any:
    journal = app.state.progress
    try:
        report = journal.start(request.progress_id, cancellable=isinstance(request, LLMReviewRequest))
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    report("Запрос принят сервером")
    cancellation = journal.token(request.progress_id)
    context = current_cancellation.set(cancellation)

    def checked_report(message: str) -> None:
        if cancellation:
            cancellation.check()
        report(message)

    try:
        result = operation(request, checked_report)
        if cancellation:
            cancellation.check()
        status = "done"
        if isinstance(result, dict) and not result.get("complete", True):
            status = "partial" if result.get("parsed") else "error"
        report({"done": "Готово", "partial": "Получен частичный результат", "error": "Ответ модели не получен"}[status])
        journal.finish(request.progress_id, status)
        return result
    except ReviewCancelled as error:
        report("Остановлено пользователем. Следующие части и повторы не отправляются.")
        journal.finish(request.progress_id, "cancelled")
        raise HTTPException(status_code=499, detail="Проверка VLM отменена. Исходная разметка сохранена.") from error
    except Exception as error:
        report(str(error.detail) if isinstance(error, HTTPException) else "Ошибка обработки; подробности в ответе API")
        journal.finish(request.progress_id, "error")
        raise
    finally:
        current_cancellation.reset(context)


@app.post("/api/llm/cancel/{identifier}")
def cancel_llm(identifier: str) -> dict[str, object]:
    try:
        return {"requested": app.state.progress.cancel(identifier)}
    except KeyError as error:
        raise HTTPException(status_code=404, detail="VLM request not found") from error


@app.get("/api/progress/{identifier}")
def get_progress(identifier: str) -> dict[str, object]:
    result = app.state.progress.snapshot(identifier)
    if result is None:
        raise HTTPException(status_code=404, detail="Progress not found")
    return result


def _llm_review(request: LLMReviewRequest, progress: Progress = silent_progress) -> dict[str, object]:
    try:
        api_key = request.api_key.strip() if request.backend == "routerai" else settings.lm_studio_api_key
        if request.backend == "routerai" and (not api_key or "\r" in api_key or "\n" in api_key):
            raise HTTPException(status_code=400, detail="Укажите корректный API-ключ RouterAI.")
        base_url = "https://routerai.ru/api/v1" if request.backend == "routerai" else settings.lm_studio_url
        run = service.load_run(request.run_id)
        unknown = set(request.providers) - run.providers.keys()
        if unknown:
            raise HTTPException(status_code=400, detail=f"Providers are not in run: {', '.join(sorted(unknown))}")
        if request.annotation_ids is not None:
            selected_ids = set(request.annotation_ids)
            known_ids = {a.id for p, r in run.providers.items() if p in request.providers for a in r.annotations}
            if selected_ids - known_ids:
                raise HTTPException(400, "Unknown annotation IDs")
            run = run.model_copy(update={"providers": {
                p: r.model_copy(update={"annotations": [a for a in r.annotations if a.id in selected_ids]})
                for p, r in run.providers.items() if p in request.providers
            }})
        image_path = service.image_path(run.image_name, run.dataset_id)
        response = review_run(
            base_url,
            api_key,
            settings.lm_studio_timeout,
            request.model,
            request.prompt,
            request.history,
            run,
            image_path,
            request.providers,
            backend=request.backend,
            request_mode=request.request_mode,
            max_output_tokens=request.max_output_tokens,
            progress=progress,
        )
        cancellation = current_cancellation.get()
        if cancellation:
            cancellation.check()
        return save_review(service, request, run, response)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Run or image not found") from error
    except LMStudioError as error:
        detail = str(error)
        if "Failed to load model" in detail:
            detail += " Откройте LM Studio, загрузите выбранную модель и дождитесь готовности сервера."
        raise HTTPException(status_code=503, detail=detail) from error


@app.post("/api/runs")
def create_run(request: RunRequest) -> Any:
    return tracked_request(request, _create_run)


def _create_run(request: RunRequest, progress: Progress = silent_progress) -> Any:
    if app.state.robustness.active():
        raise HTTPException(409, "GPU занят фоновым экспериментом. Сохранённые результаты доступны во вкладке Устойчивость.")
    try:
        return service.run(request.image_name, request.providers, force=request.force, progress=progress,
                           dataset_id=request.dataset_id, options=request.options)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except subprocess.CalledProcessError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Inference worker failed with exit code {error.returncode}. "
                "Запустите сервер через bat\\START.bat, чтобы подключились отдельные GPU-окружения."
            ),
        ) from error
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
    from labelbench.contracts import safe_child_path
    try:
        path = safe_child_path(settings.output_dir / provider, f"{run_id}/overlay.png")
    except ValueError as error:
        raise HTTPException(404, "Overlay not found") from error
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Overlay not found")
    return FileResponse(path)
