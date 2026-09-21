"""Persist review settings and outputs without credentials; optionally evaluate GT."""

import uuid
from typing import Any

from fastapi.encoders import jsonable_encoder

from labelbench.contracts import Annotation, LLMReviewRequest, RunResult
from labelbench.evaluation import evaluate
from labelbench.service import AnnotationService
from labelbench.storage import write_json


def save_review(service: AnnotationService, request: LLMReviewRequest, run: RunResult,
                response: dict[str, Any]) -> dict[str, Any]:
    identifier = uuid.uuid4().hex[:12]
    result = {**response, "artifact_id": identifier}
    if result.get("parsed") and run.dataset_id:
        truth = service.datasets.truth(run.dataset_id, run.image_name)
        predictions = [Annotation.model_validate(item) for item in result.get("refined_annotations", [])]
        result["evaluation"] = evaluate(predictions, truth)
    write_json(service.settings.output_dir / "reviews" / f"{identifier}.json",
               {"request": request.model_dump(mode="json"), "image_sha256": run.image_sha256,
                "dataset_id": run.dataset_id, "response": jsonable_encoder(result)})
    return result
