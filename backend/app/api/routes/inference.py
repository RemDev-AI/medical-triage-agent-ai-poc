# medical-triage-agent-ai-poc/backend/app/api/routes/inference.py

from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from app.api.schemas import (
    GenerateRequest,
    GenerateResponse,
)

from app.api.dependencies.modal import (
    ModalInferenceClient,
    get_modal_client,
)

from app.monitoring.latency_monitor import (
    latency_monitor,
)

from app.monitoring.request_tracker import (
    request_tracker,
)

from app.monitoring.alerting import (
    alert_manager,
)


router = APIRouter(
    prefix="/generate",
    tags=["Inference"],
)


@router.post(
    "/",
    response_model=GenerateResponse,
)
async def generate_route(
    payload: GenerateRequest,
    modal_client: ModalInferenceClient = Depends(
        get_modal_client
    ),
):
    """
    Endpoint de génération générique.

    Nouveau flux :

    Request
        ↓
    Validation Pydantic
        ↓
    Modal Client
        ↓
    Modal GPU
        ↓
    Monitoring
        ↓
    Response
    """

    request_tracker.increment_total_requests()

    start_time = time.perf_counter()

    try:

        modal_response = await modal_client.generate(
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )

        latency_seconds = (
            time.perf_counter() - start_time
        )

        latency_ms = latency_seconds * 1000

        latency_monitor.record(
            latency_ms
        )

        request_tracker.increment_success_requests()

        try:
            alert_manager.evaluate_latency(
                latency_ms
            )
        except Exception:
            pass

        generated_text = modal_response.get(
            "generated_text",
            "",
        )

        model_name = modal_response.get(
            "model_name",
            "Qwen3-Medical-Triage",
        )

        timestamp = modal_response.get(
            "timestamp",
            time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        )

        return GenerateResponse(
            generated_text=generated_text,
            model_name=model_name,
            latency_seconds=round(
                latency_seconds,
                3,
            ),
            timestamp=timestamp,
        )

    except Exception as exc:

        request_tracker.increment_error_requests()

        try:
            alert_manager.raise_alert(
                category="INFERENCE_ERROR",
                message=str(exc),
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail=(
                "Inference generation failed: "
                f"{str(exc)}"
            ),
        )
