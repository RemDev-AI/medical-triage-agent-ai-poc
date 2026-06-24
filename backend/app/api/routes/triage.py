# medical-triage-agent-ai-poc/backend/app/api/routes/triage.py

from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from backend.app.api.schemas import TriageRequest, TriageResponse

from backend.app.api.dependencies.modal import ModalInferenceClient, get_modal_client  # noqa : E501

from backend.app.monitoring.latency_monitor import latency_monitor

from backend.app.monitoring.request_tracker import request_tracker

from backend.app.monitoring.alerting import alert_manager


router = APIRouter(
    prefix="/triage",
    tags=["Triage"],
)


@router.post(
    "/",
    response_model=TriageResponse,
)
async def triage_route(
    payload: TriageRequest,
    modal_client: ModalInferenceClient = Depends(
        get_modal_client
    ),
):
    """
    Endpoint principal de triage médical.

    Nouveau pipeline :

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
    Audit
        ↓
    Response
    """

    request_tracker.increment_total_requests()

    start_time = time.perf_counter()

    try:

        triage_result = await modal_client.triage(
            symptoms=payload.symptoms,
            medical_history=payload.medical_history,
            age=payload.age,
            priority_context=payload.priority_context,
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

        return TriageResponse(
            priority_level=triage_result.get(
                "priority_level",
                "UNKNOWN",
            ),
            justification=triage_result.get(
                "justification",
                "",
            ),
            recommendations=triage_result.get(
                "recommendations",
                [],
            ),
            confidence_score=triage_result.get(
                "confidence_score",
                0.0,
            ),
            generated_at=triage_result.get(
                "generated_at",
                time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
            ),
            latency_seconds=round(
                latency_seconds,
                3,
            ),
        )

    except Exception as exc:

        request_tracker.increment_error_requests()

        try:
            alert_manager.raise_alert(
                category="TRIAGE_ERROR",
                message=str(exc),
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail=(
                "Triage engine failed: "
                f"{str(exc)}"
            ),
        )
