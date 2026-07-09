# medical-triage-agent-ai-poc/backend/app/api/routes/triage.py

from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from backend.app.api.schemas import (
    TriageRequest,
    TriageResponse,
)

from backend.app.api.dependencies.inference import (
    InferenceClient,
    get_inference_client,
)

from backend.app.monitoring.alerting import (
    alert_manager,
)


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
    inference_client: InferenceClient = Depends(
        get_inference_client,
    ),
):
    """
    Endpoint principal de triage médical.

    Pipeline d'exécution :

    Request
        ↓
    Validation Pydantic
        ↓
    InferenceClient
        ↓
    Backend d'inférence
        ↓
    Monitoring (AuditLoggingMiddleware)
        ↓
    Audit (audit_store, cf. routes/audit.py)
        ↓
    Response

    NOTE (correctif étape 3) :
    Le comptage global des requêtes et la latence
    globale sont déjà assurés par
    AuditLoggingMiddleware pour TOUTES les routes.
    Cette route ne doit donc plus appeler
    request_tracker.increment_*() ni
    latency_monitor.record() elle-même.
    """

    start_time = time.perf_counter()

    try:

        triage_result = await inference_client.triage(
            symptoms=payload.symptoms,
            medical_history=payload.medical_history,
            age=payload.age,
            priority_context=payload.priority_context,
        )

        latency_seconds = (
            time.perf_counter() - start_time
        )

        latency_ms = latency_seconds * 1000

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
