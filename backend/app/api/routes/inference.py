# medical-triage-agent-ai-poc/backend/app/api/routes/inference.py

from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from backend.app.api.schemas import (
    GenerateRequest,
    GenerateResponse,
)

from backend.app.api.dependencies.inference import (
    InferenceClient,
    get_inference_client,
)

from backend.app.monitoring.alerting import (
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
    inference_client: InferenceClient = Depends(
        get_inference_client,
    ),
):
    """
    Endpoint de génération générique.

    Flux d'exécution :

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
    Response

    NOTE (correctif étape 3) :
    Le comptage global des requêtes et la latence
    globale sont déjà assurés par
    AuditLoggingMiddleware (request_tracker,
    latency_monitor) pour TOUTES les routes.
    Cette route ne doit donc plus appeler
    request_tracker.increment_*() ni
    latency_monitor.record() elle-même, sous peine
    de compter chaque requête deux fois.
    Le chronométrage local ci-dessous sert
    uniquement à renseigner le champ
    `latency_seconds` de la réponse métier.
    """

    start_time = time.perf_counter()

    try:

        inference_response = await inference_client.generate(
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )

        latency_seconds = time.perf_counter() - start_time

        latency_ms = latency_seconds * 1000

        try:
            alert_manager.evaluate_latency(latency_ms)
        except Exception:
            pass

        generated_text = inference_response.get(
            "generated_text",
            "",
        )

        model_name = inference_response.get(
            "model_name",
            "Qwen3-Medical-Triage",
        )

        timestamp = inference_response.get(
            "timestamp",
            time.strftime("%Y-%m-%dT%H:%M:%S"),
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

        try:
            alert_manager.raise_alert(
                category="INFERENCE_ERROR",
                message=str(exc),
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail=("Inference generation failed: " f"{str(exc)}"),
        )
