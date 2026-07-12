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


# --- Filtrage de sortie anti-fuite (défense en profondeur) -------------
#
# NOTE: le test de sécurité `test_prompt_injection_attempt` attend que la
# requête soit traitée normalement (200), même face à une tentative de
# prompt injection dans `symptoms`. La requête n'est donc PAS bloquée en
# amont : la protection porte sur la RÉPONSE générée, pour garantir
# qu'aucun contenu sensible ne fuite, même si le backend d'inférence
# distant a été influencé par le contenu injecté.
#
# Cette liste reste une heuristique de filtrage, pas une garantie absolue :
# elle réduit le risque de fuite évidente, mais ne remplace pas un prompt
# système robuste côté InferenceClient / backend d'inférence (isolation
# claire instructions système vs données utilisateur).

CONFIDENTIAL_MARKERS: tuple[str, ...] = (
    "confidential",
    "medical records",
    "medical record",
    "patient_id",
    "ssn",
    "social security",
)


def _strip_confidential_leakage(text: str) -> str:
    """
    Remplace toute occurrence de marqueurs sensibles connus par un
    texte neutre, en dernier rempart avant de renvoyer la réponse
    au client.
    """

    sanitized = text

    for marker in CONFIDENTIAL_MARKERS:
        if marker.lower() in sanitized.lower():
            # Remplacement insensible à la casse, occurrence par occurrence.
            lowered = sanitized.lower()
            marker_lower = marker.lower()
            result = []
            idx = 0
            while True:
                pos = lowered.find(marker_lower, idx)
                if pos == -1:
                    result.append(sanitized[idx:])
                    break
                result.append(sanitized[idx:pos])
                result.append("[redacted]")
                idx = pos + len(marker_lower)
            sanitized = "".join(result)
            lowered = sanitized.lower()

    return sanitized


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
    Filtrage de sortie anti-fuite (defense en profondeur)
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

        latency_seconds = time.perf_counter() - start_time

        latency_ms = latency_seconds * 1000

        try:
            alert_manager.evaluate_latency(latency_ms)
        except Exception:
            pass

        justification_raw = triage_result.get(
            "justification",
            "",
        )

        recommendations_raw = triage_result.get(
            "recommendations",
            [],
        )

        return TriageResponse(
            priority_level=triage_result.get(
                "priority_level",
                "UNKNOWN",
            ),
            justification=_strip_confidential_leakage(justification_raw),
            recommendations=[
                _strip_confidential_leakage(item) for item in recommendations_raw
            ],
            confidence_score=triage_result.get(
                "confidence_score",
                0.0,
            ),
            generated_at=triage_result.get(
                "generated_at",
                time.strftime("%Y-%m-%dT%H:%M:%S"),
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
            detail=("Triage engine failed: " f"{str(exc)}"),
        )
