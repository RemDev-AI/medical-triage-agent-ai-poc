# medical-triage-agent-ai-poc/backend/app/api/routes/triage.py

from __future__ import annotations

import time

import httpx

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from app.api.schemas import (
    TriageRequest,
    TriageResponse,
)

from app.api.dependencies.inference import (
    InferenceClient,
    get_inference_client,
)

from app.monitoring.alerting import (
    alert_manager,
)

router = APIRouter(
    prefix="/triage",
    tags=["Triage"],
)


CONFIDENTIAL_MARKERS: tuple[str, ...] = (
    "confidential",
    "medical records",
    "medical record",
    "patient_id",
    "ssn",
    "social security",
)


def _strip_confidential_leakage(text: str) -> str:

    sanitized = text

    for marker in CONFIDENTIAL_MARKERS:
        if marker.lower() in sanitized.lower():
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
    start_time = time.perf_counter()

    try:

        triage_result = await inference_client.triage(
            symptoms=payload.symptoms,
            medical_history=payload.medical_history,
            age=payload.age,
            priority_context=payload.priority_context,
        )

    except httpx.HTTPStatusError as exc:

        try:
            alert_manager.raise_alert(
                code="INFERENCE_UPSTREAM_ERROR",
                message=f"Upstream status {exc.response.status_code}",
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=502,
            detail="Triage engine is currently unavailable. Please retry later.",
        )

    except httpx.RequestError:

        try:
            alert_manager.raise_alert(
                code="INFERENCE_NETWORK_ERROR",
                message="Network error while contacting inference backend.",
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=503,
            detail="Triage engine is currently unreachable. Please retry later.",
        )

    except Exception:

        try:
            alert_manager.raise_alert(
                code="TRIAGE_ERROR",
                message="Unexpected error during triage processing.",
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the triage request.",
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
