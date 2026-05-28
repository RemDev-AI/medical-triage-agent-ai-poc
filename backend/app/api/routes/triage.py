# medical-triage-agent-ai-poc/backend/app/api/routes/triage.py

import time

from fastapi import APIRouter, HTTPException

from api.schemas import (
    TriageRequest,
    TriageResponse
)

from app.llm.inference.triage_engine import (
    run_medical_triage
)


router = APIRouter(
    prefix="/triage",
    tags=["Triage"]
)


@router.post(
    "/",
    response_model=TriageResponse
)
async def triage_route(payload: TriageRequest):

    try:
        start_time = time.perf_counter()

        triage_result = run_medical_triage(
            symptoms=payload.symptoms,
            medical_history=payload.medical_history,
            age=payload.age,
            priority_context=payload.priority_context
        )

        latency = round(
            time.perf_counter() - start_time,
            3
        )

        return TriageResponse(
            priority_level=triage_result["priority_level"],
            justification=triage_result["justification"],
            recommendations=triage_result["recommendations"],
            confidence_score=triage_result["confidence_score"],
            generated_at=triage_result["generated_at"],
            latency_seconds=latency
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Triage engine failed: {str(exc)}"
        )
