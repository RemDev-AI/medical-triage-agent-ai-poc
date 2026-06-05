# medical-triage-agent-ai-poc/backend/app/api/routes/inference.py

import time

from fastapi import APIRouter, HTTPException

from api.schemas import (
    GenerateRequest,
    GenerateResponse
)

from app.llm.inference.generate import generate_text


router = APIRouter(
    prefix="/generate",
    tags=["Inference"]
)


@router.post(
    "/",
    response_model=GenerateResponse
)
async def generate_route(payload: GenerateRequest):

    start_time = time.perf_counter()

    try:
        generated_text = generate_text(
            prompt=payload.prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p
        )

        latency = round(
            time.perf_counter() - start_time,
            3
        )

        return GenerateResponse(
            generated_text=generated_text,
            model_name="Qwen3-Medical-Triage",
            latency_seconds=latency,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference generation failed: {str(exc)}"
        )
