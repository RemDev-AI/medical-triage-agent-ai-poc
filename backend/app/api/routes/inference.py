# medical-triage-agent-ai-poc/backend/app/api/routes/inference.py

from __future__ import annotations

import time
from typing import Optional, Tuple

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase

from app.api.schemas import (
    GenerateRequest,
    GenerateResponse,
)

from app.api.dependencies.inference import (
    get_generation_context,
)

from app.llm.inference.generate import generate_response

from app.monitoring.alerting import (
    alert_manager,
)

router = APIRouter(
    prefix="/generate",
    tags=["Inference"],
)


# Nom de modèle exposé dans GenerateResponse.model_name. Auparavant
# renvoyé par le backend HTTP distant (InferenceClient) ; en
# inférence locale on le fixe explicitement.
GENERATE_MODEL_NAME = "Qwen3-Medical-Triage"

# GenerateRequest n'expose qu'un "prompt" libre (endpoint générique,
# non spécialisé triage), alors que generate_response() attend un
# system_prompt et un user_prompt distincts. Hypothèse de migration :
# system prompt neutre par défaut. A ajuster si un prompt système
# spécifique est attendu pour cet endpoint générique.
DEFAULT_SYSTEM_PROMPT = "You are a helpful, safe assistant."


@router.post(
    "",
    response_model=GenerateResponse,
)
async def generate_route(
    payload: GenerateRequest,
    generation_context: Tuple[
        Optional[PreTrainedModel],
        Optional[PreTrainedTokenizerBase],
    ] = Depends(
        get_generation_context,
    ),
):
    model, tokenizer = generation_context

    start_time = time.perf_counter()

    try:

        generated_text = await generate_response(
            model=model,
            tokenizer=tokenizer,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            user_prompt=payload.prompt,
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

        return GenerateResponse(
            generated_text=generated_text,
            model_name=GENERATE_MODEL_NAME,
            latency_seconds=round(
                latency_seconds,
                3,
            ),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    except Exception as exc:

        try:
            alert_manager.raise_alert(
                code="INFERENCE_ERROR",
                message=str(exc),
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail=("Inference generation failed: " f"{str(exc)}"),
        )
