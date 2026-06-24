# medical-triage-agent-ai-poc/backend/app/deployment/modal/modal_endpoint.py

"""
modal_endpoint.py

Endpoints Modal GPU pour :

- génération médicale
- triage médical
- monitoring santé

Expose :

GET  /health
POST /generate
POST /triage

Compatible :

- Modal
- FastAPI
- vLLM
- Qwen3
- Hugging Face Hub
"""

from __future__ import annotations

import os
import time
from typing import Any

import modal
from fastapi import FastAPI
from pydantic import BaseModel, Field
from vllm import LLM
from vllm import SamplingParams

from backend.app.deployment.modal.modal_gpu_config import get_inference_gpu
from backend.app.deployment.modal.modal_image import get_modal_image

# =========================================================
# CONFIGURATION
# =========================================================

APP_NAME = "medical-triage-agent-ai-poc"

MODEL_REPOSITORY = os.getenv(
    "HF_MODEL_REPOSITORY",
    "medical-triage-agent-ai-poc-models",
)

BASE_MODEL = os.getenv(
    "HF_BASE_MODEL",
    "Qwen/Qwen3-1.7B-Base",
)

LORA_PATH = os.getenv(
    "HF_LORA_PATH",
    "lora-adapters",
)

MAX_MODEL_LEN = int(
    os.getenv(
        "MAX_MODEL_LEN",
        "4096",
    )
)

TEMPERATURE = float(
    os.getenv(
        "TEMPERATURE",
        "0.2",
    )
)

MAX_TOKENS = int(
    os.getenv(
        "MAX_TOKENS",
        "512",
    )
)

# =========================================================
# MODAL APP
# =========================================================

app = modal.App(APP_NAME)

# =========================================================
# FASTAPI
# =========================================================

web_app = FastAPI(
    title="Medical Triage Agent",
    version="1.0.0",
)

# =========================================================
# SCHEMAS
# =========================================================


class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
    )

    max_tokens: int = Field(
        default=MAX_TOKENS,
        ge=32,
        le=2048,
    )

    temperature: float = Field(
        default=TEMPERATURE,
        ge=0.0,
        le=2.0,
    )


class GenerateResponse(BaseModel):
    generated_text: str
    latency_seconds: float


class TriageRequest(BaseModel):
    symptoms: str
    medical_history: str | None = None


class TriageResponse(BaseModel):
    triage_result: str
    latency_seconds: float


# =========================================================
# MODEL SERVICE
# =========================================================

@app.cls(
    image=get_modal_image(),
    gpu="A10G",
    scaledown_window=300,
)
class MedicalInferenceService:

    @modal.enter()
    def load_model(self) -> None:

        self.start_time = time.time()

        self.llm = LLM(
            model=BASE_MODEL,
            enable_lora=True,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
        )

    def generate_text(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:

        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        outputs = self.llm.generate(
            [prompt],
            params,
        )

        return outputs[0].outputs[0].text

    def generate_triage(
        self,
        symptoms: str,
        medical_history: str | None,
    ) -> str:

        prompt = f"""
Tu es un assistant spécialisé en triage médical.

Symptômes :
{symptoms}

Antécédents :
{medical_history or "Aucun"}

Réponds avec :

1. Niveau d'urgence
2. Justification
3. Recommandation

Important :
Tu ne remplaces jamais un professionnel de santé.
"""

        return self.generate_text(
            prompt=prompt,
            temperature=0.2,
            max_tokens=512,
        )


# =========================================================
# HEALTH
# =========================================================

@web_app.get("/health")
def health() -> dict[str, Any]:

    gpu_config = get_inference_gpu()

    return {
        "status": "healthy",
        "service": APP_NAME,
        "gpu": gpu_config.gpu_name,
        "gpu_memory_gb": gpu_config.memory_gb,
        "model": BASE_MODEL,
        "timestamp": time.time(),
    }


# =========================================================
# GENERATE
# =========================================================

@web_app.post(
    "/generate",
    response_model=GenerateResponse,
)
def generate(
    request: GenerateRequest,
) -> GenerateResponse:

    started = time.time()

    service = MedicalInferenceService()

    generated = service.generate_text.remote(
        prompt=request.prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    latency = round(
        time.time() - started,
        3,
    )

    return GenerateResponse(
        generated_text=generated,
        latency_seconds=latency,
    )


# =========================================================
# TRIAGE
# =========================================================

@web_app.post(
    "/triage",
    response_model=TriageResponse,
)
def triage(
    request: TriageRequest,
) -> TriageResponse:

    started = time.time()

    service = MedicalInferenceService()

    result = service.generate_triage.remote(
        symptoms=request.symptoms,
        medical_history=request.medical_history,
    )

    latency = round(
        time.time() - started,
        3,
    )

    return TriageResponse(
        triage_result=result,
        latency_seconds=latency,
    )


# =========================================================
# ASGI ENTRYPOINT
# =========================================================

@app.function(
    image=get_modal_image(),
)
@modal.asgi_app()
def fastapi_app():

    return web_app
