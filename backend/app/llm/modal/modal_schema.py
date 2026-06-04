# medical-triage-agent-ai-poc/backend/app/llm/modal/modal_schema.py

"""
Pydantic schemas for Modal inference requests and responses.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class ModalGenerationParameters(BaseModel):
    """
    Generation parameters forwarded to Modal.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    max_new_tokens: int = Field(
        default=256,
        ge=1,
        le=2048,
        description="Maximum number of generated tokens.",
    )

    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )

    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter.",
    )

    repetition_penalty: float = Field(
        default=1.1,
        ge=0.5,
        le=3.0,
        description="Penalty applied to repeated tokens.",
    )


class ModalInferenceRequest(BaseModel):
    """
    Request payload sent to Modal endpoint.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    system_prompt: str = Field(
        ...,
        min_length=1,
        description="System instruction prompt.",
    )

    user_prompt: str = Field(
        ...,
        min_length=1,
        description="User prompt content.",
    )

    generation: ModalGenerationParameters = Field(
        default_factory=ModalGenerationParameters,
    )


class ModalUsage(BaseModel):
    """
    Token usage metadata returned by Modal.
    """

    model_config = ConfigDict(
        extra="ignore",
    )

    prompt_tokens: int = Field(
        default=0,
        ge=0,
    )

    completion_tokens: int = Field(
        default=0,
        ge=0,
    )

    total_tokens: int = Field(
        default=0,
        ge=0,
    )


class ModalInferenceResponse(BaseModel):
    """
    Successful response returned by Modal.
    """

    model_config = ConfigDict(
        extra="ignore",
    )

    generated_text: str = Field(
        ...,
        min_length=1,
        description="Generated clinical response.",
    )

    model_name: str = Field(
        ...,
        min_length=1,
        description="Inference model name.",
    )

    latency_ms: float = Field(
        ...,
        ge=0,
        description="Modal endpoint latency.",
    )

    usage: ModalUsage = Field(
        default_factory=ModalUsage,
    )


class ModalHealthResponse(BaseModel):
    """
    Optional healthcheck schema for Modal endpoint.
    """

    model_config = ConfigDict(
        extra="ignore",
    )

    status: str = Field(
        ...,
        description="Endpoint status.",
    )

    model_name: str = Field(
        ...,
        min_length=1,
    )

    version: str = Field(
        ...,
        min_length=1,
    )


class ModalErrorResponse(BaseModel):
    """
    Error payload returned by Modal.
    """

    model_config = ConfigDict(
        extra="allow",
    )

    error: str = Field(
        ...,
        min_length=1,
    )

    detail: str | None = Field(
        default=None,
    )

    code: str | None = Field(
        default=None,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )
