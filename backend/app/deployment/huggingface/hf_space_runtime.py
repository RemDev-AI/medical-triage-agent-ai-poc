# medical-triage-agent-ai-poc/backend/app/deployment/huggingface/hf_space_runtime.py

"""
Configuration runtime Hugging Face Spaces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class HFSpaceRuntimeConfig:

    hf_space: bool

    model_repository: str

    device: str

    use_vllm: bool

    load_in_4bit: bool

    load_in_8bit: bool

    monitoring_enabled: bool

    request_tracking_enabled: bool

    max_input_tokens: int

    max_output_tokens: int


def get_runtime_config() -> HFSpaceRuntimeConfig:

    return HFSpaceRuntimeConfig(
        hf_space=os.getenv(
            "HF_SPACE",
            "false",
        ).lower()
        == "true",
        model_repository=os.getenv(
            "HF_MODEL_REPOSITORY",
            "medical-triage-agent-ai-poc-models",
        ),
        device=os.getenv(
            "DEVICE",
            "auto",
        ),
        use_vllm=os.getenv(
            "USE_VLLM",
            "true",
        ).lower()
        == "true",
        load_in_4bit=os.getenv(
            "LOAD_IN_4BIT",
            "false",
        ).lower()
        == "true",
        load_in_8bit=os.getenv(
            "LOAD_IN_8BIT",
            "true",
        ).lower()
        == "true",
        monitoring_enabled=os.getenv(
            "ENABLE_MONITORING",
            "true",
        ).lower()
        == "true",
        request_tracking_enabled=os.getenv(
            "ENABLE_REQUEST_TRACKING",
            "true",
        ).lower()
        == "true",
        max_input_tokens=int(
            os.getenv(
                "MAX_INPUT_TOKENS",
                "2048",
            )
        ),
        max_output_tokens=int(
            os.getenv(
                "MAX_OUTPUT_TOKENS",
                "512",
            )
        ),
    )


runtime_config = get_runtime_config()
