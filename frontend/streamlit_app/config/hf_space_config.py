# medical-triage-agent-ai-poc/frontend/streamlit_app/config/hf_space_config.py

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class HFSpaceUIConfig:

    api_base_url: str

    environment: str

    monitoring_enabled: bool

    request_timeout: int


def get_ui_config() -> HFSpaceUIConfig:

    return HFSpaceUIConfig(
        api_base_url=os.getenv(
            "API_BASE_URL",
            "http://localhost:8000",
        ),
        environment=os.getenv(
            "ENVIRONMENT",
            "development",
        ),
        monitoring_enabled=os.getenv(
            "ENABLE_MONITORING",
            "true",
        ).lower()
        == "true",
        request_timeout=int(
            os.getenv(
                "REQUEST_TIMEOUT",
                "120",
            )
        ),
    )


ui_config = get_ui_config()
