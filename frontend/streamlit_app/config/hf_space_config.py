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

    # ---------------------------------------------------
    # Authentification JWT (POST /auth/token)
    #
    # api_access_key : secret partagé, distinct de SECRET_KEY
    # côté backend, requis pour obtenir un JWT. Doit être défini
    # dans les Secrets du Space UI, avec la MÊME valeur que dans
    # les Secrets du Space API (cf. backend/app/core/config.py).
    #
    # client_id : identifiant logique de ce frontend auprès de
    # l'API (claim "sub" du JWT, utile pour l'audit logging).
    # ---------------------------------------------------
    api_access_key: str

    client_id: str


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
        api_access_key=os.getenv(
            "API_ACCESS_KEY",
            "",
        ),
        client_id=os.getenv(
            "STREAMLIT_CLIENT_ID",
            "streamlit-ui",
        ),
    )


ui_config = get_ui_config()
