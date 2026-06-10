# medical-triage-agent-ai-poc/frontend/streamlit_app/services/metrics_api.py

from __future__ import annotations

import requests

from streamlit_app.config.settings import (
    API_BASE_URL,
    REQUEST_TIMEOUT,
)


MONITORING_ENDPOINT = (
    f"{API_BASE_URL}/monitoring/overview"
)


def get_metrics() -> dict:
    """
    Récupération des métriques
    d'observabilité du backend.
    """

    try:

        response = requests.get(
            MONITORING_ENDPOINT,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.json()

    except requests.Timeout:

        return {
            "latency": {},
            "requests": {},
            "gpu": {},
            "alerts": [
                "Monitoring API timeout"
            ],
        }

    except requests.RequestException as exc:

        return {
            "latency": {},
            "requests": {},
            "gpu": {},
            "alerts": [
                f"Monitoring API error: {exc}"
            ],
        }
