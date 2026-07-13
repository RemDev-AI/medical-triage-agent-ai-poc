# medical-triage-agent-ai-poc/frontend/streamlit_app/services/api_client.py

import requests

from streamlit_app.config.settings import (
    API_BASE_URL,
    REQUEST_TIMEOUT,
)


def check_api_health() -> dict:
    """
    Check FastAPI backend health endpoint.
    """

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 200:
            return {
                "status": "healthy",
                "message": "API FastAPI opérationnelle",
            }

        return {
            "status": "unhealthy",
            "message": f"Erreur HTTP {response.status_code}",
        }

    except requests.RequestException as exc:
        return {
            "status": "unhealthy",
            "message": str(exc),
        }
