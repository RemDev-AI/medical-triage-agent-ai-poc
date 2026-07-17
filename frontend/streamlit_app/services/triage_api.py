# medical-triage-agent-ai-poc/frontend/streamlit_app/services/triage_api.py

import requests

from streamlit_app.config.settings import API_BASE_URL
from streamlit_app.services.auth_client import get_auth_headers


def submit_triage_request(payload: dict) -> dict:

    try:
        response = requests.post(
            f"{API_BASE_URL}/triage",
            json=payload,
            headers=get_auth_headers(),
            timeout=60,
        )

        if response.status_code == 200:
            return response.json()

        if response.status_code == 401:
            return {
                "priority": "UNKNOWN",
                "justification": "Session expirée ou jeton invalide.",
                "recommendations": "Rechargez la page pour renouveler l'authentification.",
            }

        return {
            "priority": "UNKNOWN",
            "justification": f"Erreur API {response.status_code}",
            "recommendations": "Aucune recommandation disponible.",
        }

    except requests.RequestException as exc:
        return {
            "priority": "UNKNOWN",
            "justification": str(exc),
            "recommendations": "Backend inaccessible.",
        }
