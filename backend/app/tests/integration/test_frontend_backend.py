# medical-triage-agent-ai-poc/backend/app/tests/integration/test_frontend_backend.py

"""
Integration tests validating communication between:

Frontend (HF Space UI)
        ↓
Backend API (HF Space API)
        ↓
Modal GPU Inference Endpoint

These tests do not load any local model.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_frontend_health_check():
    """
    Frontend must be able to reach backend health endpoint.
    """

    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"


def test_frontend_triage_submission(auth_headers):
    """
    Simulates a triage request coming from Streamlit UI.
    """

    payload = {
        "symptoms": "maux de gorge, éternuements",
        "age": 22,
    }

    response = client.post(
        "/triage",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert "priority_level" in data
    assert "recommendations" in data

    assert isinstance(data["priority_level"], str)
    assert isinstance(data["recommendations"], list)


def test_frontend_receives_modal_metadata(auth_headers):
    """
    Verify backend returns inference metadata.

    Useful for observability and Modal monitoring.
    """

    payload = {
        "symptoms": "fièvre légère",
        "age": 35,
    }

    response = client.post(
        "/triage",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    if "provider" in data:
        assert data["provider"] in [
            "modal",
            "fallback",
        ]


def test_frontend_response_contract(auth_headers):
    """
    Contract test between frontend and backend.

    Prevents accidental API breaking changes.
    """

    payload = {
        "symptoms": "fatigue",
        "age": 40,
    }

    response = client.post(
        "/triage",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == 200

    body = response.json()

    required_fields = [
        "priority_level",
        "recommendations",
    ]

    for field in required_fields:
        assert field in body
