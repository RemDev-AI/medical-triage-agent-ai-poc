# medical-triage-agent-ai-poc/backend/app/tests/integration/test_frontend_backend.py

import pytest  # noqa : F401
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_frontend_health_check():
    """
    Verify frontend can connect to backend health endpoint
    """

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_frontend_triage_submission():
    """
    Simulate frontend triage form submission
    """

    payload = {
        "symptoms": "maux de gorge, éternuements",
        "age": 22
    }

    response = client.post("/triage", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "priority" in data
    assert "recommendation" in data
