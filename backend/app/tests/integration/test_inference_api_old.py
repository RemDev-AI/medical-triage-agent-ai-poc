# medical-triage-agent-ai-poc/backend/app/tests/integration/test_inference_api.py

import pytest  # noqa : F401
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_generate_endpoint():
    """
    /generate should return valid model output
    """

    payload = {
        "prompt": "Patient présente une fièvre persistante et toux",
        "max_tokens": 50
    }

    response = client.post("/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert isinstance(data["text"], str)


def test_triage_endpoint_inference():
    """
    /triage endpoint should provide inference results
    """

    payload = {
        "symptoms": "fatigue, vertiges",
        "age": 28
    }

    response = client.post("/triage", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "priority" in data
    assert "recommendation" in data
