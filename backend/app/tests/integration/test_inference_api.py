# medical-triage-agent-ai-poc/backend/app/tests/integration/test_inference_api.py

"""
Integration tests for inference endpoints.

Architecture:

HF Space UI
      ↓
HF Space API
      ↓
Modal Endpoint
      ↓
Qwen3 + LoRA

No local model loading is performed.
"""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_generate_endpoint():
    """
    Validate /generate endpoint returns valid text.
    """

    payload = {
        "prompt": (
            "Patient présente une fièvre persistante "
            "accompagnée de toux depuis 5 jours."
        ),
        "max_tokens": 100,
    }

    response = client.post(
        "/generate",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "text" in data
    assert isinstance(data["text"], str)
    assert len(data["text"]) > 0


def test_generate_endpoint_contract():
    """
    Contract validation for generate endpoint.
    """

    payload = {
        "prompt": "Douleur thoracique modérée.",
        "max_tokens": 50,
    }

    response = client.post(
        "/generate",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    required_fields = [
        "text",
    ]

    for field in required_fields:
        assert field in body


def test_generate_endpoint_modal_metadata():
    """
    Optional Modal metadata validation.

    Enables observability and future failover tracking.
    """

    payload = {
        "prompt": "Patient souffre de céphalées.",
        "max_tokens": 50,
    }

    response = client.post(
        "/generate",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    if "provider" in data:
        assert data["provider"] in [
            "modal",
            "fallback",
        ]

    if "latency_ms" in data:
        assert isinstance(
            data["latency_ms"],
            (int, float),
        )


def test_triage_endpoint_inference():
    """
    Validate triage endpoint returns clinical response.
    """

    payload = {
        "symptoms": "fatigue, vertiges",
        "age": 28,
    }

    response = client.post(
        "/triage",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "priority" in data
    assert "recommendation" in data

    assert isinstance(data["priority"], str)
    assert isinstance(data["recommendation"], str)


def test_triage_priority_values():
    """
    Validate allowed triage levels.
    """

    payload = {
        "symptoms": "douleur thoracique",
        "age": 62,
    }

    response = client.post(
        "/triage",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    allowed_priorities = {
        "low",
        "medium",
        "high",
        "urgent",
    }

    assert data["priority"] in allowed_priorities


def test_inference_handles_long_prompt():
    """
    Long prompts should not crash the API.
    """

    payload = {
        "prompt": "fièvre " * 500,
        "max_tokens": 100,
    }

    response = client.post(
        "/generate",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "text" in data


def test_inference_response_is_not_empty():
    """
    Generated text must not be empty.
    """

    payload = {
        "prompt": "Quels symptômes nécessitent une urgence médicale ?",
        "max_tokens": 80,
    }

    response = client.post(
        "/generate",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["text"].strip() != ""
