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

from app.main import app

client = TestClient(app)


def test_generate_endpoint(auth_headers):
    """
    Validate /generate endpoint returns valid text.
    """

    payload = {
        "prompt": (
            "Patient présente une fièvre persistante "
            "accompagnée de toux depuis 5 jours."
        ),
        "max_new_tokens": 100,
    }

    response = client.post(
        "/generate",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert "generated_text" in data
    assert isinstance(data["generated_text"], str)
    assert len(data["generated_text"]) > 0


def test_generate_endpoint_contract(auth_headers):
    """
    Contract validation for generate endpoint.
    """

    payload = {
        "prompt": "Douleur thoracique modérée.",
        "max_new_tokens": 50,
    }

    response = client.post(
        "/generate",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == 200

    body = response.json()

    required_fields = [
        "generated_text",
        "model_name",
        "latency_seconds",
        "timestamp",
    ]

    for field in required_fields:
        assert field in body


def test_generate_endpoint_modal_metadata(auth_headers):
    """
    Optional Modal metadata validation.

    Enables observability and future failover tracking.
    """

    payload = {
        "prompt": "Patient souffre de céphalées.",
        "max_new_tokens": 50,
    }

    response = client.post(
        "/generate",
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

    assert isinstance(
        data["latency_seconds"],
        (int, float),
    )


def test_triage_endpoint_inference(auth_headers):
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
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert "priority_level" in data
    assert "recommendations" in data

    assert isinstance(data["priority_level"], str)
    assert isinstance(data["recommendations"], list)


def test_triage_priority_values(auth_headers):
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
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    allowed_priorities = {
        "low",
        "medium",
        "high",
        "urgent",
    }

    assert data["priority_level"] in allowed_priorities


def test_inference_handles_long_prompt(auth_headers):
    """
    Long prompts should not crash the API.
    """

    payload = {
        "prompt": "fièvre " * 500,
        "max_new_tokens": 100,
    }

    response = client.post(
        "/generate",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert "generated_text" in data


def test_inference_response_is_not_empty(auth_headers):
    """
    Generated text must not be empty.
    """

    payload = {
        "prompt": "Quels symptômes nécessitent une urgence médicale ?",
        "max_new_tokens": 80,
    }

    response = client.post(
        "/generate",
        json=payload,
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["generated_text"].strip() != ""
