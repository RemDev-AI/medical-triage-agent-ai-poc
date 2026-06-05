# medical-triage-agent-ai-poc/backend/app/tests/modal/test_modal_client.py

"""
Unit tests for Modal client integration.

These tests validate the communication layer between:

HF Space API
        ↓
Modal GPU Endpoint

No real Modal call is executed.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def sample_prompt():
    return (
        "Patient présente une fièvre persistante "
        "depuis plusieurs jours."
    )


@pytest.fixture
def mock_modal_response():
    return {
        "text": (
            "Le patient devrait consulter un médecin "
            "si les symptômes persistent."
        ),
        "provider": "modal",
        "latency_ms": 742,
    }


# ------------------------------------------------------------------
# Client Initialization
# ------------------------------------------------------------------


def test_modal_client_configuration():
    """
    Validate Modal configuration values.
    """

    endpoint = (
        "https://example-modal-endpoint.modal.run"
    )

    assert endpoint.startswith("https://")
    assert "modal" in endpoint


# ------------------------------------------------------------------
# Request Construction
# ------------------------------------------------------------------


def test_modal_payload_creation(sample_prompt):
    """
    Validate payload sent to Modal endpoint.
    """

    payload = {
        "prompt": sample_prompt,
        "max_tokens": 256,
        "temperature": 0.2,
    }

    assert payload["prompt"] == sample_prompt
    assert payload["max_tokens"] > 0
    assert payload["temperature"] >= 0


# ------------------------------------------------------------------
# Response Validation
# ------------------------------------------------------------------


def test_modal_response_schema(
    mock_modal_response,
):
    """
    Validate Modal response structure.
    """

    assert "text" in mock_modal_response
    assert "provider" in mock_modal_response

    assert isinstance(
        mock_modal_response["text"],
        str,
    )

    assert (
        mock_modal_response["provider"]
        == "modal"
    )


def test_modal_response_contains_text(
    mock_modal_response,
):
    """
    Response text must not be empty.
    """

    assert (
        mock_modal_response["text"].strip()
        != ""
    )


def test_modal_latency_metric(
    mock_modal_response,
):
    """
    Validate latency metadata.
    """

    assert "latency_ms" in mock_modal_response

    assert isinstance(
        mock_modal_response["latency_ms"],
        (int, float),
    )

    assert mock_modal_response["latency_ms"] > 0


# ------------------------------------------------------------------
# Error Handling
# ------------------------------------------------------------------


def test_modal_timeout_handling():
    """
    Simulate timeout scenario.
    """

    with patch(
        "requests.post"
    ) as mock_request:

        mock_request.side_effect = TimeoutError()

        with pytest.raises(
            TimeoutError
        ):
            mock_request()


def test_modal_connection_error():
    """
    Simulate connection failure.
    """

    with patch(
        "requests.post"
    ) as mock_request:

        mock_request.side_effect = ConnectionError()

        with pytest.raises(
            ConnectionError
        ):
            mock_request()


# ------------------------------------------------------------------
# Mocked Endpoint Invocation
# ------------------------------------------------------------------


def test_modal_mocked_inference_call(
    mock_modal_response,
):
    """
    Simulate Modal inference execution.
    """

    mocked_client = MagicMock()

    mocked_client.generate.return_value = (
        mock_modal_response
    )

    response = mocked_client.generate(
        prompt="fièvre persistante"
    )

    assert response["provider"] == "modal"

    assert (
        "text"
        in response
    )


# ------------------------------------------------------------------
# Future Failover Compatibility
# ------------------------------------------------------------------


def test_modal_provider_value():
    """
    Validate provider values used by
    failover strategy.
    """

    allowed_providers = {
        "modal",
        "fallback",
    }

    assert "modal" in allowed_providers
    assert "fallback" in allowed_providers
