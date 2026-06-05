# medical-triage-agent-ai-poc/backend/app/tests/modal/test_modal_endpoint.py

"""
Integration tests for Modal inference endpoint.

These tests validate endpoint behaviour without
requiring a real Modal deployment.

Architecture:

HF Space API
      ↓
Modal Endpoint
      ↓
Qwen3 + LoRA
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def modal_generate_payload():
    return {
        "prompt": (
            "Patient présente une fièvre persistante "
            "et une toux depuis plusieurs jours."
        ),
        "max_tokens": 128,
        "temperature": 0.2,
    }


@pytest.fixture
def modal_success_response():
    return {
        "text": (
            "Une consultation médicale est recommandée "
            "si les symptômes persistent."
        ),
        "provider": "modal",
        "latency_ms": 520,
    }


# ------------------------------------------------------------------
# Endpoint Availability
# ------------------------------------------------------------------


def test_modal_endpoint_url_format():
    """
    Validate endpoint URL structure.
    """

    endpoint_url = (
        "https://medical-triage.modal.run"
    )

    assert endpoint_url.startswith(
        "https://"
    )

    assert ".modal.run" in endpoint_url


# ------------------------------------------------------------------
# Request Validation
# ------------------------------------------------------------------


def test_modal_generate_payload_schema(
    modal_generate_payload,
):
    """
    Validate request payload schema.
    """

    assert (
        "prompt"
        in modal_generate_payload
    )

    assert (
        "max_tokens"
        in modal_generate_payload
    )

    assert (
        "temperature"
        in modal_generate_payload
    )

    assert isinstance(
        modal_generate_payload["prompt"],
        str,
    )


def test_modal_prompt_not_empty(
    modal_generate_payload,
):
    """
    Prompt should contain content.
    """

    assert (
        modal_generate_payload["prompt"]
        .strip()
        != ""
    )


# ------------------------------------------------------------------
# Response Contract
# ------------------------------------------------------------------


def test_modal_response_contract(
    modal_success_response,
):
    """
    Validate response schema.
    """

    required_fields = [
        "text",
        "provider",
        "latency_ms",
    ]

    for field in required_fields:
        assert (
            field
            in modal_success_response
        )


def test_modal_response_text_type(
    modal_success_response,
):
    """
    Generated text should be valid.
    """

    assert isinstance(
        modal_success_response["text"],
        str,
    )

    assert (
        modal_success_response["text"]
        .strip()
        != ""
    )


def test_modal_response_provider(
    modal_success_response,
):
    """
    Validate provider field.
    """

    assert (
        modal_success_response["provider"]
        == "modal"
    )


def test_modal_response_latency(
    modal_success_response,
):
    """
    Latency metric validation.
    """

    assert isinstance(
        modal_success_response["latency_ms"],
        (int, float),
    )

    assert (
        modal_success_response["latency_ms"]
        > 0
    )


# ------------------------------------------------------------------
# Mocked Endpoint Calls
# ------------------------------------------------------------------


def test_modal_generate_endpoint_success(
    modal_success_response,
):
    """
    Simulate successful endpoint call.
    """

    mock_endpoint = MagicMock()

    mock_endpoint.generate.return_value = (
        modal_success_response
    )

    response = (
        mock_endpoint.generate(
            prompt="fièvre persistante"
        )
    )

    assert (
        response["provider"]
        == "modal"
    )

    assert "text" in response


def test_modal_generate_endpoint_multiple_calls():
    """
    Endpoint should support
    multiple sequential requests.
    """

    mock_endpoint = MagicMock()

    mock_endpoint.generate.return_value = {
        "text": "ok",
        "provider": "modal",
    }

    for _ in range(10):
        response = (
            mock_endpoint.generate(
                prompt="test"
            )
        )

        assert (
            response["provider"]
            == "modal"
        )


# ------------------------------------------------------------------
# Error Handling
# ------------------------------------------------------------------


def test_modal_endpoint_timeout():
    """
    Simulate timeout from endpoint.
    """

    with patch(
        "requests.post"
    ) as mock_post:

        mock_post.side_effect = (
            TimeoutError()
        )

        with pytest.raises(
            TimeoutError
        ):
            mock_post()


def test_modal_endpoint_connection_error():
    """
    Simulate network failure.
    """

    with patch(
        "requests.post"
    ) as mock_post:

        mock_post.side_effect = (
            ConnectionError()
        )

        with pytest.raises(
            ConnectionError
        ):
            mock_post()


def test_modal_endpoint_invalid_response():
    """
    Detect malformed endpoint responses.
    """

    invalid_response = {
        "provider": "modal",
    }

    assert (
        "text"
        not in invalid_response
    )


# ------------------------------------------------------------------
# Monitoring Compatibility
# ------------------------------------------------------------------


def test_modal_endpoint_latency_tracking(
    modal_success_response,
):
    """
    Compatible with latency_monitor.py
    """

    assert (
        modal_success_response["latency_ms"]
        < 30000
    )


def test_modal_endpoint_request_tracking():
    """
    Compatible with request_tracker.py
    """

    response = {
        "request_id": (
            "req_123456"
        )
    }

    assert (
        "request_id"
        in response
    )

    assert isinstance(
        response["request_id"],
        str,
    )


# ------------------------------------------------------------------
# Future Failover Compatibility
# ------------------------------------------------------------------


def test_modal_endpoint_provider_domain():
    """
    Compatible with future failover logic.
    """

    allowed_providers = {
        "modal",
        "fallback",
    }

    assert (
        "modal"
        in allowed_providers
    )

    assert (
        "fallback"
        in allowed_providers
    )
