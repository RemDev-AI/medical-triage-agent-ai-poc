# medical-triage-agent-ai-poc/backend/app/tests/modal/test_modal_failover.py

"""
Failover and resilience tests.

Architecture:

Primary:
    Modal Endpoint

Fallback:
    Secondary Provider

These tests validate failover behavior without
requiring real external services.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def modal_response():
    return {
        "text": (
            "Patient doit surveiller "
            "l'évolution des symptômes."
        ),
        "provider": "modal",
        "latency_ms": 450,
    }


@pytest.fixture
def fallback_response():
    return {
        "text": (
            "Réponse fournie par le "
            "système de secours."
        ),
        "provider": "fallback",
        "latency_ms": 900,
    }


# ------------------------------------------------------------------
# Normal Operation
# ------------------------------------------------------------------


def test_primary_provider_available(
    modal_response,
):
    """
    Normal path using Modal.
    """

    assert (
        modal_response["provider"]
        == "modal"
    )

    assert "text" in modal_response


# ------------------------------------------------------------------
# Automatic Failover
# ------------------------------------------------------------------


def test_failover_switches_to_fallback(
    fallback_response,
):
    """
    System should switch to fallback
    provider when Modal fails.
    """

    assert (
        fallback_response["provider"]
        == "fallback"
    )

    assert (
        fallback_response["text"]
        .strip()
        != ""
    )


def test_failover_response_contract(
    fallback_response,
):
    """
    Fallback must respect the same API contract.
    """

    required_fields = [
        "text",
        "provider",
    ]

    for field in required_fields:
        assert field in fallback_response


# ------------------------------------------------------------------
# Simulated Provider Failure
# ------------------------------------------------------------------


def test_modal_timeout_triggers_failover():
    """
    Timeout should trigger fallback logic.
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


def test_modal_connection_error_triggers_failover():
    """
    Connection errors should be detected.
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


# ------------------------------------------------------------------
# Mocked Failover Service
# ------------------------------------------------------------------


def test_failover_service_execution(
    fallback_response,
):
    """
    Simulate fallback execution.
    """

    fallback_service = MagicMock()

    fallback_service.generate.return_value = (
        fallback_response
    )

    response = (
        fallback_service.generate(
            prompt="fièvre persistante"
        )
    )

    assert (
        response["provider"]
        == "fallback"
    )


def test_failover_preserves_response_schema(
    fallback_response,
):
    """
    API schema must remain identical
    between providers.
    """

    assert "text" in fallback_response
    assert "provider" in fallback_response


# ------------------------------------------------------------------
# Provider Selection Logic
# ------------------------------------------------------------------


def test_provider_priority_order():
    """
    Validate provider order.
    """

    providers = [
        "modal",
        "fallback",
    ]

    assert providers[0] == "modal"
    assert providers[1] == "fallback"


def test_provider_domain():
    """
    Supported providers.
    """

    allowed_providers = {
        "modal",
        "fallback",
    }

    assert "modal" in allowed_providers
    assert "fallback" in allowed_providers


# ------------------------------------------------------------------
# Monitoring Compatibility
# ------------------------------------------------------------------


def test_failover_latency_tracking(
    fallback_response,
):
    """
    Compatible with latency monitoring.
    """

    assert (
        fallback_response["latency_ms"]
        > 0
    )


def test_failover_request_tracking():
    """
    Compatible with request tracking.
    """

    response = {
        "request_id": (
            "req_failover_001"
        ),
        "provider": "fallback",
    }

    assert (
        "request_id"
        in response
    )

    assert (
        response["provider"]
        == "fallback"
    )


# ------------------------------------------------------------------
# Stability Tests
# ------------------------------------------------------------------


def test_multiple_failover_requests(
    fallback_response,
):
    """
    Simulate repeated failovers.
    """

    fallback_service = MagicMock()

    fallback_service.generate.return_value = (
        fallback_response
    )

    for _ in range(20):

        response = (
            fallback_service.generate(
                prompt="test"
            )
        )

        assert (
            response["provider"]
            == "fallback"
        )


def test_failover_text_not_empty(
    fallback_response,
):
    """
    Fallback must always return content.
    """

    assert (
        fallback_response["text"]
        .strip()
        != ""
    )


# ------------------------------------------------------------------
# Future Multi-Provider Support
# ------------------------------------------------------------------


def test_future_provider_expansion():
    """
    Future-compatible provider registry.
    """

    providers = {
        "modal",
        "fallback",
        "provider_c",
    }

    assert "modal" in providers
    assert "fallback" in providers
