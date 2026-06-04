# medical-triage-agent-ai-poc/backend/app/llm/modal/modal_exceptions.py

"""
Custom exceptions for Modal inference layer.

This module centralizes all Modal-related exceptions used by:
- modal_client.py
- modal_inference.py
- generate.py
- triage_engine.py

The objective is to provide:
- clear error categorization;
- consistent error handling;
- easier monitoring and observability;
- cleaner API responses.
"""

from __future__ import annotations


class ModalInferenceError(Exception):
    """
    Base exception for all Modal inference errors.
    """

    pass


class ModalConfigurationError(ModalInferenceError):
    """
    Raised when Modal configuration is invalid.

    Examples:
    - Missing MODAL_ENDPOINT_URL
    - Missing MODAL_API_TOKEN
    - Invalid environment configuration
    """

    pass


class ModalAuthenticationError(ModalInferenceError):
    """
    Raised when Modal authentication fails.

    Examples:
    - Invalid API token
    - Unauthorized request (401)
    - Forbidden request (403)
    """

    pass


class ModalConnectionError(ModalInferenceError):
    """
    Raised when Modal endpoint is unreachable.

    Examples:
    - DNS failure
    - Network outage
    - Connection refused
    """

    pass


class ModalTimeoutError(ModalInferenceError):
    """
    Raised when Modal inference exceeds timeout limits.

    Examples:
    - Request timeout
    - Read timeout
    - Long-running inference
    """

    pass


class ModalValidationError(ModalInferenceError):
    """
    Raised when request payload validation fails.

    Examples:
    - Empty prompt
    - Invalid generation parameters
    - Missing required fields
    """

    pass


class ModalInvalidResponseError(ModalInferenceError):
    """
    Raised when Modal returns an invalid response payload.

    Examples:
    - Missing generated_text field
    - Malformed JSON
    - Unexpected response schema
    """

    pass


class ModalRateLimitError(ModalInferenceError):
    """
    Raised when Modal API rate limiting occurs.

    Examples:
    - HTTP 429
    - Quota exceeded
    - Too many requests
    """

    pass


class ModalServerError(ModalInferenceError):
    """
    Raised when Modal endpoint returns server-side errors.

    Examples:
    - HTTP 500
    - HTTP 502
    - HTTP 503
    - HTTP 504
    """

    pass
