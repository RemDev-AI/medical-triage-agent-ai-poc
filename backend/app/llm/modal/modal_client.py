# medical-triage-agent-ai-poc/backend/app/llm/modal/modal_client.py

"""
HTTP client for Modal inference endpoint.
"""

from __future__ import annotations

import logging

import httpx

from app.llm.modal.modal_exceptions import (
    ModalAuthenticationError,
    ModalConfigurationError,
    ModalConnectionError,
    ModalInvalidResponseError,
    ModalRateLimitError,
    ModalServerError,
    ModalTimeoutError,
)

from app.llm.modal.modal_schema import (
    ModalErrorResponse,
    ModalInferenceRequest,
    ModalInferenceResponse,
)

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT_SECONDS = 120.0


class ModalClient:
    """
    HTTP client used to communicate with Modal GPU endpoint.
    """

    def __init__(
        self,
        endpoint_url: str,
        api_token: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:

        if not endpoint_url:
            raise ModalConfigurationError(
                "Modal endpoint URL is missing."
            )

        if not api_token:
            raise ModalConfigurationError(
                "Modal API token is missing."
            )

        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_token = api_token
        self.timeout_seconds = timeout_seconds

    @property
    def headers(self) -> dict[str, str]:
        """
        Build HTTP headers.
        """

        return {
            "Authorization": (
                f"Bearer {self.api_token}"
            ),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def generate(
        self,
        request: ModalInferenceRequest,
    ) -> ModalInferenceResponse:
        """
        Execute inference request against Modal.
        """

        payload = request.model_dump()

        logger.info(
            "Sending inference request to Modal."
        )

        try:

            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
            ) as client:

                response = await client.post(
                    self.endpoint_url,
                    headers=self.headers,
                    json=payload,
                )

        except httpx.ConnectError as exc:

            raise ModalConnectionError(
                "Unable to reach Modal endpoint."
            ) from exc

        except httpx.ReadTimeout as exc:

            raise ModalTimeoutError(
                "Modal inference timeout exceeded."
            ) from exc

        except httpx.TimeoutException as exc:

            raise ModalTimeoutError(
                "Modal request timed out."
            ) from exc

        status_code = response.status_code

        if status_code in (401, 403):

            raise ModalAuthenticationError(
                "Modal authentication failed."
            )

        if status_code == 429:

            raise ModalRateLimitError(
                "Modal rate limit exceeded."
            )

        if status_code >= 500:

            raise ModalServerError(
                (
                    "Modal endpoint returned "
                    f"HTTP {status_code}."
                )
            )

        if status_code >= 400:

            self._raise_client_error(
                response,
            )

        return self._parse_response(
            response,
        )

    async def healthcheck(
        self,
    ) -> bool:
        """
        Verify Modal endpoint availability.
        """

        try:

            async with httpx.AsyncClient(
                timeout=10.0,
            ) as client:

                response = await client.get(
                    self.endpoint_url,
                    headers=self.headers,
                )

            return response.status_code < 500

        except Exception:

            logger.exception(
                "Modal healthcheck failed."
            )

            return False

    @staticmethod
    def _parse_response(
        response: httpx.Response,
    ) -> ModalInferenceResponse:
        """
        Parse successful Modal response.
        """

        try:

            payload = response.json()

        except ValueError as exc:

            raise ModalInvalidResponseError(
                "Modal returned invalid JSON."
            ) from exc

        try:

            return ModalInferenceResponse(
                **payload,
            )

        except Exception as exc:

            raise ModalInvalidResponseError(
                (
                    "Modal response does not "
                    "match expected schema."
                )
            ) from exc

    @staticmethod
    def _raise_client_error(
        response: httpx.Response,
    ) -> None:
        """
        Raise detailed client-side exception.
        """

        try:

            payload = ModalErrorResponse(
                **response.json(),
            )

            message = (
                payload.detail
                or payload.error
            )

        except Exception:

            message = (
                "Unknown Modal client error."
            )

        raise ModalInvalidResponseError(
            (
                f"Modal request failed "
                f"(HTTP {response.status_code}) - "
                f"{message}"
            )
        )
