# medical-triage-agent-ai-poc/backend/app/api/middleware/logging_middleware.py

from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.monitoring.latency_monitor import (
    latency_monitor,
)
from backend.app.monitoring.request_tracker import (
    request_tracker,
)


logger = logging.getLogger("audit_logger")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)


class AuditLoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        request_id = str(uuid.uuid4())

        start_time = time.perf_counter()

        endpoint = request.url.path
        method = request.method

        request_tracker.start_request(
            endpoint=endpoint,
            method=method,
        )

        try:
            response = await call_next(request)

            latency_ms = round(
                (time.perf_counter() - start_time)
                * 1000,
                2,
            )

            # ==================================================
            # MONITORING
            # ==================================================

            latency_monitor.record(
                latency_ms
            )

            success = (
                response.status_code < 400
            )

            request_tracker.end_request(
                success=success,
            )

            audit_log = {
                "request_id": request_id,
                "timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
                "method": method,
                "path": endpoint,
                "status_code": response.status_code,
                "client_ip": (
                    request.client.host
                    if request.client
                    else "unknown"
                ),
                "latency_ms": latency_ms,
            }

            logger.info(
                json.dumps(
                    audit_log,
                    ensure_ascii=False,
                )
            )

            response.headers[
                "X-Request-ID"
            ] = request_id

            return response

        except Exception:

            latency_ms = round(
                (time.perf_counter() - start_time)
                * 1000,
                2,
            )

            # ==================================================
            # MONITORING
            # ==================================================

            latency_monitor.record(
                latency_ms
            )

            request_tracker.end_request(
                success=False,
            )

            audit_log = {
                "request_id": request_id,
                "timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
                "method": method,
                "path": endpoint,
                "status_code": 500,
                "client_ip": (
                    request.client.host
                    if request.client
                    else "unknown"
                ),
                "latency_ms": latency_ms,
                "error": True,
            }

            logger.exception(
                json.dumps(
                    audit_log,
                    ensure_ascii=False,
                )
            )

            raise
