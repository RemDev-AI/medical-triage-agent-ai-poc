# medical-triage-agent-ai-poc/backend/app/api/middleware/logging_middleware.py

import json
import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger("audit_logger")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)


class AuditLoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        request_id = str(uuid.uuid4())

        start_time = time.perf_counter()

        response = await call_next(request)

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        audit_log = {
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "client_ip": request.client.host,
            "latency_ms": latency_ms
        }

        logger.info(json.dumps(audit_log))

        response.headers["X-Request-ID"] = request_id

        return response
