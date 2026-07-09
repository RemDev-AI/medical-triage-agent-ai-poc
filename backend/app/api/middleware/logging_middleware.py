# medical-triage-agent-ai-poc/backend/app/api/middleware/logging_middleware.py

from __future__ import annotations

import json
import logging
import time
import uuid

from datetime import datetime

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.monitoring.latency_monitor import (
    latency_monitor,
)
from backend.app.monitoring.request_tracker import (
    request_tracker,
)
from backend.app.monitoring.audit_store import (
    record_entry,
)


logger = logging.getLogger("audit_logger")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware de journalisation.

    Responsabilités :

    - audit des requêtes (persisté via audit_store,
      cf. routes/audit.py — correctif étape 3)
    - mesure de latence (source unique de vérité
      pour latency_monitor, cf. NOTE ci-dessous)
    - suivi du trafic (source unique de vérité
      pour request_tracker)
    - attribution d'un Request ID
    - alimentation du monitoring

    NOTE (correctif étape 3) :
    Ce middleware est l'UNIQUE point d'incrémentation
    de request_tracker et de latency_monitor pour
    l'ensemble des routes. Les routes individuelles
    (inference.py, triage.py) ne doivent plus
    dupliquer ces appels.
    """

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
                (time.perf_counter() - start_time) * 1000,
                2,
            )

            latency_monitor.record(latency_ms)

            success = response.status_code < 400

            request_tracker.end_request(
                success=success,
            )

            client_ip = request.client.host if request.client else "unknown"

            audit_log = {
                "request_id": request_id,
                "timestamp": (datetime.utcnow().isoformat()),
                "method": method,
                "path": endpoint,
                "status_code": (response.status_code),
                "client_ip": client_ip,
                "latency_ms": latency_ms,
            }

            logger.info(
                json.dumps(
                    audit_log,
                    ensure_ascii=False,
                )
            )

            record_entry(audit_log)

            response.headers["X-Request-ID"] = request_id

            return response

        except Exception:

            latency_ms = round(
                (time.perf_counter() - start_time) * 1000,
                2,
            )

            latency_monitor.record(latency_ms)

            request_tracker.end_request(
                success=False,
            )

            client_ip = request.client.host if request.client else "unknown"

            audit_log = {
                "request_id": request_id,
                "timestamp": (datetime.utcnow().isoformat()),
                "method": method,
                "path": endpoint,
                "status_code": 500,
                "client_ip": client_ip,
                "latency_ms": latency_ms,
                "error": True,
            }

            logger.exception(
                json.dumps(
                    audit_log,
                    ensure_ascii=False,
                )
            )

            record_entry(audit_log)

            raise
