# medical-triage-agent-ai-poc/backend/app/monitoring/request_tracker.py

from __future__ import annotations

import threading

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict


class RequestTracker:
    """
    Suivi du trafic API.

    Métriques :

    - total requests
    - success requests
    - failed requests
    - active requests
    - error rate
    - endpoint usage
    - HTTP methods usage

    Compatible avec :

    - FastAPI
    - Streamlit
    - Monitoring API
    - Alerting
    """

    def __init__(self) -> None:

        self.lock = threading.Lock()

        self.started_at = datetime.now(timezone.utc)

        self.total_requests = 0
        self.success_requests = 0
        self.failed_requests = 0
        self.active_requests = 0

        self.endpoint_counter = defaultdict(int)

        self.method_counter = defaultdict(int)

    #
    # API utilisée par les middlewares
    #

    def start_request(
        self,
        endpoint: str,
        method: str,
    ) -> None:

        with self.lock:

            self.total_requests += 1

            self.active_requests += 1

            self.endpoint_counter[endpoint] += 1

            self.method_counter[method.upper()] += 1

    def end_request(
        self,
        success: bool = True,
    ) -> None:

        with self.lock:

            self.active_requests = max(
                self.active_requests - 1,
                0,
            )

            if success:
                self.success_requests += 1
            else:
                self.failed_requests += 1

    #
    # API utilisée par les routes
    #

    def increment_total_requests(
        self,
    ) -> None:

        with self.lock:

            self.total_requests += 1

    def increment_success_requests(
        self,
    ) -> None:

        with self.lock:

            self.success_requests += 1

    def increment_error_requests(
        self,
    ) -> None:

        with self.lock:

            self.failed_requests += 1

    def get_error_rate(self) -> float:

        if self.total_requests == 0:
            return 0.0

        return round(
            (self.failed_requests / self.total_requests) * 100,
            2,
        )

    def reset(self) -> None:

        with self.lock:

            self.total_requests = 0

            self.success_requests = 0

            self.failed_requests = 0

            self.active_requests = 0

            self.endpoint_counter.clear()

            self.method_counter.clear()

            self.started_at = datetime.now(timezone.utc)

    def get_stats(self) -> Dict:

        with self.lock:

            return {
                "started_at": (self.started_at.isoformat()),
                "total_requests": (self.total_requests),
                "success_requests": (self.success_requests),
                "failed_requests": (self.failed_requests),
                "active_requests": (self.active_requests),
                "error_rate_percent": (self.get_error_rate()),
                "endpoint_usage": dict(self.endpoint_counter),
                "method_usage": dict(self.method_counter),
            }

    def health_status(self) -> Dict:

        stats = self.get_stats()

        return {
            "status": "healthy",
            "total_requests": stats.get(
                "total_requests",
                0,
            ),
            "active_requests": stats.get(
                "active_requests",
                0,
            ),
            "error_rate_percent": stats.get(
                "error_rate_percent",
                0.0,
            ),
        }


request_tracker = RequestTracker()
