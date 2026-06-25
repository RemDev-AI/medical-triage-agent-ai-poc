# medical-triage-agent-ai-poc/backend/app/monitoring/latency_monitor.py

from __future__ import annotations

import statistics
import threading
import time

from collections import deque
from contextlib import contextmanager
from typing import Dict
from typing import Iterator
from typing import List


class LatencyMonitor:
    """
    Monitoring des temps de réponse.

    Utilisé par :

    - routes/inference.py
    - routes/triage.py
    - routes/monitoring.py
    - alerting.py

    Métriques exposées :

    - moyenne
    - minimum
    - maximum
    - p95
    - p99
    - nombre de requêtes
    """

    def __init__(
        self,
        max_samples: int = 5000,
    ) -> None:

        self.max_samples = max_samples

        self.latencies = deque(
            maxlen=max_samples
        )

        self.lock = threading.Lock()

    def record(
        self,
        latency_ms: float,
    ) -> None:

        with self.lock:
            self.latencies.append(
                latency_ms
            )

    def count(self) -> int:

        with self.lock:
            return len(
                self.latencies
            )

    def reset(self) -> None:

        with self.lock:
            self.latencies.clear()

    def get_all(self) -> List[float]:

        with self.lock:
            return list(
                self.latencies
            )

    def _percentile(
        self,
        data: List[float],
        percentile: float,
    ) -> float:

        if not data:
            return 0.0

        data = sorted(data)

        k = (
            (len(data) - 1)
            * percentile
        )

        f = int(k)

        c = min(
            f + 1,
            len(data) - 1,
        )

        if f == c:
            return float(
                data[int(k)]
            )

        d0 = data[f] * (c - k)

        d1 = data[c] * (k - f)

        return float(
            d0 + d1
        )

    def stats(self) -> Dict[str, float]:
        """
        Retourne les statistiques
        de latence consolidées.
        """

        data = self.get_all()

        if not data:

            return {
                "count": 0,
                "avg_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
            }

        return {
            "count": len(data),
            "avg_ms": round(
                statistics.mean(data),
                2,
            ),
            "min_ms": round(
                min(data),
                2,
            ),
            "max_ms": round(
                max(data),
                2,
            ),
            "p95_ms": round(
                self._percentile(
                    data,
                    0.95,
                ),
                2,
            ),
            "p99_ms": round(
                self._percentile(
                    data,
                    0.99,
                ),
                2,
            ),
        }


latency_monitor = LatencyMonitor()


@contextmanager
def track_latency() -> Iterator[None]:
    """
    Context manager :

    Example
    -------
    with track_latency():
        ...
    """

    start = time.perf_counter()

    try:

        yield

    finally:

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000

        latency_monitor.record(
            latency_ms
        )
