# medical-triage-agent-ai-poc/backend/app/monitoring/alerting.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict
from typing import List

from backend.app.monitoring.gpu_monitor import gpu_monitor
from backend.app.monitoring.latency_monitor import latency_monitor
from backend.app.monitoring.request_tracker import request_tracker


@dataclass
class Alert:
    level: str
    code: str
    message: str
    timestamp: str


class AlertManager:
    """
    Gestionnaire centralisé d'alertes.

    Niveaux :
    - INFO
    - WARNING
    - CRITICAL
    """

    LATENCY_WARNING_MS = 1000
    LATENCY_CRITICAL_MS = 3000

    ERROR_RATE_WARNING = 5.0
    ERROR_RATE_CRITICAL = 15.0

    GPU_USAGE_WARNING = 80.0
    GPU_USAGE_CRITICAL = 95.0

    MIN_THROUGHPUT = 0.01

    def __init__(self) -> None:
        self.alert_history: List[Alert] = []

    def _create_alert(
        self,
        level: str,
        code: str,
        message: str,
    ) -> None:

        alert = Alert(
            level=level,
            code=code,
            message=message,
            timestamp=datetime.utcnow().isoformat(),
        )

        self.alert_history.append(alert)

    def evaluate_latency(self) -> None:

        stats = latency_monitor.stats()

        avg_latency = stats.get(
            "avg_ms",
            0,
        )

        if avg_latency >= self.LATENCY_CRITICAL_MS:

            self._create_alert(
                "CRITICAL",
                "LATENCY_CRITICAL",
                (
                    f"Average latency is "
                    f"{avg_latency} ms"
                ),
            )

        elif avg_latency >= self.LATENCY_WARNING_MS:

            self._create_alert(
                "WARNING",
                "LATENCY_WARNING",
                (
                    f"Average latency is "
                    f"{avg_latency} ms"
                ),
            )

    def evaluate_errors(self) -> None:

        stats = request_tracker.get_stats()

        error_rate = stats.get(
            "error_rate_percent",
            0,
        )

        if error_rate >= self.ERROR_RATE_CRITICAL:

            self._create_alert(
                "CRITICAL",
                "ERROR_RATE_CRITICAL",
                (
                    f"Error rate is "
                    f"{error_rate}%"
                ),
            )

        elif error_rate >= self.ERROR_RATE_WARNING:

            self._create_alert(
                "WARNING",
                "ERROR_RATE_WARNING",
                (
                    f"Error rate is "
                    f"{error_rate}%"
                ),
            )

    def evaluate_gpu(self) -> None:

        gpu_stats = gpu_monitor.get_gpu_stats()

        if not gpu_stats["cuda_available"]:
            return

        usage = gpu_stats.get(
            "vram_usage_percent",
            0,
        )

        if usage >= self.GPU_USAGE_CRITICAL:

            self._create_alert(
                "CRITICAL",
                "GPU_CRITICAL",
                (
                    f"GPU VRAM usage is "
                    f"{usage}%"
                ),
            )

        elif usage >= self.GPU_USAGE_WARNING:

            self._create_alert(
                "WARNING",
                "GPU_WARNING",
                (
                    f"GPU VRAM usage is "
                    f"{usage}%"
                ),
            )

    def evaluate_throughput(self) -> None:

        throughput = (
            gpu_monitor.get_throughput()
        )

        if throughput <= self.MIN_THROUGHPUT:

            self._create_alert(
                "INFO",
                "LOW_TRAFFIC",
                (
                    "Low throughput detected "
                    f"({throughput} rps)"
                ),
            )

    def evaluate_all(self) -> None:

        self.evaluate_latency()
        self.evaluate_errors()
        self.evaluate_gpu()
        self.evaluate_throughput()

    def get_alerts(self) -> List[Dict]:

        return [
            {
                "level": alert.level,
                "code": alert.code,
                "message": alert.message,
                "timestamp": alert.timestamp,
            }
            for alert in self.alert_history
        ]

    def get_active_summary(self) -> Dict:

        alerts = self.get_alerts()

        critical = len(
            [
                a
                for a in alerts
                if a["level"] == "CRITICAL"
            ]
        )

        warning = len(
            [
                a
                for a in alerts
                if a["level"] == "WARNING"
            ]
        )

        info = len(
            [
                a
                for a in alerts
                if a["level"] == "INFO"
            ]
        )

        return {
            "critical": critical,
            "warning": warning,
            "info": info,
            "total": len(alerts),
        }

    def clear(self) -> None:

        self.alert_history.clear()


alert_manager = AlertManager()
