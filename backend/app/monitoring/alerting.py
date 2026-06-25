# medical-triage-agent-ai-poc/backend/app/monitoring/alerting.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict
from typing import List

from backend.app.monitoring.gpu_monitor import (
    gpu_monitor,
)
from backend.app.monitoring.latency_monitor import (
    latency_monitor,
)
from backend.app.monitoring.request_tracker import (
    request_tracker,
)


@dataclass
class Alert:
    level: str
    code: str
    message: str
    timestamp: str


class AlertManager:
    """
    Gestionnaire centralisé d'alertes.

    Compatible :

    - Hugging Face Spaces
    - Backend d'inférence
    - FastAPI
    - Streamlit Dashboard
    """

    LATENCY_WARNING_MS = 1000
    LATENCY_CRITICAL_MS = 3000

    ERROR_RATE_WARNING = 5.0
    ERROR_RATE_CRITICAL = 15.0

    GPU_UTILIZATION_WARNING = 80.0
    GPU_UTILIZATION_CRITICAL = 95.0

    VRAM_UTILIZATION_WARNING = 80.0
    VRAM_UTILIZATION_CRITICAL = 95.0

    CONTAINER_WARNING = 10
    CONTAINER_CRITICAL = 25

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

    def raise_alert(
        self,
        category: str,
        message: str,
        level: str = "ERROR",
    ) -> None:
        """
        API utilisée par les routes FastAPI.
        """

        self._create_alert(
            level=level,
            code=category,
            message=message,
        )

    def evaluate_latency(
        self,
        latency_ms: float | None = None,
    ) -> None:
        """
        Analyse des temps de réponse.
        Compatible avec les routes FastAPI.
        """

        if latency_ms is None:

            stats = latency_monitor.stats()

            latency_ms = stats.get(
                "avg_ms",
                0.0,
            )

        if latency_ms >= self.LATENCY_CRITICAL_MS:

            self._create_alert(
                "CRITICAL",
                "LATENCY_CRITICAL",
                (
                    f"Latency is "
                    f"{latency_ms:.2f} ms"
                ),
            )

        elif latency_ms >= self.LATENCY_WARNING_MS:

            self._create_alert(
                "WARNING",
                "LATENCY_WARNING",
                (
                    f"Latency is "
                    f"{latency_ms:.2f} ms"
                ),
            )

    def evaluate_errors(self) -> None:
        """
        Analyse du taux d'erreur API.
        """

        stats = request_tracker.get_stats()

        error_rate = stats.get(
            "error_rate_percent",
            0.0,
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
        """
        Analyse des métriques GPU.
        """

        metrics = gpu_monitor.get_gpu_stats()

        gpu_metrics = metrics.get(
            "gpu",
            {},
        )

        gpu_utilization = gpu_metrics.get(
            "gpu_utilization_percent",
            0.0,
        )

        vram_utilization = gpu_metrics.get(
            "vram_usage_percent",
            0.0,
        )

        if (
            gpu_utilization
            >= self.GPU_UTILIZATION_CRITICAL
        ):
            self._create_alert(
                "CRITICAL",
                "GPU_UTILIZATION_CRITICAL",
                (
                    f"GPU utilization is "
                    f"{gpu_utilization}%"
                ),
            )

        elif (
            gpu_utilization
            >= self.GPU_UTILIZATION_WARNING
        ):
            self._create_alert(
                "WARNING",
                "GPU_UTILIZATION_WARNING",
                (
                    f"GPU utilization is "
                    f"{gpu_utilization}%"
                ),
            )

        if (
            vram_utilization
            >= self.VRAM_UTILIZATION_CRITICAL
        ):
            self._create_alert(
                "CRITICAL",
                "VRAM_UTILIZATION_CRITICAL",
                (
                    f"VRAM utilization is "
                    f"{vram_utilization}%"
                ),
            )

        elif (
            vram_utilization
            >= self.VRAM_UTILIZATION_WARNING
        ):
            self._create_alert(
                "WARNING",
                "VRAM_UTILIZATION_WARNING",
                (
                    f"VRAM utilization is "
                    f"{vram_utilization}%"
                ),
            )

    def evaluate_containers(self) -> None:
        """
        Analyse de la charge du backend
        d'inférence.
        """

        metrics = gpu_monitor.get_gpu_stats()

        containers = metrics.get(
            "containers",
            {},
        )

        active_containers = containers.get(
            "active_containers",
            0,
        )

        if (
            active_containers
            >= self.CONTAINER_CRITICAL
        ):
            self._create_alert(
                "CRITICAL",
                "CONTAINER_CRITICAL",
                (
                    f"Active containers: "
                    f"{active_containers}"
                ),
            )

        elif (
            active_containers
            >= self.CONTAINER_WARNING
        ):
            self._create_alert(
                "WARNING",
                "CONTAINER_WARNING",
                (
                    f"Active containers: "
                    f"{active_containers}"
                ),
            )

    def evaluate_all(self) -> None:

        self.evaluate_latency()
        self.evaluate_errors()
        self.evaluate_gpu()
        self.evaluate_containers()

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
