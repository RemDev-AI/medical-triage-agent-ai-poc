# medical-triage-agent-ai-poc/backend/app/monitoring/gpu_monitor.py

from __future__ import annotations

from typing import Any
from typing import Dict

from backend.app.monitoring.inference_monitor import (
    get_active_containers,
)
from backend.app.monitoring.inference_monitor import (
    get_gpu_usage,
)
from backend.app.monitoring.inference_monitor import (
    get_inference_latency,
)


class GPUMonitor:
    """
    Façade de monitoring GPU.

    Dans l'architecture retenue :

    - le GPU n'est pas hébergé localement ;
    - les métriques proviennent du backend
      d'inférence ;
    - cette classe conserve la même interface
      publique afin d'éviter toute modification
      dans les routes FastAPI et les dashboards.
    """

    def get_gpu_stats(self) -> Dict[str, Any]:
        """
        Retourne les métriques consolidées
        du backend d'inférence.
        """

        gpu_metrics = get_gpu_usage()

        latency_metrics = (
            get_inference_latency()
        )

        container_metrics = (
            get_active_containers()
        )

        return {
            "provider": "huggingface",
            "gpu": gpu_metrics,
            "latency": latency_metrics,
            "containers": container_metrics,
        }

    def health_status(self) -> Dict[str, Any]:
        """
        État simplifié utilisé par les endpoints
        health et les dashboards.
        """

        gpu_metrics = get_gpu_usage()

        return {
            "status": "healthy",
            "provider": "huggingface",
            "gpu_utilization_percent": (
                gpu_metrics.get(
                    "gpu_utilization_percent",
                    0,
                )
            ),
            "vram_usage_percent": (
                gpu_metrics.get(
                    "vram_usage_percent",
                    0,
                )
            ),
        }


gpu_monitor = GPUMonitor()
