# medical-triage-agent-ai-poc/backend/app/monitoring/gpu_monitor.py

from __future__ import annotations

from typing import Any
from typing import Dict

from backend.app.monitoring.modal_monitor import get_active_containers
from backend.app.monitoring.modal_monitor import get_gpu_usage
from backend.app.monitoring.modal_monitor import get_inference_latency


class GPUMonitor:
    """
    Façade de monitoring GPU.

    Dans l'architecture Hugging Face + Modal :

    - le GPU n'est plus local ;
    - les métriques sont récupérées depuis Modal ;
    - cette classe conserve la même interface publique
      afin d'éviter de modifier le reste du code.
    """

    def get_gpu_stats(self) -> Dict[str, Any]:
        """
        Retourne les métriques GPU Modal.
        """

        gpu_metrics = get_gpu_usage()

        latency_metrics = get_inference_latency()

        container_metrics = get_active_containers()

        return {
            "provider": "modal",
            "gpu": gpu_metrics,
            "latency": latency_metrics,
            "containers": container_metrics,
        }

    def health_status(self) -> Dict[str, Any]:
        """
        État simplifié utilisé par les endpoints health.
        """

        gpu_metrics = get_gpu_usage()

        return {
            "status": "healthy",
            "provider": "modal",
            "gpu_utilization_percent": gpu_metrics.get(
                "gpu_utilization_percent",
                0,
            ),
            "vram_usage_percent": gpu_metrics.get(
                "vram_usage_percent",
                0,
            ),
        }


gpu_monitor = GPUMonitor()
