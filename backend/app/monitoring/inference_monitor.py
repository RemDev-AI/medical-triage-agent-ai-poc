# medical-triage-agent-ai-poc/backend/app/monitoring/inference_monitor.py

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any
from typing import Dict

logger = logging.getLogger(__name__)


class InferenceMonitoringError(Exception):
    """
    Exception spécifique au monitoring
    du backend d'inférence.
    """


def _get_inference_environment() -> Dict[str, str]:
    """
    Retourne les informations de configuration
    du backend d'inférence.

    Variables attendues :

    INFERENCE_BACKEND_NAME
    INFERENCE_ENVIRONMENT
    """

    return {
        "backend_name": os.getenv(
            "INFERENCE_BACKEND_NAME",
            "medical-triage-agent-ai-poc-api",
        ),
        "environment": os.getenv(
            "INFERENCE_ENVIRONMENT",
            "production",
        ),
    }


def get_gpu_usage() -> Dict[str, Any]:
    """
    Retourne les métriques GPU du backend
    d'inférence.

    Cette fonction constitue le point d'entrée
    unique pour la récupération des métriques GPU.

    Une future intégration pourra appeler :

    - Hugging Face Space API Metrics
    - vLLM Metrics
    - TGI Metrics
    - OpenTelemetry Exporter
    - Prometheus Gateway

    sans modifier le reste du projet.
    """

    try:
        env = _get_inference_environment()

        return {
            "provider": "huggingface",
            "backend_name": env["backend_name"],
            "environment": env["environment"],
            "gpu_utilization_percent": 0.0,
            "vram_usage_percent": 0.0,
            "vram_total_mb": None,
            "vram_used_mb": None,
            "gpu_type": None,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Unable to retrieve GPU metrics"
        )

        raise InferenceMonitoringError(
            str(exc)
        ) from exc


def get_active_containers() -> Dict[str, Any]:
    """
    Retourne les informations d'exécution
    du backend d'inférence.

    Métriques visées :

    - réplicas actifs
    - workers disponibles
    - autoscaling
    - file d'attente
    """

    try:
        env = _get_inference_environment()

        return {
            "provider": "huggingface",
            "backend_name": env["backend_name"],
            "active_containers": 0,
            "pending_containers": 0,
            "running_jobs": 0,
            "queue_size": 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Unable to retrieve backend metrics"
        )

        raise InferenceMonitoringError(
            str(exc)
        ) from exc


def get_inference_latency() -> Dict[str, Any]:
    """
    Retourne les métriques de latence.

    Sources futures :

    - FastAPI middleware
    - Backend inference metrics
    - Prometheus exporter
    """

    try:
        env = _get_inference_environment()

        return {
            "provider": "huggingface",
            "backend_name": env["backend_name"],
            "average_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "requests_per_minute": 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Unable to retrieve latency metrics"
        )

        raise InferenceMonitoringError(
            str(exc)
        ) from exc


def get_inference_health() -> Dict[str, Any]:
    """
    Vue consolidée utilisée par :

    - endpoint /health
    - dashboard Streamlit
    - monitoring interne
    """

    return {
        "provider": "huggingface",
        "gpu": get_gpu_usage(),
        "containers": get_active_containers(),
        "latency": get_inference_latency(),
    }
