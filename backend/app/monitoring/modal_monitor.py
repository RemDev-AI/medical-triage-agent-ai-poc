# medical-triage-agent-ai-poc/backend/app/monitoring/modal_monitor.py

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any
from typing import Dict

logger = logging.getLogger(__name__)


class ModalMonitoringError(Exception):
    """
    Exception spécifique au monitoring Modal.
    """


def _get_modal_environment() -> Dict[str, str]:
    """
    Retourne les informations de configuration Modal.

    Variables attendues :

    MODAL_APP_NAME
    MODAL_ENVIRONMENT
    """

    return {
        "app_name": os.getenv(
            "MODAL_APP_NAME",
            "medical-triage-agent-ai-poc",
        ),
        "environment": os.getenv(
            "MODAL_ENVIRONMENT",
            "production",
        ),
    }


def get_gpu_usage() -> Dict[str, Any]:
    """
    Retourne les métriques GPU Modal.

    Cette fonction constitue le point d'entrée unique
    pour la récupération des métriques GPU.

    Une future intégration pourra appeler :

    - Modal Metrics API
    - Modal Graph API
    - OpenTelemetry Exporter
    - Prometheus Gateway

    sans modifier le reste du projet.
    """

    try:
        env = _get_modal_environment()

        #
        # Placeholder architecture.
        #
        # Remplacer ici par les appels officiels
        # Modal Metrics lorsqu'ils seront exposés
        # dans l'infrastructure cible.
        #

        return {
            "provider": "modal",
            "app_name": env["app_name"],
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
            "Unable to retrieve Modal GPU metrics"
        )

        raise ModalMonitoringError(
            str(exc)
        ) from exc


def get_active_containers() -> Dict[str, Any]:
    """
    Retourne les informations d'exécution Modal.

    Métriques visées :

    - conteneurs actifs
    - workers disponibles
    - autoscaling
    - file d'attente
    """

    try:
        env = _get_modal_environment()

        return {
            "provider": "modal",
            "app_name": env["app_name"],
            "active_containers": 0,
            "pending_containers": 0,
            "running_jobs": 0,
            "queue_size": 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as exc:
        logger.exception(
            "Unable to retrieve Modal containers"
        )

        raise ModalMonitoringError(
            str(exc)
        ) from exc


def get_inference_latency() -> Dict[str, Any]:
    """
    Retourne les métriques de latence.

    Sources futures :

    - FastAPI middleware
    - Modal endpoint metrics
    - Prometheus exporter
    """

    try:
        env = _get_modal_environment()

        return {
            "provider": "modal",
            "app_name": env["app_name"],
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

        raise ModalMonitoringError(
            str(exc)
        ) from exc


def get_modal_health() -> Dict[str, Any]:
    """
    Vue consolidée utilisée par :

    - endpoint /health
    - dashboard Streamlit
    - monitoring interne
    """

    return {
        "provider": "modal",
        "gpu": get_gpu_usage(),
        "containers": get_active_containers(),
        "latency": get_inference_latency(),
    }
