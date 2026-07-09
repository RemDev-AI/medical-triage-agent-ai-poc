# medical-triage-agent-ai-poc/backend/app/api/routes/monitoring.py

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi import HTTPException

from backend.app.monitoring.alerting import alert_manager
from backend.app.monitoring.gpu_monitor import gpu_monitor
from backend.app.monitoring.inference_monitor import (
    get_inference_health,
)
from backend.app.monitoring.latency_monitor import (
    latency_monitor,
)
from backend.app.monitoring.request_tracker import (
    request_tracker,
)

router = APIRouter(
    prefix="/monitoring",
    tags=["Monitoring"],
)


@router.get("/health")
async def monitoring_health():
    """
    Health check du module Monitoring.
    """

    return {
        "status": "healthy",
        "service": "monitoring",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


@router.get("/latency")
async def get_latency_metrics():
    """
    Statistiques de latence API.
    """

    return {
        "status": "success",
        "metrics": latency_monitor.stats(),
    }


@router.get("/gpu")
async def get_gpu_metrics():
    """
    Monitoring GPU disponible sur
    l'infrastructure d'inférence.
    """

    try:

        return {
            "status": "success",
            "metrics": gpu_monitor.get_gpu_stats(),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(f"GPU monitoring error: {exc}"),
        )


@router.get("/requests")
async def get_request_metrics():
    """
    Monitoring du trafic API.
    """

    return {
        "status": "success",
        "metrics": request_tracker.get_stats(),
    }


@router.get("/alerts")
async def get_alerts():
    """
    Alertes système.
    """

    return {
        "status": "success",
        "alerts": alert_manager.get_alerts(),
    }


@router.get("/inference")
async def get_inference_metrics():
    """
    Monitoring consolidé du backend
    d'inférence.
    """

    try:

        return {
            "status": "success",
            "metrics": get_inference_health(),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(f"Inference monitoring error: " f"{exc}"),
        )


@router.get("/overview")
async def get_monitoring_overview():
    """
    Dashboard global utilisé par :

    - Streamlit
    - API Monitoring
    - Intégration CI/CD
    """

    try:
        gpu_stats = gpu_monitor.get_gpu_stats()
    except Exception:
        gpu_stats = {
            "cuda_available": False,
        }

    try:
        alerts = alert_manager.get_alerts()
    except Exception:
        alerts = []

    try:
        inference = get_inference_health()
    except Exception:
        inference = {}

    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "latency": latency_monitor.stats(),
        "requests": request_tracker.get_stats(),
        "gpu": gpu_stats,
        "alerts": alerts,
        "inference": inference,
    }


@router.get("/summary")
async def monitoring_summary():
    """
    Résumé synthétique destiné aux
    dashboards et aux tests.
    """

    latency = latency_monitor.stats()

    requests = request_tracker.get_stats()

    try:
        gpu = gpu_monitor.get_gpu_stats()
    except Exception:
        gpu = {}

    return {
        "healthy": True,
        "total_requests": requests.get(
            "total_requests",
            0,
        ),
        "error_count": requests.get(
            "failed_requests",
            0,
        ),
        "average_latency_ms": latency.get(
            "avg_ms",
            0.0,
        ),
        "gpu_available": gpu.get(
            "cuda_available",
            False,
        ),
        "active_alerts": len(alert_manager.get_alerts()),
    }
