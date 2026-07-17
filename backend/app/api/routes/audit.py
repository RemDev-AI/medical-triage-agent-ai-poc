# medical-triage-agent-ai-poc/backend/app/api/routes/audit.py

from __future__ import annotations

from fastapi import APIRouter
from fastapi import Query

from app.api.schemas import (
    AuditLogEntry,
    AuditResponse,
)
from app.monitoring.audit_store import (
    count_entries,
    read_entries,
)

router = APIRouter(prefix="/audit", tags=["Audit"])


@router.get(
    "",
    response_model=AuditResponse,
)
async def audit_logs(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description=(
            "Nombre maximal d'entrées retournées, "
            "des plus récentes aux plus anciennes."
        ),
    ),
):
    """
    Traçabilité des interactions API (correctif
    étape 3).

    Remplace les données mockées en dur par une
    lecture réelle du journal persistant
    (backend/app/monitoring/audit_store.py),
    alimenté par AuditLoggingMiddleware pour
    chaque requête HTTP traitée par l'API.

    Limite d'usage : ce journal est local au
    conteneur et non répliqué (cf. audit_store.py).
    """

    raw_entries = read_entries(limit=limit)

    logs = [
        AuditLogEntry(
            request_id=entry.get("request_id", "unknown"),
            endpoint=entry.get("path", "unknown"),
            method=entry.get("method", "unknown"),
            status_code=entry.get("status_code", 0),
            timestamp=entry.get("timestamp"),
            latency_ms=entry.get("latency_ms", 0.0),
            client_ip=entry.get("client_ip"),
        )
        for entry in raw_entries
    ]

    return AuditResponse(
        total_logs=count_entries(),
        logs=logs,
    )
