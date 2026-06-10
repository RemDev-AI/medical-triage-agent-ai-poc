# medical-triage-agent-ai-poc/backend/app/api/routes/audit.py

from datetime import datetime
from fastapi import APIRouter

from app.api.schemas import (
    AuditLogEntry,
    AuditResponse
)

router = APIRouter(
    prefix="/audit",
    tags=["Audit"]
)


MOCK_AUDIT_LOGS = [
    AuditLogEntry(
        request_id="req-001",
        endpoint="/triage",
        method="POST",
        status_code=200,
        timestamp=datetime.utcnow(),
        latency_ms=421.5,
        client_ip="127.0.0.1"
    )
]


@router.get(
    "/",
    response_model=AuditResponse
)
async def audit_logs():

    return AuditResponse(
        total_logs=len(MOCK_AUDIT_LOGS),
        logs=MOCK_AUDIT_LOGS
    )
