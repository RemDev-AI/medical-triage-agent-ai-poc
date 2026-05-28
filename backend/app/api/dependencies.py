# medical-triage-agent-ai-poc/backend/app/api/dependencies.py

from datetime import datetime
from fastapi import HTTPException


SUPPORTED_PRIORITY_LEVELS = {
    "low",
    "medium",
    "high",
    "critical"
}


def validate_priority(priority: str) -> str:
    if priority.lower() not in SUPPORTED_PRIORITY_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported priority level: {priority}"
        )

    return priority.lower()


def utc_now() -> datetime:
    return datetime.utcnow()
