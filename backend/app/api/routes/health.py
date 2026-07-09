# medical-triage-agent-ai-poc/backend/app/api/routes/health.py

from fastapi import APIRouter
from datetime import datetime

from backend.app.api.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        service="medical-triage-api",
        version="1.0.0",
        timestamp=datetime.utcnow(),
    )
