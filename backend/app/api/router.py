# medical-triage-agent-ai-poc/backend/app/api/router.py

from fastapi import APIRouter

from app.api.routes.health import router as health_router

from app.api.routes.inference import router as inference_router

from app.api.routes.triage import router as triage_router

from app.api.routes.audit import router as audit_router

from app.api.routes.monitoring import router as monitoring_router


api_router = APIRouter()


# ------------------------------------------------------------------
# Core API Routes
# ------------------------------------------------------------------

api_router.include_router(health_router)

api_router.include_router(inference_router)

api_router.include_router(triage_router)

api_router.include_router(audit_router)


# ------------------------------------------------------------------
# Monitoring & Observability Routes (PHASE 7)
# ------------------------------------------------------------------

api_router.include_router(monitoring_router)
