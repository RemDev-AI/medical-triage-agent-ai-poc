# medical-triage-agent-ai-poc/backend/app/api/router.py

from fastapi import APIRouter

from api.routes.health import router as health_router
from api.routes.inference import router as inference_router
from api.routes.triage import router as triage_router
from api.routes.audit import router as audit_router


api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(inference_router)
api_router.include_router(triage_router)
api_router.include_router(audit_router)
