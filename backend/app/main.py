# medical-triage-agent-ai-poc/backend/app/main.py

from fastapi import FastAPI, Depends

from app.api.router import api_router

from app.api.middleware.auth_middleware import (
    JWTAuthMiddleware
)

from app.api.middleware.logging_middleware import (
    AuditLoggingMiddleware
)

from app.api.middleware.security_middleware import (
    setup_cors
)

from app.core.rate_limiter import rate_limit


app = FastAPI(
    title="Medical Triage AI API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

setup_cors(app)

app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(JWTAuthMiddleware)
app.include_router(
    api_router,
    dependencies=[Depends(rate_limit)]
)


@app.get("/")
async def root():
    return {
        "service": "Medical Triage AI",
        "status": "running"
    }
