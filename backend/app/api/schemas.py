# medical-triage-agent-ai-poc/backend/app/api/schemas.py

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str
    timestamp: datetime


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=8000)
    max_new_tokens: int = Field(default=256, ge=32, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)


class GenerateResponse(BaseModel):
    generated_text: str
    model_name: str
    latency_seconds: float
    timestamp: datetime


class TriageRequest(BaseModel):
    patient_id: Optional[str] = None

    # Texte libre décrivant les symptômes. Corrige le bug: était
    # List[str], ce qui rejetait en 422 tout payload envoyant une
    # chaîne (cf. test_prompt_injection_attempt). Depuis la migration
    # vers l'inférence locale, routes/triage.py convertit ce champ en
    # `[symptoms]` avant l'appel à TriageEngine.run_triage(), qui
    # attend `symptoms: List[str]`.
    symptoms: str = Field(..., min_length=1, max_length=2000)

    # Texte libre également. Depuis la migration vers l'inférence
    # locale, routes/triage.py convertit ce champ en
    # `[medical_history]` avant l'appel à TriageEngine.run_triage(),
    # qui attend `medical_history: Optional[List[str]]`.
    medical_history: Optional[str] = Field(default=None, max_length=2000)

    age: Optional[int] = Field(default=None, ge=0, le=120)

    priority_context: Optional[str] = Field(default="standard", max_length=256)


class TriageResponse(BaseModel):
    priority_level: str
    justification: str
    recommendations: List[str]

    confidence_score: float = Field(..., ge=0.0, le=1.0)

    generated_at: datetime
    latency_seconds: float


class AuditLogEntry(BaseModel):
    request_id: str
    endpoint: str
    method: str
    status_code: int
    timestamp: datetime
    latency_ms: float
    client_ip: Optional[str] = None


class AuditResponse(BaseModel):
    total_logs: int
    logs: List[AuditLogEntry]
