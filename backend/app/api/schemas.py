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

    # Ajouté suite au cahier des charges CHSA : l'agent est un outil
    # d'ASSISTANCE au personnel soignant, pas un décideur autonome.
    # True dès que l'extraction du format structuré de la sortie du
    # modèle est incomplète/ambiguë (cf. TriageEngine.parse_response),
    # ou par défaut prudent si l'information n'est pas disponible
    # (mieux vaut une revue humaine superflue qu'une absente).
    # Le frontend / SIH consommant cette réponse DOIT traiter True
    # comme un signal bloquant ou très visible, pas comme une donnée
    # secondaire noyée dans le JSON.
    requires_human_review: bool = Field(default=True)

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


class ClinicalAuditLogEntry(BaseModel):
    """
    Entrée de traçabilité CLINIQUE, distincte d'AuditLogEntry (log
    HTTP générique). Introduite pour répondre à l'exigence CHSA de
    "traçabilité de chaque interaction pour les audits médicaux" —
    AuditLogEntry (request_id/path/status_code/latency_ms) ne permet
    pas de reconstituer une décision clinique lors d'un audit.

    request_id permet la corrélation avec l'entrée AuditLogEntry
    correspondante du journal HTTP générique.
    """

    request_id: str
    timestamp: datetime

    patient_id: Optional[str] = None
    age: Optional[int] = None
    symptoms: Optional[str] = None
    medical_history: Optional[str] = None
    priority_context: Optional[str] = None

    model_name: Optional[str] = None
    raw_priority: Optional[str] = None
    priority_level: Optional[str] = None
    justification: Optional[str] = None
    recommendations: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    requires_human_review: Optional[bool] = None
    raw_response: Optional[str] = None
    latency_seconds: Optional[float] = None

    error: bool = False


class ClinicalAuditResponse(BaseModel):
    total_logs: int
    logs: List[ClinicalAuditLogEntry]
