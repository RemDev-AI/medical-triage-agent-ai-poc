# medical-triage-agent-ai-poc/backend/app/api/routes/triage.py

from __future__ import annotations

import time
import unicodedata

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException

from app.api.schemas import (
    TriageRequest,
    TriageResponse,
)

from app.api.dependencies.inference import (
    get_triage_engine,
)

from app.llm.inference.triage_engine import TriageEngine

from app.monitoring.alerting import (
    alert_manager,
)

router = APIRouter(
    prefix="/triage",
    tags=["Triage"],
)


CONFIDENTIAL_MARKERS: tuple[str, ...] = (
    "confidential",
    "medical records",
    "medical record",
    "patient_id",
    "ssn",
    "social security",
)


_PRIORITY_MAPPING: dict[str, str] = {
    "faible": "low",
    "low": "low",
    "modere": "medium",
    "moyen": "medium",
    "medium": "medium",
    "eleve": "high",
    "high": "high",
    "urgent": "urgent",
    "critique": "urgent",
}


def _normalize_priority(raw_value: str) -> str:
    """
    TriageEngine.parse_response() renvoie "priority" en français
    (ex: "FAIBLE", "ÉLEVÉ"), alors que TriageResponse.priority_level
    doit appartenir au domaine anglais {low, medium, high, urgent}
    attendu par les contrats d'API.

    On normalise la casse et les accents avant de mapper vers la
    valeur canonique. Si la valeur est vide ou inconnue, on retombe
    sur "medium" par prudence et on lève une alerte pour tracer les
    sorties inattendues du modèle plutôt que de les ignorer
    silencieusement.
    """

    if not raw_value:
        return "medium"

    normalized = unicodedata.normalize("NFKD", raw_value.strip().lower())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))

    if normalized not in _PRIORITY_MAPPING:
        try:
            alert_manager.raise_alert(
                code="TRIAGE_UNKNOWN_PRIORITY",
                message=f"Unknown priority value returned by triage engine: {raw_value!r}",
            )
        except Exception:
            pass
        return "medium"

    return _PRIORITY_MAPPING[normalized]


def _strip_confidential_leakage(text: str) -> str:

    sanitized = text

    for marker in CONFIDENTIAL_MARKERS:
        if marker.lower() in sanitized.lower():
            lowered = sanitized.lower()
            marker_lower = marker.lower()
            result = []
            idx = 0
            while True:
                pos = lowered.find(marker_lower, idx)
                if pos == -1:
                    result.append(sanitized[idx:])
                    break
                result.append(sanitized[idx:pos])
                result.append("[redacted]")
                idx = pos + len(marker_lower)
            sanitized = "".join(result)
            lowered = sanitized.lower()

    return sanitized


def _split_recommendations(raw: str) -> list[str]:
    """
    TriageEngine.parse_response() renvoie "recommendations" comme un
    unique bloc de texte (issu de l'extraction de la section
    "RECOMMANDATIONS:" dans la sortie du modèle), alors que
    TriageResponse.recommendations attend une List[str].

    On segmente ligne par ligne, en retirant les puces ("-", "*")
    éventuelles. Si aucune ligne exploitable n'est trouvée mais que le
    modèle a bien renvoyé du texte, on renvoie ce texte comme
    recommandation unique plutôt que de perdre l'information.
    """

    if not raw or raw == "Non disponible":
        return []

    lines = [
        line.strip().lstrip("-*").strip() for line in raw.splitlines() if line.strip()
    ]

    return lines if lines else [raw.strip()]


@router.post(
    "",
    response_model=TriageResponse,
)
async def triage_route(
    payload: TriageRequest,
    triage_engine: TriageEngine = Depends(
        get_triage_engine,
    ),
):
    start_time = time.perf_counter()

    try:

        # ------------------------------------------------------
        # Adaptation de types : TriageRequest expose des champs
        # texte libre (str), alors que TriageEngine.run_triage
        # attend des listes (symptoms: List[str],
        # medical_history: Optional[List[str]]).
        #
        # Hypothèse de migration : on transmet chaque champ texte
        # comme un unique élément de liste plutôt que de le
        # découper arbitrairement (par virgule, etc.), afin de ne
        # pas altérer le contenu clinique saisi par l'utilisateur.
        # A confirmer si un découpage plus fin est souhaité.
        # ------------------------------------------------------

        result = await triage_engine.run_triage(
            patient_age=payload.age if payload.age is not None else 0,
            symptoms=[payload.symptoms],
            medical_history=(
                [payload.medical_history] if payload.medical_history else None
            ),
        )

    except Exception:

        try:
            alert_manager.raise_alert(
                code="TRIAGE_ERROR",
                message="Unexpected error during triage processing.",
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the triage request.",
        )

    latency_seconds = time.perf_counter() - start_time

    latency_ms = latency_seconds * 1000

    try:
        alert_manager.evaluate_latency(latency_ms)
    except Exception:
        pass

    triage_data = result.get("triage", {})

    justification_raw = triage_data.get(
        "justification",
        "",
    )

    recommendations_raw = triage_data.get(
        "recommendations",
        "",
    )

    return TriageResponse(
        priority_level=_normalize_priority(
            triage_data.get(
                "priority",
                "UNKNOWN",
            ),
        ),
        justification=_strip_confidential_leakage(justification_raw),
        recommendations=[
            _strip_confidential_leakage(item)
            for item in _split_recommendations(recommendations_raw)
        ],
        confidence_score=triage_data.get(
            "confidence_score",
            0.0,
        ),
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        latency_seconds=round(
            latency_seconds,
            3,
        ),
    )
