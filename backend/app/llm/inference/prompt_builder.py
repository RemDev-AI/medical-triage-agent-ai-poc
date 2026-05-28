# medical-triage-agent-ai-poc/backend/app/llm/inference/prompt_builder.py

"""
Prompt builder for medical triage inference.
"""

from __future__ import annotations

from typing import Dict, List, Optional


SYSTEM_PROMPT = """
Tu es un assistant IA spécialisé en triage médical.

Objectifs :
- analyser les symptômes patients ;
- identifier le niveau de priorité ;
- produire une justification clinique concise ;
- proposer des recommandations prudentes ;
- éviter les hallucinations médicales ;
- ne jamais produire de diagnostic définitif.

Règles :
- rester factuel ;
- signaler les urgences vitales ;
- recommander une consultation médicale si nécessaire ;
- ne jamais inventer de données patient ;
- utiliser un ton clinique professionnel.

Niveaux de priorité autorisés :
- CRITIQUE
- URGENT
- MODÉRÉ
- FAIBLE

Format de réponse obligatoire :

PRIORITÉ:
JUSTIFICATION:
RECOMMANDATIONS:
""".strip()


def format_symptoms(symptoms: List[str]) -> str:
    """
    Format symptoms list.
    """

    if not symptoms:
        return "- Aucun symptôme renseigné"

    return "\n".join(
        f"- {symptom.strip()}"
        for symptom in symptoms
    )


def format_medical_history(
    medical_history: Optional[List[str]],
) -> str:
    """
    Format medical history.
    """

    if not medical_history:
        return "- Aucun antécédent connu"

    return "\n".join(
        f"- {item.strip()}"
        for item in medical_history
    )


def build_triage_prompt(
    patient_age: int,
    symptoms: List[str],
    medical_history: Optional[List[str]] = None,
    vital_signs: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build final medical triage prompt.
    """

    vital_signs = vital_signs or {}

    vitals_text = "\n".join(
        f"- {key}: {value}"
        for key, value in vital_signs.items()
    )

    if not vitals_text:
        vitals_text = "- Non renseignés"

    prompt = f"""
Analyse ce cas patient.

=== PATIENT ===

Âge :
{patient_age}

=== SYMPTÔMES ===

{format_symptoms(symptoms)}

=== ANTÉCÉDENTS ===

{format_medical_history(medical_history)}

=== SIGNES VITAUX ===

{vitals_text}

=== OBJECTIF ===

Déterminer :
1. la priorité clinique ;
2. une justification courte ;
3. les recommandations adaptées.

Respecte STRICTEMENT le format demandé.
""".strip()

    return prompt
