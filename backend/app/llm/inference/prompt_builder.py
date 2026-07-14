# medical-triage-agent-ai-poc/backend/app/llm/inference/prompt_builder.py

"""
Prompt builder for medical triage inference.
"""

from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional

SYSTEM_PROMPT = """
Tu es un assistant IA spécialisé en triage médical.

Mission :
Analyser les informations patient fournies et attribuer un niveau
de priorité clinique adapté.

Contraintes importantes :
- Ne jamais produire de diagnostic médical définitif.
- Ne jamais inventer de symptômes ou d'informations absentes.
- Utiliser uniquement les données fournies.
- Rester prudent en cas d'informations incomplètes.
- Privilégier la sécurité du patient.
- Signaler immédiatement toute urgence potentielle.

Niveaux de priorité autorisés :

CRITIQUE
URGENT
MODÉRÉ
FAIBLE

Tu dois toujours répondre EXACTEMENT avec le format suivant :

PRIORITÉ:
<CRITIQUE|URGENT|MODÉRÉ|FAIBLE>

JUSTIFICATION:
<justification clinique concise>

RECOMMANDATIONS:
<recommandations prudentes et adaptées>

Ne jamais ajouter d'autres sections.
Ne jamais ajouter d'introduction.
Ne jamais ajouter de conclusion.
""".strip()


def format_symptoms(
    symptoms: List[str],
) -> str:
    """
    Format symptoms list.
    """

    if not symptoms:
        return "- Aucun symptôme renseigné"

    return "\n".join(f"- {symptom.strip()}" for symptom in symptoms)


def format_medical_history(
    medical_history: Optional[List[str]],
) -> str:
    """
    Format medical history.
    """

    if not medical_history:
        return "- Aucun antécédent connu"

    return "\n".join(f"- {item.strip()}" for item in medical_history)


def format_vital_signs(
    vital_signs: Optional[Dict[str, str]],
) -> str:
    """
    Format vital signs.
    """

    if not vital_signs:
        return "- Non renseignés"

    return "\n".join(f"- {key}: {value}" for key, value in vital_signs.items())


def build_triage_prompt(
    patient_age: int,
    symptoms: List[str],
    medical_history: Optional[List[str]] = None,
    vital_signs: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build final medical triage prompt.
    """

    prompt = f"""
Analyse le cas clinique suivant.

=== PATIENT ===

ÂGE :
{patient_age}

=== SYMPTÔMES ===

{format_symptoms(symptoms)}

=== ANTÉCÉDENTS ===

{format_medical_history(medical_history)}

=== SIGNES VITAUX ===

{format_vital_signs(vital_signs)}

=== TÂCHE ===

Déterminer :

1. Le niveau de priorité clinique.
2. Une justification clinique concise.
3. Les recommandations adaptées.

Rappel :

- Utiliser uniquement les informations fournies.
- Ne jamais inventer de données.
- Ne jamais poser de diagnostic définitif.
- Respecter STRICTEMENT le format demandé.
""".strip()

    return prompt
