# medical-triage-agent-ai-poc/backend/app/llm/inference/triage_engine.py

"""
Main medical triage engine.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Dict
from typing import List
from typing import Optional

# from transformers import PreTrainedModel
# from transformers import PreTrainedTokenizerBase
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers import PreTrainedTokenizerBase

from app.llm.inference.generate import (
    build_generation_metadata,
    clean_response,
    generate_response,
)

from app.llm.inference.prompt_builder import (
    SYSTEM_PROMPT,
    build_triage_prompt,
)

from app.deployment.huggingface.hf_space_runtime import (
    runtime_config,
)

# PHASE 7 - Monitoring
from app.monitoring.gpu_monitor import gpu_monitor

logger = logging.getLogger(__name__)


VALID_PRIORITIES = {
    "CRITIQUE",
    "URGENT",
    "MODÉRÉ",
    "FAIBLE",
}

# Seuil de confiance en dessous duquel une revue humaine est
# obligatoire avant toute action clinique sur la base du triage
# proposé. Ajouté suite au cahier des charges CHSA : l'agent est un
# outil d'ASSISTANCE au personnel soignant, pas un décideur autonome
# — un score bas (parsing incomplet/échoué) doit être signalé de
# façon explicite et non-ignorable, jamais juste consigné en silence
# dans un champ numérique noyé dans la réponse.
#
# Choix du seuil : _CONFIDENCE_FULL_MATCH (0.9) est la seule valeur
# correspondant à une extraction complète et sans ambiguïté des 3
# champs (PRIORITÉ/JUSTIFICATION/RECOMMANDATIONS). Tout ce qui est
# strictement inférieur signale qu'au moins un champ a été manqué ou
# deviné/replié par défaut — la revue humaine est donc requise dès
# qu'on n'est pas au score plein, pas seulement en cas d'échec total.
HUMAN_REVIEW_CONFIDENCE_THRESHOLD = 0.9

# Confiance heuristique associée à un parsing réussi
# vs. un repli sur une valeur par défaut (champ non
# détecté dans la sortie du modèle). Ajouté à l'étape 3
# pour renseigner TriageResponse.confidence_score, requis
# par le schéma mais absent du moteur jusqu'ici.
_CONFIDENCE_FULL_MATCH = 0.9
_CONFIDENCE_PARTIAL_MATCH = 0.5


class TriageEngine:
    """
    Clinical triage inference engine.

    Deux modes de fonctionnement (étape 3) :

    - vLLM (runtime_config.use_vllm == True) :
      model/tokenizer ne sont pas requis, la
      génération est déléguée à
      backend.app.llm.inference.vllm_engine.

    - Transformers (historique) : model/tokenizer
      Hugging Face chargés localement, requis.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_name: str = "Qwen3-Medical-Triage",
    ) -> None:

        if not runtime_config.use_vllm and (model is None or tokenizer is None):
            raise ValueError(
                "model and tokenizer are required "
                "when runtime_config.use_vllm is "
                "False."
            )

        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name

    async def run_triage(
        self,
        patient_age: int,
        symptoms: List[str],
        medical_history: Optional[List[str]] = None,
        vital_signs: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """
        Execute medical triage inference.
        """

        start_time = time.time()

        try:

            prompt = build_triage_prompt(
                patient_age=patient_age,
                symptoms=symptoms,
                medical_history=medical_history,
                vital_signs=vital_signs,
            )

            raw_response = await generate_response(
                model=self.model,
                tokenizer=self.tokenizer,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
            )

            raw_response = clean_response(
                raw_response,
            )

            parsed_response = self.parse_response(
                raw_response,
            )

            latency_seconds = time.time() - start_time

            metadata = build_generation_metadata(
                latency_seconds=latency_seconds,
                model_name=self.model_name,
            )

            # ----------------------------------
            # PHASE 7 Monitoring
            # ----------------------------------

            gpu_monitor.increment_request()

            return {
                "triage": parsed_response,
                "metadata": metadata,
                "raw_response": raw_response,
            }

        except Exception:

            logger.exception("Triage inference failed.")

            raise

    def parse_response(
        self,
        response: str,
    ) -> Dict:
        """
        Parse structured LLM response.
        """

        priority = self.extract_field(
            response,
            "PRIORITÉ",
        )

        justification = self.extract_field(
            response,
            "JUSTIFICATION",
        )

        recommendations = self.extract_field(
            response,
            "RECOMMANDATIONS",
        )

        priority, priority_matched = self.normalize_priority(
            priority,
        )

        justification_matched = justification != "Non disponible"

        recommendations_matched = recommendations != "Non disponible"

        confidence_score = self._compute_confidence(
            priority_matched,
            justification_matched,
            recommendations_matched,
        )

        # Revue humaine obligatoire dès que le score de confiance
        # n'est pas au maximum, OU — garde-fou indépendant, qui ne
        # dépend pas du calcul de ratio — dès que la priorité
        # elle-même n'a pas pu être reconnue avec certitude. On ne
        # veut pas qu'un futur changement dans _compute_confidence
        # puisse, par effet de bord, faire passer un cas où la
        # priorité a été deviné/repliée en CRITIQUE (fail-safe) sous
        # le radar de la revue humaine.
        requires_human_review = (
            confidence_score < HUMAN_REVIEW_CONFIDENCE_THRESHOLD or not priority_matched
        )

        if requires_human_review:
            logger.warning(
                "Triage nécessite une revue humaine (confidence_score=%s, "
                "priority_matched=%s) — priority=%s",
                confidence_score,
                priority_matched,
                priority,
            )

        return {
            "priority": priority,
            "justification": justification,
            "recommendations": recommendations,
            "confidence_score": confidence_score,
            "requires_human_review": requires_human_review,
        }

    @staticmethod
    def _compute_confidence(
        *field_matches: bool,
    ) -> float:
        """
        Score de confiance heuristique basé sur le
        nombre de champs correctement extraits du
        format structuré attendu (PRIORITÉ /
        JUSTIFICATION / RECOMMANDATIONS).

        Introduit à l'étape 3 pour satisfaire
        TriageResponse.confidence_score, requis par
        le schéma Pydantic mais jusqu'ici toujours
        renvoyé à 0.0 par défaut.
        """

        if not field_matches:
            return 0.0

        ratio = sum(1 for m in field_matches if m) / len(field_matches)

        if ratio == 1.0:
            return _CONFIDENCE_FULL_MATCH

        if ratio == 0.0:
            return 0.0

        return round(
            _CONFIDENCE_PARTIAL_MATCH * ratio,
            2,
        )

    @staticmethod
    def extract_field(
        text: str,
        field_name: str,
    ) -> str:
        """
        Extract section field from model output.

        Rendu tolérant (2026-07-21) après un cas observé en
        production où le modèle a produit "RECOMMANDATION:" au
        singulier (au lieu de "RECOMMANDATIONS:" comme demandé dans
        SYSTEM_PROMPT) noyé dans le texte de JUSTIFICATION, faisant
        échouer l'extraction du champ recommandations ET, plus
        grave, celle de PRIORITÉ pour la même réponse — un modèle de
        1.7B post-DPO ne respecte pas toujours le format à la
        lettre, le parsing doit encaisser ces variations plutôt que
        de basculer silencieusement en confidence basse à chaque
        petit écart de forme.

        Tolérances ajoutées :
        - insensible à la casse (le modèle peut varier majuscules/
          minuscules) ;
        - accepte le singulier ET le pluriel pour un nom de champ se
          terminant par "S" (ex: "RECOMMANDATIONS" matche aussi
          "RECOMMANDATION") ;
        - le lookahead de fin de section ne s'arrête plus sur
          N'IMPORTE QUEL mot capitalisé suivi de ":" (risque de
          coupure prématurée si la justification elle-même contient
          un mot capitalisé suivi de ":"), mais uniquement sur l'un
          des en-têtes de section réellement attendus.
        """

        # Tolère le pluriel optionnel (RECOMMANDATIONS -> RECOMMANDATION(S)?)
        if field_name.endswith("S"):
            field_pattern = re.escape(field_name[:-1]) + "S?"
        else:
            field_pattern = re.escape(field_name)

        known_headers = ("PRIORITÉ", "JUSTIFICATION", "RECOMMANDATIONS?")
        lookahead_headers = "|".join(known_headers)

        pattern = (
            rf"{field_pattern}\s*:\s*(.*?)" rf"(?=\n\s*(?:{lookahead_headers})\s*:|\Z)"
        )

        match = re.search(
            pattern,
            text,
            re.DOTALL | re.IGNORECASE,
        )

        if not match:
            return "Non disponible"

        return match.group(1).strip()

    @staticmethod
    def normalize_priority(
        priority: str,
    ) -> tuple[str, bool]:
        """
        Normalize medical priority.

        Returns
        -------
        tuple[str, bool]
            La priorité normalisée, et un booléen
            indiquant si la valeur brute était déjà
            valide (utilisé pour le calcul du
            confidence_score).
        """

        priority = priority.upper().strip()

        if priority not in VALID_PRIORITIES:

            logger.warning(
                "Unknown priority detected: %s — escalating to CRITIQUE "
                "as a fail-safe (a parsing failure must never silently "
                "downgrade clinical priority).",
                priority,
            )

            # SÉCURITÉ CLINIQUE : en cas d'échec de reconnaissance du
            # champ PRIORITÉ (sortie du modèle mal formée, hors
            # format attendu, etc.), on ne doit JAMAIS retomber sur
            # une valeur médiane comme "MODÉRÉ" — cela équivaudrait à
            # sous-prioriser silencieusement un cas potentiellement
            # critique simplement parce que le parsing a échoué.
            # Le principe de précaution en triage médical impose
            # d'escalader vers le niveau le plus prudent (CRITIQUE)
            # chaque fois qu'on ne peut pas établir la priorité avec
            # certitude, et de laisser le confidence_score bas
            # signaler qu'une revue humaine est nécessaire.
            return "CRITIQUE", False

        return priority, True
