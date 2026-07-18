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

        return {
            "priority": priority,
            "justification": justification,
            "recommendations": recommendations,
            "confidence_score": confidence_score,
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
        """

        pattern = rf"{field_name}\s*:\s*(.*?)" rf"(?=\n[A-ZÉ]+:|\Z)"

        match = re.search(
            pattern,
            text,
            re.DOTALL,
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
                "Unknown priority detected: %s",
                priority,
            )

            return "MODÉRÉ", False

        return priority, True
