# medical-triage-agent-ai-poc/backend/app/llm/inference/triage_engine.py

"""
Main medical triage engine.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional

from app.llm.inference.generate import (
    build_generation_metadata,
    clean_response,
    generate_response,
)

from app.llm.inference.prompt_builder import (
    SYSTEM_PROMPT,
    build_triage_prompt,
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


class TriageEngine:
    """
    Clinical triage inference engine.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name

    def run_triage(
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

            raw_response = generate_response(
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

            latency_seconds = (
                time.time() - start_time
            )

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

        except Exception as exc:

            logger.exception(
                "Triage inference failed"
            )

            raise exc

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

        priority = self.normalize_priority(
            priority,
        )

        return {
            "priority": priority,
            "justification": justification,
            "recommendations": recommendations,
        }

    @staticmethod
    def extract_field(
        text: str,
        field_name: str,
    ) -> str:
        """
        Extract section field.
        """

        pattern = (
            rf"{field_name}\s*:\s*(.*?)"
            rf"(?=\n[A-ZÉ]+:|\Z)"
        )

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
    ) -> str:
        """
        Normalize medical priority.
        """

        priority = priority.upper().strip()

        if priority not in VALID_PRIORITIES:

            logger.warning(
                "Unknown priority detected: %s",
                priority,
            )

            return "MODÉRÉ"

        return priority
