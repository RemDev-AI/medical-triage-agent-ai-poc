# medical-triage-agent-ai-poc/backend/app/training/evaluation/safety_evaluator.py

"""
Clinical safety evaluation module.

Responsibilities:
- Fusion:
    - hallucinations
    - dangerous recommendations
- Compute global safety score
- Validate clinical safety thresholds
- Produce a structured safety report

Compatible with:
- SFT evaluation
- DPO evaluation
- Google Colab
- W&B logging
- HF Hub model evaluation
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List

from backend.app.training.evaluation.clinical_thresholds import (
    MAX_DANGEROUS_RATE,
    MAX_HALLUCINATION_RATE,
    MIN_SAFETY_SCORE,
)
from backend.app.training.evaluation.dangerous_recommendation_detector import (  # noqa : F401
    dangerous_recommendation_rate,
    evaluate_responses as evaluate_dangerous_responses,
)
from backend.app.training.evaluation.hallucination_detector import (  # noqa : F401
    evaluate_responses as evaluate_hallucination_responses,
    hallucination_rate,
    unsafe_claim_rate,
)


# ============================================================
# HELPERS
# ============================================================

def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Clamp a value into [0,1].
    """

    return max(minimum, min(maximum, value))


# ============================================================
# SAFETY SCORE
# ============================================================

def compute_safety_score(
    hallucination_rate_value: float,
    dangerous_rate_value: float,
) -> float:
    """
    Compute a global safety score.

    Formula:

        safety_score =
            1 - average(
                hallucination_rate,
                dangerous_rate
            )

    Examples:

        hallucination_rate = 0.04
        dangerous_rate     = 0.01

        safety_score = 0.975

    Returns:
        Float in [0,1]
    """

    risk_score = (
        hallucination_rate_value
        + dangerous_rate_value
    ) / 2.0

    return _clamp(1.0 - risk_score)


# ============================================================
# THRESHOLD VALIDATION
# ============================================================

def safety_thresholds_passed(
    hallucination_rate_value: float,
    dangerous_rate_value: float,
    safety_score: float,
) -> bool:
    """
    Validate all safety thresholds.
    """

    return all(
        [
            hallucination_rate_value
            <= MAX_HALLUCINATION_RATE,
            dangerous_rate_value
            <= MAX_DANGEROUS_RATE,
            safety_score
            >= MIN_SAFETY_SCORE,
        ]
    )


# ============================================================
# CORE EVALUATION
# ============================================================

def evaluate_safety(
    responses: List[str],
) -> Dict[str, Any]:
    """
    Evaluate model safety on a dataset.

    Returns:

        {
            "hallucination_rate": float,
            "unsafe_claim_rate": float,
            "dangerous_rate": float,
            "safety_score": float,
            "thresholds_passed": bool
        }
    """

    hallucination_metrics = (
        evaluate_hallucination_responses(
            responses
        )
    )

    dangerous_metrics = (
        evaluate_dangerous_responses(
            responses
        )
    )

    hallucination_rate_value = (
        hallucination_metrics["hallucination_rate"]
    )

    unsafe_claim_rate_value = (
        hallucination_metrics["unsafe_claim_rate"]
    )

    dangerous_rate_value = (
        dangerous_metrics["dangerous_rate"]
    )

    safety_score = compute_safety_score(
        hallucination_rate_value=(
            hallucination_rate_value
        ),
        dangerous_rate_value=(
            dangerous_rate_value
        ),
    )

    thresholds_passed = (
        safety_thresholds_passed(
            hallucination_rate_value=(
                hallucination_rate_value
            ),
            dangerous_rate_value=(
                dangerous_rate_value
            ),
            safety_score=safety_score,
        )
    )

    return {
        "hallucination_rate":
            hallucination_rate_value,
        "unsafe_claim_rate":
            unsafe_claim_rate_value,
        "dangerous_rate":
            dangerous_rate_value,
        "safety_score":
            safety_score,
        "thresholds_passed":
            thresholds_passed,
    }


# ============================================================
# EXTENDED REPORT
# ============================================================

def evaluate_safety_detailed(
    responses: List[str],
) -> Dict[str, Any]:
    """
    Detailed safety evaluation.

    Includes all intermediate metrics
    for reporting and experiment tracking.
    """

    hallucination_metrics = (
        evaluate_hallucination_responses(
            responses
        )
    )

    dangerous_metrics = (
        evaluate_dangerous_responses(
            responses
        )
    )

    hallucination_rate_value = (
        hallucination_metrics["hallucination_rate"]
    )

    dangerous_rate_value = (
        dangerous_metrics["dangerous_rate"]
    )

    safety_score = compute_safety_score(
        hallucination_rate_value,
        dangerous_rate_value,
    )

    return {
        "hallucination": hallucination_metrics,
        "dangerous_recommendations":
            dangerous_metrics,
        "safety_score":
            safety_score,
        "thresholds": {
            "max_hallucination_rate":
                MAX_HALLUCINATION_RATE,
            "max_dangerous_rate":
                MAX_DANGEROUS_RATE,
            "min_safety_score":
                MIN_SAFETY_SCORE,
        },
        "thresholds_passed":
            safety_thresholds_passed(
                hallucination_rate_value,
                dangerous_rate_value,
                safety_score,
            ),
    }


# ============================================================
# CONVENIENCE API
# ============================================================

def is_model_safe(
    responses: List[str],
) -> bool:
    """
    Returns True if the model passes
    all safety checks.
    """

    result = evaluate_safety(
        responses
    )

    return bool(
        result["thresholds_passed"]
    )


def get_safety_score(
    responses: List[str],
) -> float:
    """
    Convenience helper returning
    only the safety score.
    """

    result = evaluate_safety(
        responses
    )

    return float(
        result["safety_score"]
    )
