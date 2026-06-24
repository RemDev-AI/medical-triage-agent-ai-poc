# medical-triage-agent-ai-poc/backend/app/training/evaluation/clinical_metrics.py

"""
Clinical evaluation metrics for medical triage models.

Metrics:
- priority_accuracy
- clinical_accuracy
- recommendation_accuracy
- safety_accuracy

Compatible with:
- SFT evaluation
- DPO evaluation
- Clinical benchmark datasets
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List


def _safe_divide(numerator: int, denominator: int) -> float:
    """
    Safe division helper.

    Returns:
        float in [0,1]
    """
    if denominator == 0:
        return 0.0

    return numerator / denominator


def priority_accuracy(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute triage priority accuracy.

    Example priorities:
        - low
        - medium
        - high
        - critical

    Args:
        predictions:
            Model predicted priorities

        references:
            Ground truth priorities

    Returns:
        Accuracy score in [0,1]
    """

    if len(predictions) != len(references):
        raise ValueError(
            "predictions and references must have same length."
        )

    correct = sum(
        pred == ref
        for pred, ref in zip(predictions, references)
    )

    return _safe_divide(correct, len(references))


def clinical_accuracy(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute clinical decision accuracy.

    Example:
        diagnosis class
        urgency class
        routing decision

    Args:
        predictions:
            Predicted labels

        references:
            Ground truth labels

    Returns:
        Accuracy score in [0,1]
    """

    if len(predictions) != len(references):
        raise ValueError(
            "predictions and references must have same length."
        )

    correct = sum(
        pred == ref
        for pred, ref in zip(predictions, references)
    )

    return _safe_divide(correct, len(references))


def recommendation_accuracy(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute recommendation accuracy.

    Example:
        - call emergency services
        - consult physician
        - home care
        - immediate ER visit

    Args:
        predictions:
            Predicted recommendations

        references:
            Ground truth recommendations

    Returns:
        Accuracy score in [0,1]
    """

    if len(predictions) != len(references):
        raise ValueError(
            "predictions and references must have same length."
        )

    correct = sum(
        pred == ref
        for pred, ref in zip(predictions, references)
    )

    return _safe_divide(correct, len(references))


def safety_accuracy(
    safe_predictions: List[bool],
) -> float:
    """
    Compute safety accuracy.

    Assumes:
        True  -> safe response
        False -> unsafe response

    Args:
        safe_predictions:
            List of safety decisions.

    Returns:
        Safety accuracy score in [0,1]
    """

    if not safe_predictions:
        return 0.0

    safe_count = sum(safe_predictions)

    return _safe_divide(
        safe_count,
        len(safe_predictions),
    )


def compute_clinical_metrics(
    priority_predictions: List[str],
    priority_references: List[str],
    clinical_predictions: List[str],
    clinical_references: List[str],
    recommendation_predictions: List[str],
    recommendation_references: List[str],
    safe_predictions: List[bool],
) -> Dict[str, Any]:
    """
    Compute all clinical metrics.

    Returns:
        {
            "priority_accuracy": ...,
            "clinical_accuracy": ...,
            "recommendation_accuracy": ...,
            "safety_accuracy": ...
        }
    """

    return {
        "priority_accuracy": priority_accuracy(
            priority_predictions,
            priority_references,
        ),
        "clinical_accuracy": clinical_accuracy(
            clinical_predictions,
            clinical_references,
        ),
        "recommendation_accuracy": recommendation_accuracy(
            recommendation_predictions,
            recommendation_references,
        ),
        "safety_accuracy": safety_accuracy(
            safe_predictions,
        ),
    }
