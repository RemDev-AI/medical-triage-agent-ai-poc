# medical-triage-agent-ai-poc/backend/app/llm/loaders/quantization_loader.py

"""
Quantization loader utilities.

Support:
- 4-bit NF4 quantization
- 8-bit quantization
- bfloat16
- CUDA optimization

Note: torch et transformers sont importés en lazy (à l'intérieur des
fonctions) et non en tête de module. Ces packages sont volontairement
absents de requirements-ci.txt (cf. commentaire dans ce fichier) pour
éviter le "No space left on device" sur les runners GitHub Actions ;
l'inférence réelle est mockée en CI. Ne pas remonter ces imports au
niveau module, sous peine de faire échouer l'import du module entier
en CI (ModuleNotFoundError: No module named 'torch').
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch
    from transformers import BitsAndBytesConfig


_ALLOWED_QUANT_TYPES = {
    "nf4",
    "fp4",
}


def _get_allowed_dtypes() -> dict:
    """
    Build the string -> torch.dtype mapping (lazy, needs torch).
    """

    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }


def _get_torch_dtype(
    compute_dtype: str,
) -> "torch.dtype":
    """
    Convert string dtype into torch dtype.
    """

    allowed_dtypes = _get_allowed_dtypes()

    if compute_dtype not in allowed_dtypes:
        raise ValueError(f"Unsupported compute dtype: {compute_dtype}")

    return allowed_dtypes[compute_dtype]


def build_quantization_config(
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    use_double_quant: bool = True,
    compute_dtype: str = "bfloat16",
) -> "Optional[BitsAndBytesConfig]":
    """
    Build bitsandbytes quantization config.

    Args:
        load_in_4bit:
            Enable 4-bit quantization.

        load_in_8bit:
            Enable 8-bit quantization.

        bnb_4bit_quant_type:
            Quantization strategy.

        use_double_quant:
            Enable nested quantization.

        compute_dtype:
            Compute precision.

    Returns:
        BitsAndBytesConfig or None.
    """

    if load_in_4bit and load_in_8bit:
        raise ValueError(
            "4-bit and 8-bit quantization " "cannot be enabled simultaneously."
        )

    if not load_in_4bit and not load_in_8bit:
        return None

    if bnb_4bit_quant_type not in _ALLOWED_QUANT_TYPES:
        raise ValueError(f"Unsupported quantization type: " f"{bnb_4bit_quant_type}")

    torch_dtype = _get_torch_dtype(compute_dtype)

    # Lazy import: transformers n'est chargé qu'au moment réel de la
    # construction de la config, pas à l'import du module.
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=torch_dtype,
    )


# ==========================================================
# ALIAS (API attendue par les tests / consommateurs externes)
# ==========================================================
#
# `get_quantization_config` est un alias de `build_quantization_config`.
# Conservé séparément (plutôt qu'un renommage) pour ne pas casser
# l'import existant dans model_loader.py :
#
#     from app.llm.loaders.quantization_loader import (
#         build_quantization_config,
#     )

get_quantization_config = build_quantization_config
