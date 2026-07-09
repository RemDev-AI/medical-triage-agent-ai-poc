# medical-triage-agent-ai-poc/backend/app/llm/loaders/quantization_loader.py

"""
Quantization loader utilities.

Support:
- 4-bit NF4 quantization
- 8-bit quantization
- bfloat16
- CUDA optimization
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import BitsAndBytesConfig


_ALLOWED_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_ALLOWED_QUANT_TYPES = {
    "nf4",
    "fp4",
}


def _get_torch_dtype(
    compute_dtype: str,
) -> torch.dtype:
    """
    Convert string dtype into torch dtype.
    """

    if compute_dtype not in _ALLOWED_DTYPES:
        raise ValueError(f"Unsupported compute dtype: {compute_dtype}")

    return _ALLOWED_DTYPES[compute_dtype]


def build_quantization_config(
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    use_double_quant: bool = True,
    compute_dtype: str = "bfloat16",
) -> Optional[BitsAndBytesConfig]:
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

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=torch_dtype,
    )
