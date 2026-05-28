# medical-triage-agent-ai-poc/backend/app/llm/inference/generate.py

"""
Inference generation utilities.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


def generate_response(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Generate medical inference response.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }

    logger.info("Generating inference response...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    logger.info("Inference generation completed.")

    return response.strip()


def clean_response(
    response: str,
) -> str:
    """
    Clean malformed outputs.
    """

    response = response.replace("<|im_end|>", "")
    response = response.replace("<|endoftext|>", "")

    return response.strip()


def build_generation_metadata(
    latency_seconds: float,
    model_name: str,
) -> Dict:
    """
    Build inference metadata.
    """

    return {
        "latency_seconds": round(latency_seconds, 2),
        "model_name": model_name,
    }
