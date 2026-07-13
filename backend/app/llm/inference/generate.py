# medical-triage-agent-ai-poc/backend/app/llm/inference/generate.py

"""
Inference generation utilities.

Responsibilities:
- Build chat prompts
- Run model.generate() (Transformers) OR dispatch to
  vLLM AsyncLLMEngine (étape 3), selon
  runtime_config.use_vllm
- Decode outputs
- Clean malformed responses
- Build inference metadata
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase

from app.deployment.huggingface.hf_space_runtime import (
    runtime_config,
)

logger = logging.getLogger(__name__)


async def generate_response(
    model: Optional[PreTrainedModel],
    tokenizer: Optional[PreTrainedTokenizerBase],
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Generate a response.

    Deux chemins d'exécution possibles (correctif
    étape 3) :

    - runtime_config.use_vllm == True :
      dispatch vers vLLM AsyncLLMEngine
      (backend.app.llm.inference.vllm_engine).
      `model` et `tokenizer` peuvent alors être
      None : ils ne sont pas requis par ce chemin.

    - runtime_config.use_vllm == False :
      chemin historique Transformers
      (model.generate()), model/tokenizer requis.

    Parameters
    ----------
    model:
        Loaded Hugging Face model (chemin
        Transformers uniquement).

    tokenizer:
        Loaded Hugging Face tokenizer (chemin
        Transformers uniquement).

    system_prompt:
        System instruction.

    user_prompt:
        User input.

    max_new_tokens, temperature, top_p,
    repetition_penalty:
        Paramètres de sampling communs aux deux
        moteurs.

    Returns
    -------
    str
        Generated response.
    """

    if runtime_config.use_vllm:

        logger.info("Dispatching generation to vLLM " "AsyncLLMEngine.")

        from app.llm.inference.vllm_engine import (
            generate_response_vllm,
        )

        return await generate_response_vllm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    if model is None or tokenizer is None:
        raise ValueError(
            "model and tokenizer are required " "when runtime_config.use_vllm is False."
        )

    logger.info("Starting inference generation " "(Transformers backend).")

    prompt = _build_chat_prompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )

    device = next(model.parameters()).device

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output_ids[
        0,
        inputs["input_ids"].shape[1] :,
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    response = clean_response(
        response=response,
    )

    logger.info("Inference generation completed " "(Transformers backend).")

    return response


def _build_chat_prompt(
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Build Qwen-compatible chat prompt.
    """

    return (
        f"<|im_start|>system\n"
        f"{system_prompt}\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_prompt}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def clean_response(
    response: str,
) -> str:
    """
    Remove malformed tokens and artifacts.
    """

    artifacts = [
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
        "</s>",
        "<s>",
    ]

    for artifact in artifacts:
        response = response.replace(
            artifact,
            "",
        )

    return response.strip()


def build_generation_metadata(
    latency_seconds: float,
    model_name: str,
) -> Dict[str, Any]:
    """
    Build inference metadata.
    """

    return {
        "latency_seconds": round(
            latency_seconds,
            2,
        ),
        "model_name": model_name,
    }
