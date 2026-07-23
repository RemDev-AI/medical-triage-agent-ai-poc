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

# import torch
# from transformers import PreTrainedModel
# from transformers import PreTrainedTokenizerBase
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

    import torch  # lazy import — uniquement nécessaire sur ce chemin (non-vLLM)

    prompt = build_chat_prompt_with_tokenizer(
        tokenizer=tokenizer,
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

    ATTENTION (2026-07-22) : ceci est un gabarit ChatML FAIT MAIN,
    utilisé uniquement en dernier recours (cf.
    build_chat_prompt_with_tokenizer ci-dessous) si le tokenizer réel
    du modèle n'expose aucun chat_template. Un modèle publié avec son
    propre chat_template.jinja (comme _MERGED_MODEL_NAME) peut avoir
    été fine-tuné sur un format sensiblement différent (rôles,
    marqueurs spéciaux, sections système/outils) — imposer ce
    gabarit générique à un tel modèle désynchronise le prompt du
    format d'entraînement réel, et peut faire "glisser" la génération
    vers des patterns appris hors-sujet une fois la réponse attendue
    épuisée (observé en prod : dérive vers un faux dialogue
    ".debugLineAssistant"/".debugLineUser" évoquant une requête SQL
    sur l'historique patient, non filtrée par clean_response).
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


def build_chat_prompt_with_tokenizer(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Construit le prompt via le VRAI chat_template du tokenizer
    (tokenizer.apply_chat_template), plutôt que le gabarit ChatML
    fait main de _build_chat_prompt.

    Introduit suite à un bug observé en prod : le gabarit fait main
    ne correspondait pas au chat_template.jinja réellement publié
    avec le modèle fusionné (4.12 Ko — donc pas un simple ChatML),
    ce qui pouvait faire dériver la génération vers des patterns de
    formatage hors-sujet appris à l'entraînement (marqueurs de type
    outil/debug), une fois la réponse attendue terminée.

    Repli explicite (avec warning loggé, jamais silencieux) sur
    _build_chat_prompt si :
    - tokenizer est None (ex: appelant qui n'a pas encore de
      tokenizer chargé) ;
    - le tokenizer n'expose aucun chat_template ;
    - apply_chat_template lève une exception pour toute autre
      raison (template malformé, etc.).
    """

    if tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
        logger.warning(
            "Aucun chat_template disponible sur le tokenizer : repli "
            "sur le gabarit ChatML fait main (_build_chat_prompt). Le "
            "format de prompt peut ne pas correspondre exactement à "
            "celui utilisé lors du fine-tuning du modèle."
        )
        return _build_chat_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        logger.exception(
            "apply_chat_template a échoué — repli sur le gabarit "
            "ChatML fait main (_build_chat_prompt). A investiguer : "
            "ceci ne devrait normalement pas arriver si le "
            "chat_template.jinja publié avec le modèle est valide."
        )
        return _build_chat_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
