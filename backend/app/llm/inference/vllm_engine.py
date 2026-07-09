# medical-triage-agent-ai-poc/backend/app/llm/inference/vllm_engine.py

"""
Moteur d'inférence vLLM (in-process).

Introduit à l'étape 3 pour répondre au point 2 du
cahier des charges :

    "Conteneuriser l'application avec Docker et
    l'exposer via une API (FastAPI) en utilisant
    vLLM pour une inférence optimisée."

Approche retenue : intégration en process via
AsyncLLMEngine (pas de serveur HTTP vLLM séparé).
L'adaptateur LoRA (checkpoint DPO final,
RemDev-AI/medical-triage-agent-ai-poc-models) est
chargé directement par vLLM via --enable-lora,
sans fusion préalable (merge_adapter=False),
conformément au format déjà produit à l'étape 2
(adapter_config.json / adapter_model.safetensors).

Ce module expose la même surface fonctionnelle que
generate.py (generate_response) afin de permettre un
dispatch transparent selon runtime_config.use_vllm,
sans modifier triage_engine.py.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from backend.app.deployment.huggingface.hf_space_runtime import (
    runtime_config,
)
from backend.app.llm.inference.generate import (
    clean_response,
    _build_chat_prompt,
)

logger = logging.getLogger(__name__)

_engine_lock = threading.Lock()
_engine_instance = None
_LORA_ADAPTER_NAME = "medical-triage-lora"

# Modèle de base utilisé pour le fine-tuning
# (cf. backend/app/training/sft/sft_config_validation.yaml,
# clé model.base_model). Il s'agit d'un modèle public
# Hugging Face, distinct du dépôt de résultats
# RemDev-AI/medical-triage-agent-ai-poc-models (qui ne
# contient QUE des adaptateurs LoRA, jamais les poids
# complets du modèle de base).
_BASE_MODEL_NAME = "Qwen/Qwen3-1.7B-Base"


def _resolve_adapter_repository() -> str:
    """
    Résout le dépôt Hugging Face contenant
    l'adaptateur LoRA final.

    Correctif : "ref/" est le SEUL adaptateur de
    production (commit "DPO validation run — final
    model", résultat final du pipeline SFT → DPO).
    "sft-final/" est un stade intermédiaire
    (adaptateur SFT pré-DPO) et ne doit jamais être
    utilisé en production.
    """

    base_repo = runtime_config.model_repository

    return f"{base_repo}/ref"


def get_vllm_engine():
    """
    Retourne l'instance singleton d'AsyncLLMEngine,
    en la créant si nécessaire (lazy init, thread-safe).

    Le chargement est volontairement paresseux : il ne
    doit être déclenché qu'au premier appel réel de
    génération, afin de ne jamais impacter le démarrage
    de l'API (health checks, /docs, tests unitaires) ni
    l'exécution de la CI (aucun GPU disponible).
    """

    global _engine_instance

    if _engine_instance is not None:
        return _engine_instance

    with _engine_lock:

        if _engine_instance is not None:
            return _engine_instance

        try:
            from vllm import AsyncEngineArgs
            from vllm import AsyncLLMEngine
            from vllm.lora.request import LoRARequest  # noqa : F401
        except ImportError as exc:
            raise RuntimeError(
                "vLLM n'est pas installé ou "
                "indisponible dans cet environnement. "
                "Vérifiez requirements.txt et la "
                "présence d'un GPU compatible."
            ) from exc

        engine_args = AsyncEngineArgs(
            model=_BASE_MODEL_NAME,
            enable_lora=True,
            max_lora_rank=64,
            dtype="bfloat16",
            max_model_len=(
                runtime_config.max_input_tokens + runtime_config.max_output_tokens
            ),
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
        )

        logger.info(
            "Initializing vLLM AsyncLLMEngine "
            "(base_model=%s, adapter=%s, "
            "enable_lora=True)",
            _BASE_MODEL_NAME,
            _resolve_adapter_repository(),
        )

        _engine_instance = AsyncLLMEngine.from_engine_args(engine_args)

        logger.info("vLLM AsyncLLMEngine initialized.")

        return _engine_instance


async def generate_response_vllm(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    request_id: Optional[str] = None,
) -> str:
    """
    Génère une réponse via vLLM (AsyncLLMEngine),
    en utilisant l'adaptateur LoRA final chargé
    dynamiquement (--enable-lora), sans merge.

    Signature volontairement alignée sur
    generate.generate_response() (mêmes paramètres
    de sampling) afin de permettre un dispatch
    transparent depuis TriageEngine, sans changer
    l'API publique existante.
    """

    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    import uuid

    engine = get_vllm_engine()

    prompt = _build_chat_prompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    lora_request = LoRARequest(
        _LORA_ADAPTER_NAME,
        1,
        _resolve_adapter_repository(),
    )

    resolved_request_id = request_id or str(uuid.uuid4())

    final_output = None

    async for output in engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=resolved_request_id,
        lora_request=lora_request,
    ):
        final_output = output

    if final_output is None or not final_output.outputs:
        raise RuntimeError(
            "vLLM did not return any output for " f"request_id={resolved_request_id}"
        )

    raw_text = final_output.outputs[0].text

    return clean_response(raw_text)


def is_vllm_enabled() -> bool:
    """
    Point de vérité unique pour savoir si le
    moteur vLLM doit être utilisé, piloté par
    la variable d'environnement USE_VLLM
    (cf. hf_space_runtime.py).
    """

    return runtime_config.use_vllm
