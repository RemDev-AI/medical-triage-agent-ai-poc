# medical-triage-agent-ai-poc/backend/app/api/dependencies/inference.py

"""
Dépendances FastAPI pour l'inférence locale.

Remplace l'ancien InferenceClient (appel HTTP externe) par un accès
direct au moteur d'inférence local, chargé paresseusement (lazy,
thread-safe) au premier appel réel — même principe que
app.llm.inference.vllm_engine.get_vllm_engine().

Pourquoi paresseux et non chargé dans app.main::lifespan :
le chargement du modèle Transformers (téléchargement d'adaptateur +
poids) est coûteux et nécessite potentiellement un GPU. Le charger au
démarrage bloquerait aussi bien les health checks que les tests
(TestClient(app) exécute le lifespan à chaque test). En le rendant
paresseux, les tests peuvent simplement surcharger
get_triage_engine / get_generation_context via
app.dependency_overrides, exactement comme avec l'ancien
InferenceClient, sans jamais déclencher de chargement réel.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase

from app.llm.inference.triage_engine import TriageEngine
from app.deployment.huggingface.hf_space_runtime import runtime_config

_engine_lock = threading.Lock()
_triage_engine_instance: Optional[TriageEngine] = None


def _build_triage_engine() -> TriageEngine:
    """
    Construit (une seule fois) l'instance de TriageEngine.

    - runtime_config.use_vllm == True : model/tokenizer restent None,
      la génération est déléguée à vllm_engine (lui-même paresseux).
    - runtime_config.use_vllm == False : charge le modèle Transformers
      + l'adaptateur LoRA (même résolution que vllm_engine, policy
      post-DPO) une seule fois, réutilisé pour tous les appels
      suivants.
    """

    global _triage_engine_instance

    if _triage_engine_instance is not None:
        return _triage_engine_instance

    with _engine_lock:

        if _triage_engine_instance is not None:
            return _triage_engine_instance

        model = None
        tokenizer = None

        if not runtime_config.use_vllm:

            from transformers import AutoTokenizer

            from app.llm.loaders.model_loader import load_model
            from app.llm.inference.vllm_engine import (
                _BASE_MODEL_NAME,
                _ensure_adapter_downloaded,
            )

            adapter_local_path = _ensure_adapter_downloaded()

            model = load_model(
                base_model_name=_BASE_MODEL_NAME,
                adapter_path=adapter_local_path,
                load_in_4bit=runtime_config.load_in_4bit,
                load_in_8bit=runtime_config.load_in_8bit,
            )

            tokenizer = AutoTokenizer.from_pretrained(adapter_local_path)

        _triage_engine_instance = TriageEngine(
            model=model,
            tokenizer=tokenizer,
        )

        return _triage_engine_instance


def get_triage_engine() -> TriageEngine:
    """
    Factory FastAPI dependency, utilisée par routes/triage.py.

    En tests, surcharger via :
        app.dependency_overrides[get_triage_engine] = lambda: FakeEngine()
    pour éviter tout chargement réel.
    """

    return _build_triage_engine()


def get_generation_context() -> (
    Tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizerBase]]
):
    """
    Factory FastAPI dependency, utilisée par routes/inference.py.

    Retourne le couple (model, tokenizer) du TriageEngine partagé.
    Les deux valent None lorsque runtime_config.use_vllm est True.
    """

    engine = _build_triage_engine()

    return engine.model, engine.tokenizer
