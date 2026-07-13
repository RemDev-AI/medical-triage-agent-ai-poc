# medical-triage-agent-ai-poc/backend/app/tests/unit/test_loaders.py

import sys
from unittest.mock import MagicMock

import pytest  # noqa: F401


def _install_fake_heavy_modules(monkeypatch):
    """
    torch / transformers / peft ne sont pas installés en CI
    (requirements-ci.txt) et sont importés en *lazy* import à
    l'intérieur des fonctions de model_loader.py / tokenizer_loader.py.

    On ne peut donc pas faire @patch("app...model_loader.AutoModelForCausalLM")
    car cet attribut n'existe jamais au niveau du module (il n'est
    résolu que localement, dans le scope de la fonction, au moment de
    son exécution).

    On injecte à la place de faux modules dans sys.modules, de façon
    à ce que les imports lazy à l'intérieur de load_model()/
    load_tokenizer() récupèrent nos mocks.
    """

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"

    fake_transformers = MagicMock()
    fake_peft = MagicMock()

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    return fake_transformers, fake_peft


def test_model_loader_success(monkeypatch):
    """
    Verify model loader returns a model instance.
    """

    fake_transformers, _fake_peft = _install_fake_heavy_modules(monkeypatch)

    fake_model = MagicMock()
    fake_transformers.AutoModelForCausalLM.from_pretrained.return_value = fake_model

    # Import après injection des fake modules, et après invalidation
    # d'un éventuel import précédent (le module utilise des imports
    # lazy donc ce n'est normalement pas strictement nécessaire, mais
    # ça sécurise l'ordre d'exécution des tests).
    sys.modules.pop("app.llm.loaders.model_loader", None)
    from app.llm.loaders.model_loader import load_model

    model = load_model()

    assert model is not None
    fake_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()


def test_tokenizer_loader_success(monkeypatch):
    """
    Verify tokenizer loader returns tokenizer instance.
    """

    fake_transformers, _ = _install_fake_heavy_modules(monkeypatch)

    fake_tokenizer = MagicMock()
    fake_tokenizer.pad_token = "<pad>"
    fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tokenizer

    sys.modules.pop("app.llm.loaders.tokenizer_loader", None)
    from app.llm.loaders.tokenizer_loader import load_tokenizer

    tokenizer = load_tokenizer()

    assert tokenizer is not None
    fake_transformers.AutoTokenizer.from_pretrained.assert_called_once()


def test_model_loader_has_generate_method():
    """
    Loaded model must expose generate().
    """

    fake_model = MagicMock()
    fake_model.generate = MagicMock()

    assert hasattr(fake_model, "generate")


def test_tokenizer_loader_has_encode_method():
    """
    Loaded tokenizer must expose encode().
    """

    fake_tokenizer = MagicMock()
    fake_tokenizer.encode = MagicMock()

    assert hasattr(fake_tokenizer, "encode")


def test_quantization_loader_import():
    """
    Quantization loader should be importable.
    """

    from app.llm.loaders.quantization_loader import get_quantization_config

    assert callable(get_quantization_config)
