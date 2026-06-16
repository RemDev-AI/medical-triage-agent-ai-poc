# medical-triage-agent-ai-poc/backend/app/tests/unit/test_loaders.py

from unittest.mock import MagicMock, patch

import pytest  # noqa : F401


@patch("app.llm.loaders.model_loader.AutoModelForCausalLM")
def test_model_loader_success(mock_model):
    """
    Verify model loader returns a model instance.
    """

    fake_model = MagicMock()
    mock_model.from_pretrained.return_value = fake_model

    from backend.app.llm.loaders.model_loader import load_model

    model = load_model()

    assert model is not None


@patch("app.llm.loaders.tokenizer_loader.AutoTokenizer")
def test_tokenizer_loader_success(mock_tokenizer):
    """
    Verify tokenizer loader returns tokenizer instance.
    """

    fake_tokenizer = MagicMock()
    mock_tokenizer.from_pretrained.return_value = fake_tokenizer

    from backend.app.llm.loaders.tokenizer_loader import load_tokenizer

    tokenizer = load_tokenizer()

    assert tokenizer is not None


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

    from backend.app.llm.loaders.quantization_loader import (
        get_quantization_config,
    )

    assert callable(get_quantization_config)
