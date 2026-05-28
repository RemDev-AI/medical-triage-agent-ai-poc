# medical-triage-agent-ai-poc/backend/app/llm/loaders/model_loader.py

"""
Main model loader for inference.

Features:
- Qwen loading
- PEFT LoRA loading
- 4-bit / 8-bit quantization
- bf16 support
- automatic CUDA mapping
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
)

from loaders.quantization_loader import (
    build_quantization_config,
)

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Centralized inference model loader.
    """

    def __init__(
        self,
        base_model_name: str,
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        compute_dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.compute_dtype = compute_dtype
        self.device_map = device_map

    def load_model(self):
        """
        Load base model and optionally inject LoRA adapter.
        """

        logger.info("Starting model loading...")
        start_time = time.time()

        quantization_config = build_quantization_config(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            compute_dtype=self.compute_dtype,
        )

        dtype_mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        torch_dtype = dtype_mapping.get(
            self.compute_dtype,
            torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        logger.info("Base model loaded successfully.")

        if self.adapter_path:
            adapter_dir = Path(self.adapter_path)

            if not adapter_dir.exists():
                raise FileNotFoundError(
                    f"LoRA adapter not found: {adapter_dir}"
                )

            logger.info("Loading PEFT adapter...")

            model = PeftModel.from_pretrained(
                model,
                self.adapter_path,
            )

            logger.info("PEFT adapter loaded successfully.")

        model.eval()

        elapsed = round(time.time() - start_time, 2)

        logger.info(
            "Model fully loaded in %s seconds.",
            elapsed,
        )

        return model

    @staticmethod
    def print_gpu_memory():
        """
        Print GPU memory statistics.
        """

        if not torch.cuda.is_available():
            logger.warning("CUDA unavailable.")
            return

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        logger.info(
            "GPU memory allocated: %.2f GB",
            allocated,
        )

        logger.info(
            "GPU memory reserved: %.2f GB",
            reserved,
        )
