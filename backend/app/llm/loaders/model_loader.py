# medical-triage-agent-ai-poc/backend/app/llm/loaders/model_loader.py

"""
Main model loader for inference.

Features:
- Qwen loading
- PEFT LoRA loading
- 4-bit / 8-bit quantization
- bf16 support
- automatic CUDA mapping
- Hugging Face Hub adapter loading
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from backend.app.llm.loaders.quantization_loader import (
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
        merge_adapter: bool = False,
    ):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.compute_dtype = compute_dtype
        self.device_map = device_map
        self.merge_adapter = merge_adapter

    def load_model(
        self,
    ) -> AutoModelForCausalLM:
        """
        Load base model and optionally inject LoRA adapter.
        """

        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot enable both 4-bit and 8-bit quantization.")

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

        effective_device_map = self.device_map

        if not torch.cuda.is_available():
            effective_device_map = "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map=effective_device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        logger.info("Base model loaded successfully.")

        if self.adapter_path:

            adapter_is_local = Path(self.adapter_path).exists()

            if adapter_is_local:
                logger.info(
                    "Loading local PEFT adapter: %s",
                    self.adapter_path,
                )
            else:
                logger.info(
                    "Loading Hugging Face Hub adapter: %s",
                    self.adapter_path,
                )

            model = PeftModel.from_pretrained(
                model,
                self.adapter_path,
            )

            logger.info("PEFT adapter loaded successfully.")

            if self.merge_adapter:
                logger.info("Merging LoRA adapter into base model...")

                model = model.merge_and_unload()

                logger.info("LoRA adapter merged successfully.")

        model.eval()

        elapsed = round(
            time.time() - start_time,
            2,
        )

        logger.info(
            "Model fully loaded in %s seconds.",
            elapsed,
        )

        return model

    @staticmethod
    def print_gpu_memory() -> None:
        """
        Print GPU memory statistics.
        """

        if not torch.cuda.is_available():
            logger.warning("CUDA unavailable.")
            return

        allocated = torch.cuda.memory_allocated() / 1024**3

        reserved = torch.cuda.memory_reserved() / 1024**3

        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        max_reserved = torch.cuda.max_memory_reserved() / 1024**3

        logger.info(
            "GPU memory allocated: %.2f GB",
            allocated,
        )

        logger.info(
            "GPU memory reserved: %.2f GB",
            reserved,
        )

        logger.info(
            "GPU peak allocated: %.2f GB",
            max_allocated,
        )

        logger.info(
            "GPU peak reserved: %.2f GB",
            max_reserved,
        )
