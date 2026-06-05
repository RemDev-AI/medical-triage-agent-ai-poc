# medical-triage-agent-ai-poc/backend/app/monitoring/gpu_monitor.py

from __future__ import annotations

import threading
import time
from typing import Dict

import torch


class GPUMonitor:
    """
    Monitoring GPU pour le moteur d'inférence.

    Métriques suivies :
    - CUDA disponible
    - nombre de GPU
    - nom du GPU
    - VRAM totale
    - VRAM utilisée
    - VRAM libre
    - throughput (requêtes/seconde)
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()

        self.request_count = 0
        self.start_time = time.time()

    def increment_request(self) -> None:
        """
        À appeler après chaque inférence.
        """

        with self.lock:
            self.request_count += 1

    def reset_throughput(self) -> None:
        """
        Réinitialise le compteur.
        """

        with self.lock:
            self.request_count = 0
            self.start_time = time.time()

    def get_throughput(self) -> float:
        """
        Requêtes / seconde.
        """

        with self.lock:
            elapsed = max(time.time() - self.start_time, 1e-6)

            return round(
                self.request_count / elapsed,
                3,
            )

    def get_gpu_stats(self) -> Dict:
        """
        Retourne les métriques GPU.
        """

        cuda_available = torch.cuda.is_available()

        if not cuda_available:
            return {
                "cuda_available": False,
                "gpu_count": 0,
                "gpu_name": "CPU_ONLY",
                "vram_total_mb": 0,
                "vram_used_mb": 0,
                "vram_free_mb": 0,
                "vram_usage_percent": 0.0,
                "throughput_rps": self.get_throughput(),
            }

        device_index = torch.cuda.current_device()

        gpu_name = torch.cuda.get_device_name(device_index)

        properties = torch.cuda.get_device_properties(device_index)

        total_memory = properties.total_memory

        allocated_memory = torch.cuda.memory_allocated(device_index)

        reserved_memory = torch.cuda.memory_reserved(device_index)

        free_memory = max(
            total_memory - reserved_memory,
            0,
        )

        usage_percent = (
            (reserved_memory / total_memory) * 100
            if total_memory > 0
            else 0
        )

        return {
            "cuda_available": True,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": gpu_name,
            "vram_total_mb": round(
                total_memory / 1024 / 1024,
                2,
            ),
            "vram_used_mb": round(
                allocated_memory / 1024 / 1024,
                2,
            ),
            "vram_reserved_mb": round(
                reserved_memory / 1024 / 1024,
                2,
            ),
            "vram_free_mb": round(
                free_memory / 1024 / 1024,
                2,
            ),
            "vram_usage_percent": round(
                usage_percent,
                2,
            ),
            "throughput_rps": self.get_throughput(),
        }

    def health_status(self) -> Dict:
        """
        État simplifié utilisé par les endpoints health.
        """

        metrics = self.get_gpu_stats()

        return {
            "status": "healthy",
            "cuda": metrics["cuda_available"],
            "gpu": metrics["gpu_name"],
            "throughput_rps": metrics["throughput_rps"],
            "vram_usage_percent": metrics[
                "vram_usage_percent"
            ],
        }


gpu_monitor = GPUMonitor()
