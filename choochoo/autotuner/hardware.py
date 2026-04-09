"""Hardware detection and capability probing."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional

import psutil
import torch


class HardwareDetector:
    """Detect all relevant hardware characteristics for training optimization."""

    def detect(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}

        info["cpu_cores"] = psutil.cpu_count(logical=False) or psutil.cpu_count()
        info["cpu_threads"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = psutil.virtual_memory().total / (1024**3)

        if not torch.cuda.is_available():
            info["num_gpus"] = 0
            info["has_cuda"] = False
            return info

        info["has_cuda"] = True
        info["num_gpus"] = torch.cuda.device_count()
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(info["num_gpus"])
        ]
        info["vram_per_gpu_gb"] = [
            torch.cuda.get_device_properties(i).total_memory / (1024**3)
            for i in range(info["num_gpus"])
        ]
        info["vram_per_gpu_gb_min"] = min(info["vram_per_gpu_gb"])
        info["vram_total_gb"] = sum(info["vram_per_gpu_gb"])

        props = [
            torch.cuda.get_device_properties(i) for i in range(info["num_gpus"])
        ]
        info["compute_capabilities"] = [
            (p.major, p.minor) for p in props
        ]
        info["compute_capability_min"] = min(info["compute_capabilities"])

        # bf16 support requires compute capability >= 8.0 (Ampere+)
        info["has_bf16"] = all(
            cc[0] >= 8 for cc in info["compute_capabilities"]
        )
        # Flash attention requires sm >= 7.5
        info["has_flash_attn"] = self._probe_flash_attn() and all(
            cc[0] >= 8 or (cc[0] == 7 and cc[1] >= 5)
            for cc in info["compute_capabilities"]
        )
        info["has_xformers"] = self._probe_xformers()
        info["has_nvlink"] = self._probe_nvlink(info["num_gpus"])

        return info

    @staticmethod
    def _probe_flash_attn() -> bool:
        try:
            import flash_attn
            # A functional flash_attn has __version__. Stub modules (installed to
            # work around ABI crashes) are empty ModuleType objects without it.
            return hasattr(flash_attn, "__version__")
        except ImportError:
            return False

    @staticmethod
    def _probe_xformers() -> bool:
        try:
            import xformers  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _probe_nvlink(num_gpus: int) -> bool:
        if num_gpus <= 1:
            return False
        try:
            result = subprocess.run(
                ["nvidia-smi", "nvlink", "--status", "-i", "0"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return "Active" in result.stdout
        except Exception:
            return False

    @staticmethod
    def get_free_vram_gb(device: int = 0) -> float:
        if not torch.cuda.is_available():
            return 0.0
        free, total = torch.cuda.mem_get_info(device)
        return free / (1024**3)

    @staticmethod
    def get_used_vram_gb(device: int = 0) -> float:
        if not torch.cuda.is_available():
            return 0.0
        free, total = torch.cuda.mem_get_info(device)
        return (total - free) / (1024**3)
