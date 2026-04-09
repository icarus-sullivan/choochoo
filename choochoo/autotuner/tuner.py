"""Main AutoTuner — orchestrates all hardware-aware optimization decisions."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from .batch import BatchSizeTuner, MemoryBudgetEstimator
from .compile import benchmark_compile
from .hardware import HardwareDetector
from .vram24 import VRAM24Optimizer

logger = logging.getLogger(__name__)


class AutoTuner:
    """Full hardware-aware optimization system.

    Call .tune(cfg, model, probe_fn) to get a fully resolved config.
    """

    def __init__(self):
        self._hw = HardwareDetector()
        self._hw_info: Optional[Dict[str, Any]] = None
        self._vram24 = VRAM24Optimizer()

    def detect_hardware(self) -> Dict[str, Any]:
        self._hw_info = self._hw.detect()
        return self._hw_info

    def select_precision(self) -> str:
        hw = self._hw_info or self._hw.detect()
        if hw.get("has_bf16", False):
            logger.info("Precision: bf16 (native support detected)")
            return "bf16"
        if hw.get("has_cuda", False):
            logger.info("Precision: fp16 (bf16 not available)")
            return "fp16"
        logger.info("Precision: fp32 (no GPU)")
        return "fp32"

    def select_distributed_strategy(self, model: nn.Module) -> str:
        hw = self._hw_info or self._hw.detect()
        num_gpus = hw.get("num_gpus", 0)

        if num_gpus <= 1:
            return "single"

        num_params = sum(p.numel() for p in model.parameters()) / 1e9
        vram_min = hw.get("vram_per_gpu_gb_min", 16)

        # Use FSDP for large models (>2B params) or tight VRAM
        if num_params > 2.0 or vram_min < 24:
            if num_gpus >= 2:
                logger.info(f"Distributed: FSDP ({num_gpus} GPUs, {num_params:.1f}B params)")
                return "fsdp"

        logger.info(f"Distributed: DDP ({num_gpus} GPUs)")
        return "ddp"

    def find_optimal_batch(
        self,
        probe_fn: Callable[[int], None],
        device: int = 0,
        target_vram_util: float = 0.92,
    ) -> tuple:
        tuner = BatchSizeTuner(
            target_vram_utilization=target_vram_util,
            min_batch=1,
            max_batch=64,
        )
        return tuner.find_optimal(probe_fn, device)

    def tune_dataloader(self) -> Dict[str, Any]:
        hw = self._hw_info or self._hw.detect()
        cpu_cores = hw.get("cpu_cores", 4)
        ram_gb = hw.get("ram_gb", 16)

        num_workers = min(max(cpu_cores // 2, 2), 12)
        prefetch_factor = 4 if ram_gb > 32 else 2
        persistent = True

        logger.info(
            f"DataLoader: num_workers={num_workers}, prefetch={prefetch_factor}"
        )
        return {
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "pin_memory": True,
            "persistent_workers": persistent,
        }

    def benchmark_compile(
        self,
        model: nn.Module,
        forward_fn: Callable,
        mode: str = "reduce-overhead",
    ) -> tuple:
        return benchmark_compile(model, forward_fn, mode)

    def optimize_runtime(self, model: nn.Module, cfg: Any) -> nn.Module:
        """Apply runtime optimizations: channels_last, compile, flash attn."""
        hw = self._hw_info or self._hw.detect()

        if getattr(cfg.performance, "channels_last", False):
            try:
                model = model.to(memory_format=torch.channels_last)
                logger.info("Applied channels_last memory format")
            except Exception:
                pass

        return model

    def warmup_profile(
        self,
        train_step_fn: Callable[[int], float],
        n_steps: int = 30,
    ) -> float:
        """Run warmup steps and return average throughput (samples/sec)."""
        times = []
        import time
        for i in range(n_steps):
            t0 = time.perf_counter()
            train_step_fn(i)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        # Discard first 10 (JIT warmup)
        steady = times[10:] if len(times) > 10 else times
        mean_step = sum(steady) / len(steady)
        logger.info(f"Warmup profile: mean step={mean_step*1000:.1f}ms")
        return mean_step

    def tune(
        self,
        cfg: Any,
        model: nn.Module,
        probe_fn: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        """Full auto-tuning pass. Returns dict of resolved settings."""
        logger.info("=" * 60)
        logger.info("AutoTuner: starting hardware analysis")

        hw = self.detect_hardware()
        logger.info(
            f"Hardware: {hw.get('num_gpus', 0)} GPU(s), "
            f"{hw.get('vram_per_gpu_gb_min', 0):.0f}GB VRAM min, "
            f"{hw.get('cpu_cores', 0)} CPU cores, "
            f"{hw.get('ram_gb', 0):.0f}GB RAM"
        )

        precision = self.select_precision()
        strategy = self.select_distributed_strategy(model)
        dl_cfg = self.tune_dataloader()

        target_vram_util = float(getattr(cfg.training, "target_vram_utilization", 0.92))
        batch_size = None
        grad_accum = None

        # Apply allocator tuning for all CUDA GPUs (sets PYTORCH_CUDA_ALLOC_CONF
        # and per-process memory fraction). Benefits constrained and large GPUs alike.
        if hw.get("has_cuda", False) and hw.get("num_gpus", 0) > 0:
            self._vram24.configure_memory_settings(device=0)

        # 24GB specialised path: use static memory budgeting instead of binary search
        # to avoid the OOM-probe overhead on constrained GPUs.
        is_24gb = hw.get("num_gpus", 0) > 0 and self._vram24.is_24gb_gpu(0)
        if is_24gb and hw.get("has_cuda", False):
            logger.info("24GB VRAM GPU detected — applying specialised memory path")

            dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
            param_dtype = dtype_map.get(precision, torch.bfloat16)
            budget = self._vram24.estimate_budget(model, param_dtype=param_dtype)
            self._vram24.log_budget(budget)

            batch_size = budget.recommended_batch_size
            grad_accum = budget.recommended_grad_accum

        elif probe_fn is not None and hw.get("has_cuda", False):
            try:
                batch_size, grad_accum = self.find_optimal_batch(
                    probe_fn, target_vram_util=target_vram_util
                )
            except Exception as e:
                logger.warning(f"Batch size tuning failed: {e}. Using defaults.")

        result = {
            "precision": precision,
            "distributed_strategy": strategy,
            "dataloader": dl_cfg,
            "hardware": hw,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "is_24gb_path": is_24gb,
        }

        logger.info(f"AutoTuner results: {result}")
        logger.info("=" * 60)
        return result
