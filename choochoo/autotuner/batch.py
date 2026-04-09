"""Binary-search batch size auto-tuner with VRAM safety margin."""

from __future__ import annotations

import gc
import logging
from typing import Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class BatchSizeTuner:
    """Find max batch size that fits in VRAM using binary search.

    Uses a probe function that runs a single forward+backward pass.
    """

    def __init__(
        self,
        target_vram_utilization: float = 0.92,
        min_batch: int = 1,
        max_batch: int = 64,
    ):
        self.target_vram_util = target_vram_utilization
        self.min_batch = min_batch
        self.max_batch = max_batch

    def find_optimal(
        self,
        probe_fn: Callable[[int], None],
        device: int = 0,
    ) -> Tuple[int, int]:
        """Binary search for optimal batch size.

        Args:
            probe_fn: Function that takes batch_size and runs a forward+backward
                      pass. Should raise RuntimeError (OOM) if it fails.
            device: CUDA device index.

        Returns:
            (optimal_batch_size, recommended_grad_accumulation_steps)
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available; returning default batch size 1.")
            return 1, 1

        lo, hi = self.min_batch, self.max_batch
        best = self.min_batch

        while lo <= hi:
            mid = (lo + hi) // 2
            self._clear_cache(device)
            try:
                probe_fn(mid)
                self._clear_cache(device)

                # Check VRAM utilization
                util = self._vram_utilization(device)
                if util <= self.target_vram_util:
                    best = mid
                    lo = mid + 1
                else:
                    # Fits but over target — this is still valid, keep it
                    best = mid
                    break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    hi = mid - 1
                    self._clear_cache(device)
                else:
                    raise

        # Compute gradient accumulation for effective batch of 16+ if needed
        target_effective = max(best, 16)
        grad_accum = max(1, target_effective // best)

        logger.info(
            f"Batch size tuning: batch={best}, grad_accum={grad_accum}, "
            f"effective={best * grad_accum}"
        )
        return best, grad_accum

    @staticmethod
    def _vram_utilization(device: int) -> float:
        free, total = torch.cuda.mem_get_info(device)
        return 1.0 - free / total

    @staticmethod
    def _clear_cache(device: int) -> None:
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


class MemoryBudgetEstimator:
    """Estimate memory usage components for 24GB VRAM budget planning."""

    @staticmethod
    def estimate_model_memory_gb(model: torch.nn.Module, dtype: torch.dtype) -> float:
        bytes_per_param = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int8: 1,
        }.get(dtype, 4)
        num_params = sum(p.numel() for p in model.parameters())
        return num_params * bytes_per_param / (1024**3)

    @staticmethod
    def estimate_optimizer_memory_gb(model: torch.nn.Module, optimizer_type: str = "adamw") -> float:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # AdamW: 2 states (m, v) in fp32
        multiplier = 8 if optimizer_type == "adamw" else 4
        return trainable * multiplier / (1024**3)

    @staticmethod
    def estimate_activation_memory_gb(
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_layers: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> float:
        bytes_per = 2 if dtype in (torch.float16, torch.bfloat16) else 4
        # Rough: batch * seq * hidden * layers * 4 (fwd activations factor)
        bytes_total = batch_size * seq_len * hidden_dim * num_layers * 4 * bytes_per
        return bytes_total / (1024**3)
