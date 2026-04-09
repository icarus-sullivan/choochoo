"""Training profiler and bottleneck detection system."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    step: int
    data_time_ms: float
    forward_ms: float
    backward_ms: float
    optimizer_ms: float
    total_ms: float
    gpu_util_pct: float
    vram_used_gb: float
    samples_per_sec: float


@dataclass
class ProfileSummary:
    mean_step_ms: float
    mean_data_ms: float
    mean_forward_ms: float
    mean_backward_ms: float
    mean_optimizer_ms: float
    mean_gpu_util: float
    mean_vram_gb: float
    samples_per_sec: float
    bottleneck: str
    suggestions: List[str] = field(default_factory=list)


class TrainingProfiler:
    """Lightweight profiler that tracks per-step timing and GPU metrics.

    Does not use torch.profiler (too heavy for continuous use). Uses CUDA
    events for accurate GPU timing.
    """

    def __init__(self, window: int = 50, batch_size: int = 1):
        self.window = window
        self.batch_size = batch_size
        self._history: Deque[StepMetrics] = deque(maxlen=window)

        self._data_start: Optional[float] = None
        self._fwd_start: Optional[torch.cuda.Event] = None
        self._bwd_start: Optional[torch.cuda.Event] = None
        self._opt_start: Optional[torch.cuda.Event] = None
        self._step_wall_start: Optional[float] = None

        self._last_data_ms: float = 0.0
        self._fwd_event_end: Optional[torch.cuda.Event] = None
        self._bwd_event_end: Optional[torch.cuda.Event] = None
        self._opt_event_end: Optional[torch.cuda.Event] = None

    def start_data(self) -> None:
        self._data_start = time.perf_counter()
        self._step_wall_start = time.perf_counter()

    def end_data(self) -> None:
        if self._data_start is not None:
            self._last_data_ms = (time.perf_counter() - self._data_start) * 1000

    def start_forward(self) -> None:
        if torch.cuda.is_available():
            self._fwd_start = torch.cuda.Event(enable_timing=True)
            self._fwd_event_end = torch.cuda.Event(enable_timing=True)
            self._fwd_start.record()

    def end_forward(self) -> None:
        if self._fwd_event_end is not None:
            self._fwd_event_end.record()

    def start_backward(self) -> None:
        if torch.cuda.is_available():
            self._bwd_start = torch.cuda.Event(enable_timing=True)
            self._bwd_event_end = torch.cuda.Event(enable_timing=True)
            self._bwd_start.record()

    def end_backward(self) -> None:
        if self._bwd_event_end is not None:
            self._bwd_event_end.record()

    def start_optimizer(self) -> None:
        if torch.cuda.is_available():
            self._opt_start = torch.cuda.Event(enable_timing=True)
            self._opt_event_end = torch.cuda.Event(enable_timing=True)
            self._opt_start.record()

    def end_optimizer(self) -> None:
        if self._opt_event_end is not None:
            self._opt_event_end.record()

    def record_step(self, step: int) -> None:
        """Synchronize CUDA and record this step's metrics."""
        if not torch.cuda.is_available():
            return

        torch.cuda.synchronize()

        fwd_ms = (
            self._fwd_start.elapsed_time(self._fwd_event_end)
            if self._fwd_start and self._fwd_event_end
            else 0.0
        )
        bwd_ms = (
            self._bwd_start.elapsed_time(self._bwd_event_end)
            if self._bwd_start and self._bwd_event_end
            else 0.0
        )
        opt_ms = (
            self._opt_start.elapsed_time(self._opt_event_end)
            if self._opt_start and self._opt_event_end
            else 0.0
        )

        total_ms = (time.perf_counter() - self._step_wall_start) * 1000 if self._step_wall_start else 1.0
        samples_per_sec = (self.batch_size * 1000.0) / max(total_ms, 1.0)

        # GPU utilization via NVML if available
        gpu_util = self._get_gpu_util()
        vram_used = self._get_vram_used()

        metrics = StepMetrics(
            step=step,
            data_time_ms=self._last_data_ms,
            forward_ms=fwd_ms,
            backward_ms=bwd_ms,
            optimizer_ms=opt_ms,
            total_ms=total_ms,
            gpu_util_pct=gpu_util,
            vram_used_gb=vram_used,
            samples_per_sec=samples_per_sec,
        )
        self._history.append(metrics)

    @staticmethod
    def _get_gpu_util() -> float:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except Exception:
            return 0.0

    @staticmethod
    def _get_vram_used() -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / (1024**3)
        return 0.0

    def summarize(self) -> ProfileSummary:
        if not self._history:
            return ProfileSummary(0, 0, 0, 0, 0, 0, 0, 0, "unknown")

        metrics = list(self._history)

        def avg(fn):
            return sum(fn(m) for m in metrics) / len(metrics)

        mean_total = avg(lambda m: m.total_ms)
        mean_data = avg(lambda m: m.data_time_ms)
        mean_fwd = avg(lambda m: m.forward_ms)
        mean_bwd = avg(lambda m: m.backward_ms)
        mean_opt = avg(lambda m: m.optimizer_ms)
        mean_util = avg(lambda m: m.gpu_util_pct)
        mean_vram = avg(lambda m: m.vram_used_gb)
        samples_sec = avg(lambda m: m.samples_per_sec)

        bottleneck, suggestions = self._detect_bottleneck(
            mean_data, mean_fwd, mean_bwd, mean_util, mean_total
        )

        return ProfileSummary(
            mean_step_ms=mean_total,
            mean_data_ms=mean_data,
            mean_forward_ms=mean_fwd,
            mean_backward_ms=mean_bwd,
            mean_optimizer_ms=mean_opt,
            mean_gpu_util=mean_util,
            mean_vram_gb=mean_vram,
            samples_per_sec=samples_sec,
            bottleneck=bottleneck,
            suggestions=suggestions,
        )

    @staticmethod
    def _detect_bottleneck(
        data_ms: float, fwd_ms: float, bwd_ms: float, gpu_util: float, total_ms: float
    ):
        suggestions = []
        bottleneck = "compute"

        data_ratio = data_ms / max(total_ms, 1.0)
        if data_ratio > 0.20:
            bottleneck = "dataloader"
            suggestions.append(
                f"Dataloader consuming {data_ratio*100:.0f}% of step time → increase num_workers or prefetch_factor"
            )

        if gpu_util < 85 and data_ratio < 0.10:
            bottleneck = "cpu"
            suggestions.append(
                "GPU utilization below 85% → potential CPU bottleneck; consider async preprocessing"
            )

        if gpu_util < 60:
            suggestions.append("GPU idle time high → consider increasing batch size")

        return bottleneck, suggestions
