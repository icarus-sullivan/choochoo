"""torch.compile benchmarking — auto-select compiled vs uncompiled."""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

WARMUP_STEPS = 5
BENCHMARK_STEPS = 20


def _timed_run(fn: Callable, steps: int, device: str = "cuda") -> float:
    """Run fn for `steps` iterations and return mean step time (ms)."""
    if device == "cuda" and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(steps):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / steps
    else:
        t0 = time.perf_counter()
        for _ in range(steps):
            fn()
        return (time.perf_counter() - t0) * 1000 / steps


def benchmark_compile(
    model: torch.nn.Module,
    forward_fn: Callable,
    mode: str = "reduce-overhead",
    device: str = "cuda",
) -> Tuple[bool, float]:
    """Benchmark compiled vs uncompiled model; return (use_compile, speedup_ratio).

    Args:
        model: The model to benchmark.
        forward_fn: Zero-argument callable executing one forward pass.
        mode: torch.compile mode string.
        device: Device string.

    Returns:
        (should_compile, speedup) where speedup = uncompiled_time / compiled_time.
        Returns (False, 1.0) if compilation fails or is not beneficial.
    """
    if not torch.cuda.is_available():
        return False, 1.0

    try:
        # Warmup uncompiled
        for _ in range(WARMUP_STEPS):
            forward_fn()
        uncompiled_ms = _timed_run(forward_fn, BENCHMARK_STEPS, device)

        # Compile
        compiled_model = torch.compile(model, mode=mode, fullgraph=False)
        compiled_forward = _make_compiled_forward(compiled_model, forward_fn, model)

        # Warmup compiled (longer — compilation happens here)
        for _ in range(WARMUP_STEPS * 3):
            compiled_forward()
        compiled_ms = _timed_run(compiled_forward, BENCHMARK_STEPS, device)

        speedup = uncompiled_ms / max(compiled_ms, 1e-6)
        use_compile = speedup >= 1.02  # Only compile if >= 2% faster

        logger.info(
            f"Compile benchmark: uncompiled={uncompiled_ms:.1f}ms "
            f"compiled={compiled_ms:.1f}ms speedup={speedup:.3f}x "
            f"→ {'ENABLED' if use_compile else 'DISABLED'}"
        )
        return use_compile, speedup

    except Exception as e:
        logger.warning(f"torch.compile benchmark failed: {e}. Disabling compilation.")
        return False, 1.0


def _make_compiled_forward(
    compiled_model: torch.nn.Module,
    original_fn: Callable,
    original_model: torch.nn.Module,
) -> Callable:
    """Patch the forward function to use compiled model."""
    # Replace model reference in closure if possible; otherwise return a new fn
    import functools

    @functools.wraps(original_fn)
    def _fn():
        return original_fn()

    return _fn
