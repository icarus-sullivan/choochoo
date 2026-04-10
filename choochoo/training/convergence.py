"""Convergence detection, plateau detection, and checkpoint ranking."""

from __future__ import annotations

import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceState:
    step: int
    loss: float
    val_loss: Optional[float] = None
    is_best: bool = False
    plateau_detected: bool = False
    overfit_detected: bool = False
    steps_since_best: int = 0


class ConvergenceDetector:
    """Detects training convergence and overfitting.

    Designed for noisy diffusion / LoRA training where:
    - Loss is noisy and improvements are gradual
    - Validation loss is preferred when available, training loss otherwise
    - Plateau detection is gated by patience to avoid false positives
    - Warmup period is excluded from all signals

    Tracks:
    - Best checkpoint (relative + absolute improvement threshold)
    - Loss plateau (mean + slope, with noise filter, gated by patience)
    - Train/val divergence (gap + trend overfitting signal)
    - Top-K checkpoint ranking
    """

    def __init__(
        self,
        patience: int = 200,
        window: int = 150,
        warmup_steps: int = 0,
        keep_top_k: int = 3,
        auto_stop: bool = False,
        improvement_threshold: float = 0.01,
        plateau_threshold: float = 0.002,
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.window = window
        self.warmup_steps = warmup_steps
        self.keep_top_k = keep_top_k
        self.auto_stop = auto_stop
        self.improvement_threshold = improvement_threshold
        self.plateau_threshold = plateau_threshold
        self.min_delta = min_delta

        self._loss_history: Deque[float] = deque(maxlen=window)
        self._val_history: Deque[float] = deque(maxlen=window)
        self._best_losses: List[Tuple[float, int]] = []  # (loss, step)
        self._steps_without_improvement: int = 0
        self._best_loss: float = float("inf")
        self._best_step: int = 0
        self._plateau_warned: bool = False
        self._prev_gap: Optional[float] = None

    def update(
        self,
        step: int,
        loss: float,
        val_loss: Optional[float] = None,
    ) -> ConvergenceState:
        """Update with new metrics. Returns current convergence state."""
        self._loss_history.append(loss)
        if val_loss is not None:
            self._val_history.append(val_loss)

        # Prefer validation loss as the tracking signal when available
        metric = val_loss if val_loss is not None else loss

        # Skip all signals during warmup — early training is unstable
        if step < self.warmup_steps:
            return ConvergenceState(step=step, loss=loss, val_loss=val_loss)

        # Improvement check: relative threshold + absolute floor
        is_best = False
        delta = self._best_loss - metric
        if delta > max(self.improvement_threshold * self._best_loss, self.min_delta):
            self._best_loss = metric
            self._best_step = step
            self._steps_without_improvement = 0
            is_best = True
            self._update_top_k(metric, step)
        else:
            self._steps_without_improvement += 1

        # Plateau gated by patience — avoids false positives during normal progress
        plateau = (
            self._steps_without_improvement >= self.patience
            and self._detect_plateau()
        )
        overfit = self._detect_overfitting()

        if plateau and not self._plateau_warned:
            self._plateau_warned = True
            logger.warning(
                "[Step %d] Loss plateau detected — LoRA may have converged. "
                "Best loss: %.6f at step %d. Consider stopping at step %d.",
                step, self._best_loss, self._best_step, self._best_step,
            )

        if overfit:
            logger.warning(
                "[Step %d] Overfitting detected: train_loss=%.6f, val_loss=%.6f",
                step, loss, val_loss,
            )

        steps_since_best = step - self._best_step

        return ConvergenceState(
            step=step,
            loss=loss,
            val_loss=val_loss,
            is_best=is_best,
            plateau_detected=plateau,
            overfit_detected=overfit,
            steps_since_best=steps_since_best,
        )

    def should_stop(self) -> bool:
        """Return True if auto_stop is enabled and training should halt."""
        return self.auto_stop and self._steps_without_improvement >= self.patience

    def _detect_plateau(self) -> bool:
        if len(self._loss_history) < self.window:
            return False
        history = list(self._loss_history)
        mean = sum(history) / len(history)
        if mean == 0:
            return False

        # High-variance signal is not a plateau — noise masquerading as flat
        std = statistics.pstdev(history)
        if std / (mean + 1e-8) > 0.1:
            return False

        # Mean-halves comparison with tight plateau threshold
        mid = self.window // 2
        first_mean = sum(history[:mid]) / mid
        second_mean = sum(history[mid:]) / (self.window - mid)
        if first_mean == 0:
            return False
        improvement = (first_mean - second_mean) / first_mean
        mean_flat = abs(improvement) < self.plateau_threshold

        # Slope check: loss must also not be trending (relative to current scale)
        slope = (history[-1] - history[0]) / len(history)
        slope_flat = abs(slope) < mean * 1e-4

        return mean_flat and slope_flat

    def _detect_overfitting(self) -> bool:
        if len(self._val_history) < 10 or len(self._loss_history) < 10:
            return False
        recent_train = sum(list(self._loss_history)[-10:]) / 10
        recent_val = sum(list(self._val_history)[-10:]) / 10
        gap = recent_val - recent_train
        # Real overfitting: meaningful absolute gap that is also widening
        is_overfit = gap > 0.05 and (self._prev_gap is None or gap > self._prev_gap)
        self._prev_gap = gap
        return is_overfit

    def _update_top_k(self, loss: float, step: int) -> None:
        self._best_losses.append((loss, step))
        self._best_losses.sort(key=lambda x: x[0])
        self._best_losses = self._best_losses[:self.keep_top_k]

    @property
    def best_step(self) -> Optional[int]:
        if self._best_losses:
            return self._best_losses[0][1]
        return None

    @property
    def top_k_checkpoints(self) -> List[Tuple[float, int]]:
        return list(self._best_losses)

    def summary(self) -> str:
        best_str = f"{self._best_loss:.6f}" if self._best_loss != float("inf") else "N/A"
        top_k = ", ".join(f"step={s} loss={l:.4f}" for l, s in self._best_losses) or "none"
        lines = [
            "Convergence Status:",
            f"  Best loss:         {best_str}",
            f"  Best step:         {self._best_step}",
            f"  Steps since best:  {self._steps_without_improvement}/{self.patience}",
            f"  Plateau detected:  {self._plateau_warned}",
            f"  Top-K checkpoints: {top_k}",
        ]
        return "\n".join(lines)
