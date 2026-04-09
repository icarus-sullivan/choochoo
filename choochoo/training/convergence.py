"""Convergence detection, plateau detection, and checkpoint ranking."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceState:
    step: int
    loss: float
    val_loss: Optional[float] = None
    is_best: bool = False
    plateau_detected: bool = False
    overfit_detected: bool = False


class ConvergenceDetector:
    """Detects training convergence and overfitting.

    Tracks:
    - Loss plateau (< threshold improvement over patience steps)
    - Train/val divergence (overfitting signal)
    - Best checkpoint tracking with top-K ranking
    """

    def __init__(
        self,
        patience: int = 200,
        threshold: float = 0.01,
        window: int = 50,
        keep_top_k: int = 3,
        auto_stop: bool = False,
    ):
        self.patience = patience
        self.threshold = threshold
        self.window = window
        self.keep_top_k = keep_top_k
        self.auto_stop = auto_stop

        self._loss_history: Deque[float] = deque(maxlen=window)
        self._val_history: Deque[float] = deque(maxlen=window)
        self._best_losses: List[Tuple[float, int]] = []  # (loss, step)
        self._steps_without_improvement = 0
        self._best_loss = float("inf")
        self._plateau_warned = False

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

        is_best = False
        if loss < self._best_loss * (1 - self.threshold):
            self._best_loss = loss
            self._steps_without_improvement = 0
            is_best = True
            self._update_top_k(loss, step)
        else:
            self._steps_without_improvement += 1

        plateau = self._detect_plateau()
        overfit = self._detect_overfitting()

        if plateau and not self._plateau_warned:
            self._plateau_warned = True
            logger.warning(
                f"[Step {step}] Loss plateau detected — LoRA may have converged. "
                f"Best loss: {self._best_loss:.6f} at step {self.best_step}. "
                f"Consider stopping at step {self.best_step}."
            )

        if overfit:
            logger.warning(
                f"[Step {step}] Overfitting detected: "
                f"train_loss={loss:.6f}, val_loss={val_loss:.6f}"
            )

        return ConvergenceState(
            step=step,
            loss=loss,
            val_loss=val_loss,
            is_best=is_best,
            plateau_detected=plateau,
            overfit_detected=overfit,
        )

    def should_stop(self) -> bool:
        """Return True if auto_stop is enabled and training should halt."""
        return self.auto_stop and self._steps_without_improvement >= self.patience

    def _detect_plateau(self) -> bool:
        if len(self._loss_history) < self.window:
            return False
        first_half = list(self._loss_history)[:self.window // 2]
        second_half = list(self._loss_history)[self.window // 2:]
        first_mean = sum(first_half) / len(first_half)
        second_mean = sum(second_half) / len(second_half)
        if first_mean == 0:
            return False
        improvement = (first_mean - second_mean) / first_mean
        return abs(improvement) < self.threshold

    def _detect_overfitting(self) -> bool:
        if len(self._val_history) < 10 or len(self._loss_history) < 10:
            return False
        recent_train = sum(list(self._loss_history)[-10:]) / 10
        recent_val = sum(list(self._val_history)[-10:]) / 10
        # Overfitting if val is significantly higher than train
        return recent_val > recent_train * 1.2

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
        lines = [
            f"Best loss: {self._best_loss:.6f} at step {self.best_step}",
            f"Steps without improvement: {self._steps_without_improvement}/{self.patience}",
            "Top-K checkpoints: " + ", ".join(f"step={s} loss={l:.4f}" for l, s in self._best_losses),
        ]
        return "\n".join(lines)
