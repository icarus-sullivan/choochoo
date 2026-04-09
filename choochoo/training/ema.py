"""Exponential Moving Average (EMA) for LoRA parameter smoothing."""

from __future__ import annotations

import copy
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


class EMAModel:
    """Maintains an EMA copy of model parameters.

    Only tracks trainable (LoRA) parameters to minimize memory overhead.
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 100,
        use_ema_warmup: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.optimization_step = 0

        params = list(parameters)
        self.shadow_params = [
            p.clone().detach().to(device or p.device) for p in params
        ]
        # Keep reference to live params for updates
        self._params_refs = params

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        if self.use_ema_warmup:
            cur_decay = 1 - (1 + step) ** -0.8
        else:
            cur_decay = (1 + step) / (10 + step)
        return min(cur_decay, self.decay)

    @torch.no_grad()
    def step(self) -> None:
        self.optimization_step += 1
        decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, self._params_refs):
            if param.requires_grad:
                s_param.mul_(decay).add_(param.data, alpha=1 - decay)
            else:
                s_param.copy_(param.data)

    def copy_to(self, parameters: Optional[Iterable[nn.Parameter]] = None) -> None:
        """Copy EMA params into live model params (for evaluation)."""
        params = list(parameters or self._params_refs)
        for s_param, param in zip(self.shadow_params, params):
            param.data.copy_(s_param.data)

    def restore(self, parameters: Optional[Iterable[nn.Parameter]] = None) -> None:
        """Restore live model params after copy_to (swap back)."""
        # Assumes caller saved originals; just re-copy from shadow isn't enough
        # Caller should use context manager pattern
        pass

    def state_dict(self) -> dict:
        return {
            "step": self.optimization_step,
            "shadow_params": [p.clone() for p in self.shadow_params],
            "decay": self.decay,
        }

    def load_state_dict(self, state: dict) -> None:
        self.optimization_step = state["step"]
        self.decay = state["decay"]
        for s, loaded in zip(self.shadow_params, state["shadow_params"]):
            s.copy_(loaded)
