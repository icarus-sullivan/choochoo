"""Learning rate scheduler construction."""

from __future__ import annotations

import math
from typing import Optional

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_warmup_steps: int = 100,
    num_training_steps: int = 2000,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build learning rate scheduler."""

    if scheduler_type == "cosine":
        return _cosine_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, num_cycles, last_epoch
        )
    elif scheduler_type == "linear":
        return _linear_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, last_epoch
        )
    elif scheduler_type == "constant":
        return _constant_with_warmup(optimizer, num_warmup_steps, last_epoch)
    elif scheduler_type == "cosine_with_restarts":
        return _cosine_restarts(
            optimizer, num_warmup_steps, num_training_steps, num_cycles, last_epoch
        )
    elif scheduler_type == "polynomial":
        return _polynomial_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, last_epoch
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


def _cosine_with_warmup(optimizer, warmup, total, cycles, last_epoch):
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycles * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def _linear_with_warmup(optimizer, warmup, total, last_epoch):
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def _constant_with_warmup(optimizer, warmup, last_epoch):
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def _cosine_restarts(optimizer, warmup, total, num_cycles, last_epoch):
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((num_cycles * progress) % 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def _polynomial_with_warmup(optimizer, warmup, total, last_epoch, power=2.0):
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        if step > total:
            return 0.0
        pct_remaining = 1 - (step - warmup) / (total - warmup)
        return pct_remaining ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
