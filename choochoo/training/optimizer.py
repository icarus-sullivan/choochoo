"""Fused optimizer construction (AdamW, Lion, 8-bit AdamW)."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_optimizer(
    params: Iterable[nn.Parameter],
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    fused: bool = True,
    use_8bit: bool = False,
) -> torch.optim.Optimizer:
    """Build the best available optimizer.

    Priority: fused AdamW > 8-bit AdamW > standard AdamW > Lion
    """
    param_list = list(params)

    if use_8bit:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(
                param_list, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
            )
            logger.info("Optimizer: 8-bit AdamW (bitsandbytes)")
            return opt
        except ImportError:
            logger.warning("bitsandbytes not available; falling back to standard AdamW")

    if optimizer_type == "lion":
        try:
            from lion_pytorch import Lion
            opt = Lion(param_list, lr=lr, weight_decay=weight_decay, betas=(betas[0], betas[1]))
            logger.info("Optimizer: Lion")
            return opt
        except ImportError:
            logger.warning("lion_pytorch not installed; falling back to AdamW")

    # Try fused AdamW (PyTorch 2.x, requires CUDA)
    if fused and torch.cuda.is_available():
        try:
            opt = torch.optim.AdamW(
                param_list,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
                fused=True,
            )
            logger.info("Optimizer: fused AdamW")
            return opt
        except Exception:
            pass

    opt = torch.optim.AdamW(
        param_list, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
    )
    logger.info("Optimizer: standard AdamW")
    return opt
