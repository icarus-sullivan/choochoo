"""DDP setup with optimal settings for LoRA training."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def setup_ddp(
    model: nn.Module,
    cfg: DictConfig,
    device_id: int = 0,
) -> nn.Module:
    """Wrap model in DDP with settings optimized for LoRA.

    Key settings for LoRA:
    - find_unused_parameters=False: LoRA injects new params, but frozen base
      params are not unused — they receive gradients via chain rule.
    - gradient_as_bucket_view=True: reduces memory allocation
    - static_graph=True: enables static computation graph optimizations
    """
    cfg_ddp = cfg.distributed.ddp

    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1

    # All ranks transfer in parallel — each GPU owns its own PCIe lane.
    # Parallel transfer halves per-GPU bandwidth vs serial but completes in the
    # same wall-clock time and avoids leaving NCCL idle long enough to hit the
    # watchdog timeout.
    logger.info("Rank %d/%d: transferring model to cuda:%d...", rank, world - 1, device_id)
    model = model.to(f"cuda:{device_id}")
    logger.info("Rank %d/%d: transfer complete", rank, world - 1)

    wrapped = DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=bool(cfg_ddp.get("find_unused_parameters", False)),
        gradient_as_bucket_view=bool(cfg_ddp.get("gradient_as_bucket_view", True)),
        static_graph=bool(cfg_ddp.get("static_graph", True)),
        bucket_cap_mb=int(cfg_ddp.get("bucket_cap_mb", 200)),
    )

    logger.info(
        f"DDP setup: device={device_id}, "
        f"static_graph={cfg_ddp.get('static_graph', True)}"
    )
    return wrapped
