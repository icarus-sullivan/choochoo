"""Distributed training initialization and device setup."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import datetime

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class DistributedSetup:
    rank: int
    local_rank: int
    world_size: int
    strategy: str
    device: torch.device
    is_main: bool
    cpu_group: Optional[object] = None  # gloo process group for CPU-side barriers (no NCCL timeout)


def init_distributed(strategy: str = "auto", port: int = 29500) -> DistributedSetup:
    """Initialize distributed training. Returns DistributedSetup.

    Supports:
    - Single GPU (no overhead)
    - DDP via torchrun
    - FSDP via torchrun
    - Auto-detect from env vars
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not dist.is_initialized():
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", str(port))
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size,
            # Large models (20B+) take ~10min per GPU to transfer over PCIe. Serial
            # DDP transfer takes N × transfer_time, easily exceeding the default 10min
            # NCCL watchdog timeout. Set to 2 hours to cover any realistic configuration.
            timeout=datetime.timedelta(hours=2),
        )
        torch.cuda.set_device(local_rank)
        _cpu_group = dist.new_group(list(range(world_size)), backend="gloo")
    else:
        _cpu_group = None

    if strategy == "auto":
        if world_size == 1:
            strategy = "single"
        elif world_size <= 4:
            strategy = "ddp"
        else:
            strategy = "fsdp"

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    setup = DistributedSetup(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        strategy=strategy,
        device=device,
        is_main=(rank == 0),
        cpu_group=_cpu_group if dist.is_initialized() else None,
    )

    if rank == 0:
        logger.info(
            f"Distributed: strategy={strategy}, world_size={world_size}, "
            f"device={device}"
        )

    return setup


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
