"""FSDP configuration with WAN-aware block-level wrapping policies."""

from __future__ import annotations

import functools
import logging
from typing import Optional, Set, Type

import torch
import torch.nn as nn
from omegaconf import DictConfig

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        BackwardPrefetch,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import (
        ModuleWrapPolicy,
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
    _FSDP_AVAILABLE = True
except ImportError:
    _FSDP_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_wan_fsdp_policy(
    model: nn.Module,
    min_params: int = 10_000_000,
    wrap_strategy: str = "wan_optimized",
) -> object:
    """Get FSDP auto-wrap policy for WAN DiT architecture.

    WAN-optimized strategy:
    - Detect DiT/transformer blocks
    - Wrap at block level (not per-layer) to minimize communication
    - Only wrap modules above min_params threshold (avoid over-sharding small layers)
    - Hybrid: large blocks → FSDP, small utility layers → remain un-sharded
    """
    if not _FSDP_AVAILABLE:
        raise ImportError("FSDP not available in this PyTorch version")

    if wrap_strategy == "wan_optimized":
        # Collect block classes above parameter threshold
        block_classes = _find_block_classes(model, min_params)
        if block_classes:
            logger.info(
                f"FSDP wrap policy: WAN block-level wrapping "
                f"({len(block_classes)} block types, min_params={min_params:,})"
            )
            return ModuleWrapPolicy(block_classes)

    # Size-based fallback
    logger.info(f"FSDP wrap policy: size-based (min_params={min_params:,})")
    return functools.partial(size_based_auto_wrap_policy, min_num_params=min_params)


def _find_block_classes(model: nn.Module, min_params: int) -> Set[Type[nn.Module]]:
    """Find all module types that look like DiT blocks and exceed min_params."""
    candidates: dict = {}

    for name, module in model.named_modules():
        if module is model:
            continue
        param_count = sum(p.numel() for p in module.parameters())
        if param_count < min_params:
            continue

        cls = type(module)
        cls_name = cls.__name__

        # Heuristics for WAN DiT block detection
        is_block = any(
            kw in cls_name
            for kw in ("Block", "Layer", "EncoderLayer", "DecoderLayer", "TransformerBlock")
        ) or (
            hasattr(module, "self_attn") or
            hasattr(module, "attn") or
            (hasattr(module, "mlp") and hasattr(module, "norm"))
        )

        if is_block:
            candidates[cls] = param_count

    # Sort by param count, take top classes (likely main blocks)
    return set(
        cls for cls, _ in sorted(candidates.items(), key=lambda x: -x[1])
    )


def setup_fsdp(
    model: nn.Module,
    cfg: DictConfig,
    device_id: int = 0,
) -> nn.Module:
    """Wrap model in FSDP with WAN-optimized settings."""
    if not _FSDP_AVAILABLE:
        raise ImportError("torch.distributed.fsdp is not available")

    cfg_fsdp = cfg.distributed.fsdp
    dtype_str = cfg.model.dtype

    # Mixed precision policy
    param_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype_str)
    reduce_dtype = torch.float32  # Always reduce in fp32 for stability

    mixed_precision = None
    if param_dtype is not None:
        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=param_dtype,
        )

    # Sharding strategy
    sharding = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }.get(cfg_fsdp.get("sharding_strategy", "FULL_SHARD"), ShardingStrategy.FULL_SHARD)

    # CPU offload
    cpu_offload = CPUOffload(offload_params=True) if cfg_fsdp.get("cpu_offload", False) else None

    # Wrap policy
    min_params = int(cfg_fsdp.get("min_params", 1e7))
    auto_wrap = cfg_fsdp.get("auto_wrap", "wan_optimized")
    wrap_policy = get_wan_fsdp_policy(model, min_params=min_params, wrap_strategy=auto_wrap)

    wrapped = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=sharding,
        cpu_offload=cpu_offload,
        device_id=device_id,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        limit_all_gathers=True,
        use_orig_params=True,  # Required for LoRA parameter access
    )

    logger.info(
        f"FSDP setup: sharding={sharding.name}, mixed_precision={dtype_str}, "
        f"cpu_offload={cpu_offload is not None}"
    )
    return wrapped
