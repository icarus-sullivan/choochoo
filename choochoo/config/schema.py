"""Configuration schema, validation, and loading for choochoo."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf

from .defaults import DEFAULTS


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning new dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: Union[str, Path]) -> DictConfig:
    """Load, validate, and merge a YAML config with defaults."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}

    merged = _deep_merge(DEFAULTS, user_cfg)
    cfg = OmegaConf.create(merged)
    validate_config(cfg)
    return cfg


def validate_config(cfg: DictConfig) -> None:
    """Validate config fields and raise descriptive errors."""
    if not cfg.get("name") or not str(cfg.name).strip():
        raise ValueError(
            "name is required. Set a short identifier for this run "
            "(e.g. name: my_concept). It is used in all output filenames."
        )
    if not str(cfg.name).replace("-", "").replace("_", "").isalnum():
        raise ValueError(
            f"name must contain only letters, numbers, hyphens, and underscores. Got: '{cfg.name}'"
        )

    valid_model_types = {"wan22", "wan21", "wan_i2v", "wan22_dual", "qwen_edit", "qwen", "ltx2"}
    if cfg.model.type not in valid_model_types:
        raise ValueError(
            f"model.type must be one of {valid_model_types}, got '{cfg.model.type}'"
        )

    valid_regimes = {"high", "low", "both", "auto"}
    regime = str(cfg.model.get("noise_regime", "auto")).lower()
    if regime not in valid_regimes:
        raise ValueError(
            f"model.noise_regime must be one of {valid_regimes}, got '{regime}'"
        )

    if cfg.model.type == "wan22_dual":
        dual = cfg.model.get("dual", {})
        if not dual.get("high_noise_path") or not dual.get("low_noise_path"):
            raise ValueError(
                "model.type=wan22_dual requires model.dual.high_noise_path "
                "and model.dual.low_noise_path to be set"
            )

    if cfg.lora.rank < 1 or cfg.lora.rank > 512:
        raise ValueError(f"lora.rank must be in [1, 512], got {cfg.lora.rank}")

    if cfg.lora.alpha <= 0:
        raise ValueError(f"lora.alpha must be > 0, got {cfg.lora.alpha}")

    if cfg.lora.dropout < 0 or cfg.lora.dropout >= 1:
        raise ValueError(f"lora.dropout must be in [0, 1), got {cfg.lora.dropout}")

    valid_strategies = {"auto", "single", "ddp", "fsdp", "deepspeed"}
    if cfg.distributed.strategy not in valid_strategies:
        raise ValueError(
            f"distributed.strategy must be one of {valid_strategies}"
        )

    valid_dtypes = {"auto", "bf16", "fp16", "fp32"}
    if cfg.model.dtype not in valid_dtypes:
        raise ValueError(f"model.dtype must be one of {valid_dtypes}")

    valid_schedulers = {"cosine", "linear", "constant", "cosine_with_restarts", "polynomial"}
    if cfg.training.lr_scheduler not in valid_schedulers:
        raise ValueError(f"training.lr_scheduler must be one of {valid_schedulers}")

    sample_cfg = cfg.get("sample", {})
    if sample_cfg and sample_cfg.get("prompts"):
        import logging as _logging
        _log = _logging.getLogger(__name__)
        model_type = cfg.model.type
        if model_type == "wan_i2v":
            _log.warning(
                "sample.prompts set but model_type=wan_i2v requires a source frame; "
                "samples will be skipped during training."
            )


def resolve_auto_values(cfg: DictConfig, hardware_info: Dict[str, Any]) -> DictConfig:
    """Replace 'auto' values with hardware-appropriate values."""
    cfg = OmegaConf.to_container(cfg, resolve=True)

    num_gpus = hardware_info.get("num_gpus", 1)
    vram_per_gpu_list = hardware_info.get("vram_per_gpu_gb", [16])
    vram_per_gpu = min(vram_per_gpu_list) if isinstance(vram_per_gpu_list, list) else vram_per_gpu_list
    has_bf16 = hardware_info.get("has_bf16", True)
    cpu_cores = hardware_info.get("cpu_cores", 8)

    # Precision
    if cfg["model"]["dtype"] == "auto":
        cfg["model"]["dtype"] = "bf16" if has_bf16 else "fp16"

    if cfg["training"]["mixed_precision"] == "auto":
        cfg["training"]["mixed_precision"] = "bf16" if has_bf16 else "fp16"

    # Distributed strategy
    if cfg["distributed"]["strategy"] == "auto":
        if num_gpus == 1:
            cfg["distributed"]["strategy"] = "single"
        elif num_gpus <= 4:
            cfg["distributed"]["strategy"] = "ddp"
        else:
            cfg["distributed"]["strategy"] = "fsdp"

    # torch.compile
    if cfg["performance"]["compile"] == "auto":
        cfg["performance"]["compile"] = True

    # Flash attention
    if cfg["performance"]["flash_attention"] == "auto":
        cfg["performance"]["flash_attention"] = hardware_info.get("has_flash_attn", False)

    if cfg["performance"]["xformers"] == "auto":
        cfg["performance"]["xformers"] = hardware_info.get("has_xformers", False)

    # Gradient checkpointing — disable at 32 GB+ unless the model is qwen_edit,
    # which has 20B params and long token sequences that OOM without recomputation.
    if cfg["performance"]["gradient_checkpointing"] == "auto":
        model_type_for_ckpt = cfg["model"].get("type", "")
        if model_type_for_ckpt == "qwen_edit":
            cfg["performance"]["gradient_checkpointing"] = True
        else:
            cfg["performance"]["gradient_checkpointing"] = vram_per_gpu < 32

    model_type = cfg["model"].get("type", "")

    # Dual-model low_vram: auto-resolve based on whether both 14B transformers fit in VRAM.
    # Not user-settable — derived purely from hardware.
    if model_type == "wan22_dual":
        vram_per_model_gb = 28   # conservative for WAN 14B bf16
        total_vram = (
            sum(vram_per_gpu_list) if isinstance(vram_per_gpu_list, list) else vram_per_gpu_list
        )
        cfg["model"]["_low_vram"] = total_vram < vram_per_model_gb * 2
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "wan22_dual: total_vram=%.0fGB → _low_vram=%s",
            total_vram, cfg["model"]["_low_vram"],
        )

    # Batch size and gradient accumulation
    if cfg["training"]["batch_size"] == "auto":
        if model_type == "qwen_edit":
            # 20B model with long token sequences (4096 tokens at 1024px) —
            # attention activation memory dominates; batch=1 is the safe ceiling
            # until flash attention is available on sm_120.
            cfg["training"]["batch_size"] = 1
        elif vram_per_gpu >= 96:        # RTX 6000 Blackwell (96 GB)
            cfg["training"]["batch_size"] = 8
        elif vram_per_gpu >= 80:        # A100 80 GB
            cfg["training"]["batch_size"] = 4
        elif vram_per_gpu >= 32:        # RTX 5090 (32 GB), A40/A6000 (48 GB)
            cfg["training"]["batch_size"] = 2
        else:
            cfg["training"]["batch_size"] = 1
    if cfg["training"]["gradient_accumulation_steps"] == "auto":
        cfg["training"]["gradient_accumulation_steps"] = 1

    # Data workers
    if cfg["data"]["num_workers"] == "auto":
        cfg["data"]["num_workers"] = min(cpu_cores // 2, 8)

    if cfg["data"]["prefetch_factor"] == "auto":
        cfg["data"]["prefetch_factor"] = 2

    return OmegaConf.create(cfg)


class TrainingConfig:
    """Resolved training configuration with convenience accessors."""

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        return cls(load_config(path))

    @property
    def model(self) -> DictConfig:
        return self._cfg.model

    @property
    def lora(self) -> DictConfig:
        return self._cfg.lora

    @property
    def training(self) -> DictConfig:
        return self._cfg.training

    @property
    def distributed(self) -> DictConfig:
        return self._cfg.distributed

    @property
    def performance(self) -> DictConfig:
        return self._cfg.performance

    @property
    def data(self) -> DictConfig:
        return self._cfg.data

    @property
    def logging(self) -> DictConfig:
        return self._cfg.logging

    def resolve(self, hardware_info: Dict[str, Any]) -> "TrainingConfig":
        """Return new config with all 'auto' values resolved."""
        resolved_cfg = resolve_auto_values(self._cfg, hardware_info)
        return TrainingConfig(resolved_cfg)

    def to_dict(self) -> dict:
        return OmegaConf.to_container(self._cfg, resolve=True)

    def __repr__(self) -> str:
        return f"TrainingConfig(\n{OmegaConf.to_yaml(self._cfg)})"
