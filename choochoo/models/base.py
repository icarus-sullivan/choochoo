"""Abstract base class for model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig


class BaseModelAdapter(ABC):
    """Interface every model backend must implement.

    Adapters are responsible for:
    - Loading the pretrained model
    - Exposing it for LoRA injection
    - Defining the forward pass for training
    - Computing the training loss
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model: Optional[nn.Module] = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None

    @abstractmethod
    def load_model(self) -> nn.Module:
        """Load and return the trainable model (UNet/DiT/transformer)."""
        ...

    @abstractmethod
    def inject_lora(self, injector: Any) -> None:
        """Inject LoRA adapters into the loaded model."""
        ...

    @abstractmethod
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Run forward pass. Returns dict with at least 'loss' and 'pred'."""
        ...

    @abstractmethod
    def loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
        """Compute scalar loss from forward outputs."""
        ...

    def get_trainable_params(self):
        """Return parameters that require grad (LoRA only by default)."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_collate_fn(self):
        """Return a custom batch collate function, or None for PyTorch default.

        Adapters that need custom collation (timestep injection, latent caching, etc.)
        should override this. The DataPipeline calls this instead of branching on model_type.
        """
        return None

    def sample(self, sample_cfg: Any) -> Optional[list]:
        """Generate inference samples at a training checkpoint.

        Returns a list of result dicts, one per prompt:
            {"mime": "video/mp4", "data": [PIL.Image, ...]}   # video frames
            {"mime": "image/png", "data": PIL.Image}           # single image

        Return None if this adapter does not support sampling (default).
        """
        return None

    def prepare_dtype(self) -> torch.dtype:
        dtype_str = self.cfg.model.dtype
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            # "auto" means resolve_auto_values hasn't run yet (model loaded before
            # hardware detection). Default to bfloat16 — safe on all modern GPUs
            # and 2× more memory-efficient than float32.
            "auto": torch.bfloat16,
        }.get(dtype_str, torch.bfloat16)

    def enable_gradient_checkpointing(self, selective: bool = True) -> None:
        """Enable gradient checkpointing.

        When selective=True, only checkpoints the largest attention-heavy blocks
        (DiT layers, TransformerBlocks) rather than every module. This yields
        most of the memory savings at lower recomputation cost vs global checkpointing.
        """
        if selective:
            patched = self._apply_selective_checkpointing(self.model)
            if patched > 0:
                import logging
                logging.getLogger(__name__).info(
                    f"Selective gradient checkpointing: patched {patched} large blocks"
                )
                return
        # Fallback: full model checkpointing via HF/diffusers API
        if hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing()
        elif hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def _apply_selective_checkpointing(self, model: nn.Module, min_params: int = 5_000_000) -> int:
        """Wrap large blocks with torch.utils.checkpoint. Returns count of patched modules."""
        import torch.utils.checkpoint as ckpt

        patched = 0
        for name, module in model.named_modules():
            if module is model:
                continue
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            param_count += sum(p.numel() for p in module.parameters())
            if param_count < min_params:
                continue
            cls_name = type(module).__name__
            is_block = any(
                kw in cls_name
                for kw in ("Block", "Layer", "EncoderLayer", "DecoderLayer", "TransformerBlock")
            ) or (hasattr(module, "attn") and hasattr(module, "mlp"))
            if not is_block:
                continue
            original_forward = module.forward

            def make_checkpointed(fwd):
                def checkpointed_forward(*args, **kwargs):
                    # use_reentrant=False supports both *args and **kwargs
                    return ckpt.checkpoint(fwd, *args, use_reentrant=False, **kwargs)
                return checkpointed_forward

            module.forward = make_checkpointed(original_forward)
            patched += 1

        return patched

    def enable_flash_attention(self) -> None:
        """Enable flash attention if the model supports it."""
        if hasattr(self.model, "set_use_flash_attention_2"):
            self.model.set_use_flash_attention_2(True)

    def get_fsdp_wrap_policy(self):
        """Return FSDP auto-wrap policy. Override in subclasses for model-specific wrapping."""
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=int(self.cfg.distributed.fsdp.min_params),
        )

    @property
    def param_count(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    @property
    def trainable_param_count(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
