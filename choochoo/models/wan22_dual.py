"""WAN 2.2 dual-model adapter: trains high-noise and low-noise transformers together.

Based on ai-toolkit's DualWanTransformer3DModel design:
https://github.com/ostris/ai-toolkit/tree/main/extensions_built_in/diffusion_models/wan22
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .wan22 import WANAdapter, wan_collate_fn, _WAN_BOUNDARY_T2V

logger = logging.getLogger(__name__)


class DualWanTransformer(nn.Module):
    """Routes forward calls to the high-noise or low-noise transformer by mean timestep.

    Mirrors ai-toolkit's DualWanTransformer3DModel. Per-batch routing (not per-sample):
    the entire batch goes to whichever transformer owns the batch's mean timestep.

    When _low_vram=True (auto-resolved from hardware), the inactive model is swapped
    to CPU before the forward call to free activation memory, then stays on CPU until
    the next step. This is slower but allows dual-model training on ≤56 GB VRAM.
    """

    def __init__(
        self,
        high_model: nn.Module,
        low_model: nn.Module,
        boundary: float = _WAN_BOUNDARY_T2V,
        low_vram: bool = False,
    ):
        super().__init__()
        self.transformer_high = high_model
        self.transformer_low = low_model
        self.boundary = boundary
        self.low_vram = low_vram

    def forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor, **kwargs):
        use_high = timestep.float().mean().item() >= self.boundary
        active   = self.transformer_high if use_high else self.transformer_low
        inactive = self.transformer_low  if use_high else self.transformer_high

        if self.low_vram:
            inactive.to("cpu")
            torch.cuda.empty_cache()
            active.to(hidden_states.device)

        logger.debug(
            "DualWanTransformer: routing to %s model (t_mean=%.4f, boundary=%.3f)",
            "high" if use_high else "low",
            timestep.float().mean().item(),
            self.boundary,
        )
        return active(hidden_states, timestep=timestep, **kwargs)

    def parameters(self, recurse: bool = True):
        """Yield parameters from both transformers (for optimizer coverage)."""
        yield from self.transformer_high.parameters(recurse=recurse)
        yield from self.transformer_low.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        yield from self.transformer_high.named_parameters(
            prefix=f"{prefix}transformer_high", recurse=recurse,
        )
        yield from self.transformer_low.named_parameters(
            prefix=f"{prefix}transformer_low", recurse=recurse,
        )


class WANDualAdapter(WANAdapter):
    """Adapter for WAN 2.2 dual-model training (high + low noise checkpoints).

    Loads two separate WanPipeline checkpoints and wraps them in DualWanTransformer.
    LoRA is injected independently into each transformer. A single optimizer trains
    both LoRAs simultaneously. Two safetensors files are saved per checkpoint step.

    Config:
        model.type: wan22_dual
        model.dual.high_noise_path: /path/to/high_noise_transformer
        model.dual.low_noise_path:  /path/to/low_noise_transformer
        model.dual.noise_boundary:  0.875   # t2v default; use 0.9 for i2v
        model.noise_regime: both   # always full range for dual training
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._high_transformer: Optional[nn.Module] = None
        self._low_transformer: Optional[nn.Module] = None
        self._low_injector = None

    def load_model(self) -> nn.Module:
        from diffusers import WanPipeline

        dual = self.cfg.model.dual
        dtype = self.prepare_dtype()
        boundary = float(dual.get("noise_boundary", _WAN_BOUNDARY_T2V))
        low_vram = bool(self.cfg.model.get("_low_vram", False))

        high_path = dual.high_noise_path
        low_path  = dual.low_noise_path

        logger.info(f"Loading WAN dual: high={high_path}")
        high_pipe = WanPipeline.from_pretrained(
            high_path, torch_dtype=dtype, low_cpu_mem_usage=True,
        )
        logger.info(f"Loading WAN dual: low={low_path}")
        low_pipe = WanPipeline.from_pretrained(
            low_path, torch_dtype=dtype, low_cpu_mem_usage=True,
        )

        self._high_transformer = high_pipe.transformer
        self._low_transformer  = low_pipe.transformer
        self._high_transformer.requires_grad_(False)
        self._low_transformer.requires_grad_(False)

        self.model = DualWanTransformer(
            self._high_transformer, self._low_transformer,
            boundary=boundary, low_vram=low_vram,
        )

        # Use high-noise pipeline's VAE/text-encoder (identical across both checkpoints)
        self.vae          = high_pipe.vae.requires_grad_(False)
        self.text_encoder = high_pipe.text_encoder.requires_grad_(False)
        self.tokenizer    = high_pipe.tokenizer
        self.noise_scheduler = high_pipe.scheduler

        logger.info(
            "WAN dual loaded: boundary=%.3f, low_vram=%s, "
            "high=%.2fB params, low=%.2fB params",
            boundary, low_vram,
            sum(p.numel() for p in self._high_transformer.parameters()) / 1e9,
            sum(p.numel() for p in self._low_transformer.parameters()) / 1e9,
        )
        return self.model

    def detect_lora_targets(self) -> list:
        """WAN dual-model LoRA target detection — 1:1 with official DiffSynth targets.

        Probes _high_transformer (both share the same architecture).
        Official targets: q, k, v, o, ffn.0, ffn.2
        """
        import re

        if self._high_transformer is None:
            raise RuntimeError("Call load_model() before detect_lora_targets()")

        patterns = [
            r".*\.q$",      # attention query
            r".*\.k$",      # attention key
            r".*\.v$",      # attention value
            r".*\.o$",      # attention output
            r".*\.ffn\.0$", # feed-forward input projection
            r".*\.ffn\.2$", # feed-forward output projection
        ]

        linear_layers = [
            name for name, module in self._high_transformer.named_modules()
            if isinstance(module, nn.Linear)
        ]

        matched = {p: [] for p in patterns}
        for name in linear_layers:
            for p in patterns:
                if re.search(p, name):
                    matched[p].append(name)

        active_patterns = [p for p, hits in matched.items() if hits]

        logger.info("=== WAN Dual LoRA Target Detection ===")
        for p, hits in matched.items():
            if hits:
                logger.info(f"  {p:35s} -> {len(hits)} layers")
        logger.info(f"  Total Linear layers:   {len(linear_layers)}")
        logger.info(f"  Active LoRA patterns:  {len(active_patterns)}")

        total_hits = sum(len(v) for v in matched.values())
        if total_hits < 20:
            logger.warning("Very low LoRA coverage — check target patterns!")

        return active_patterns

    def inject_lora(self, injector: Any) -> None:
        if self._high_transformer is None or self._low_transformer is None:
            raise RuntimeError("Call load_model() before inject_lora()")

        injector.target_modules = self._resolve_target_modules()

        # High-noise LoRA
        injector.inject(self._high_transformer)

        # Low-noise LoRA — independent weights via deep copy of injector config
        low_injector = copy.deepcopy(injector)
        low_injector.inject(self._low_transformer)
        self._low_injector = low_injector

        high_trainable = sum(
            p.numel() for p in self._high_transformer.parameters() if p.requires_grad
        )
        low_trainable = sum(
            p.numel() for p in self._low_transformer.parameters() if p.requires_grad
        )
        logger.info(
            "WAN dual LoRA injected: high=%dM trainable, low=%dM trainable, "
            "rank=%d, alpha=%s",
            high_trainable // 1_000_000,
            low_trainable // 1_000_000,
            injector.rank,
            injector.alpha,
        )

    def get_trainable_params(self):
        """Return LoRA params from both transformers for the optimizer."""
        params = [p for p in self._high_transformer.parameters() if p.requires_grad]
        params += [p for p in self._low_transformer.parameters() if p.requires_grad]
        return params

    def get_collate_fn(self):
        """Dual training always uses the full timestep range."""
        return wan_collate_fn   # default min_t=0.001, max_t=0.999

    def sample(self, sample_cfg) -> Optional[list]:
        # Use high-noise transformer for sampling (covers full denoising trajectory)
        original_model = self.model
        self.model = self._high_transformer
        try:
            return super().sample(sample_cfg)
        finally:
            self.model = original_model

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.get_trainable_params())
