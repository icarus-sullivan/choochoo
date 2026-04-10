"""LTX-2 model adapter stub — designed for clean future plug-in."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .base import BaseModelAdapter

logger = logging.getLogger(__name__)


class LTX2Adapter(BaseModelAdapter):
    """Stub adapter for LTX-2 video generation model.

    Designed to be filled in when LTX-2 becomes publicly available.
    Architecture is expected to be a causal video DiT with:
    - Temporal causal masking
    - Flow matching training objective
    - T5 text encoder
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        logger.warning(
            "LTX2Adapter is a stub. Full implementation pending model release."
        )

    def load_model(self) -> nn.Module:
        logger.info("LTX-2: attempting to load via diffusers (if available)")
        try:
            from diffusers import LTXVideoTransformer3DModel, LTXPipeline

            pipe = LTXPipeline.from_pretrained(
                self.cfg.model.pretrained_path,
                torch_dtype=self.prepare_dtype(),
            )
            self.model = pipe.transformer
            self.vae = pipe.vae
            self.text_encoder = pipe.text_encoder
            self.tokenizer = pipe.tokenizer
            self.noise_scheduler = pipe.scheduler
        except Exception as e:
            logger.warning(f"LTX-2 load failed: {e}. Falling back to stub.")
            self.model = _LTX2StubModel()

        self.model.requires_grad_(False)
        return self.model

    def inject_lora(self, injector: Any) -> None:
        if self.model is None:
            raise RuntimeError("Call load_model() before inject_lora()")
        injector.target_modules = self._resolve_target_modules()
        injector.inject(self.model)
        logger.info(f"LTX-2 LoRA injected: {injector.num_injected()} layers")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        latents = batch["latents"]
        encoder_hidden_states = batch.get("encoder_hidden_states")
        timesteps = batch["timesteps"]

        noise = torch.randn_like(latents)
        t = timesteps

        # Flow matching: interpolate between data and noise
        sigma = t.float().view(-1, *([1] * (latents.ndim - 1))) / 1000.0
        noisy = (1 - sigma) * latents + sigma * noise
        velocity_target = noise - latents  # Flow matching target

        pred = self.model(
            noisy,
            timestep=t,
            encoder_hidden_states=encoder_hidden_states,
        )

        loss = self.loss({"pred": pred, "target": velocity_target}, batch)
        return {"loss": loss, "pred": pred}

    def loss(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            outputs["pred"], outputs.get("target", outputs["pred"])
        )

    def sample(self, sample_cfg) -> Optional[list]:
        import torch
        from diffusers import LTXPipeline

        s = sample_cfg
        results = []
        try:
            if self.vae is not None:
                self.vae = self.vae.to("cuda")
            if self.text_encoder is not None:
                self.text_encoder = self.text_encoder.to("cuda")
            pipe = LTXPipeline(
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                transformer=self.model,
                scheduler=self.noise_scheduler,
                vae=self.vae,
            )
            gen = torch.Generator("cuda").manual_seed(int(s.get("seed", 42)))
            for prompt in s.get("prompts", []):
                with torch.inference_mode():
                    frames = pipe(
                        prompt=prompt,
                        height=int(s.get("height", 480)),
                        width=int(s.get("width", 480)),
                        num_frames=int(s.get("num_frames", 42)),
                        num_inference_steps=int(s.get("steps", 20)),
                        guidance_scale=float(s.get("cfg", 4.0)),
                        generator=gen,
                    ).frames[0]
                results.append({"mime": "video/mp4", "data": frames})
        finally:
            if self.vae is not None:
                self.vae = self.vae.to("cpu")
            if self.text_encoder is not None:
                self.text_encoder = self.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        return results


class _LTX2StubModel(nn.Module):
    """Minimal stub for LTX-2 forward pass testing."""

    def __init__(self, hidden_dim: int = 2048, num_heads: int = 32, num_layers: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_in = nn.Linear(16, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, batch_first=True)
            for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(hidden_dim, 16)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
    ) -> torch.Tensor:
        b = x.shape[0]
        flat = x.reshape(b, x.shape[1], -1).permute(0, 2, 1)
        flat = self.proj_in(flat)
        for block in self.blocks:
            flat = block(flat)
        flat = self.proj_out(flat)
        return flat.permute(0, 2, 1).reshape(x.shape)
