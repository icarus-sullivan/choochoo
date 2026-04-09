"""WAN Image-to-Video adapter for fine-tuning I2V models."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from .wan22 import WANAdapter, _WAN_BOUNDARY_I2V

logger = logging.getLogger(__name__)

# WAN I2V transformer expects 36-channel input:
#   channels  0–15 : noisy video latents
#   channels 16–31 : VAE-encoded first frame (expanded across all T frames)
#   channels 32–35 : binary mask (1.0 at frame 0 = conditioned, 0.0 elsewhere)
_I2V_IN_CHANNELS = 36
_I2V_NOISE_CH    = 16   # noisy latent channels
_I2V_SRC_CH      = 16   # source frame latent channels
_I2V_MASK_CH     =  4   # conditioning mask channels


class WANi2vAdapter(WANAdapter):
    """Adapter for WAN I2V (Image-to-Video) models.

    WAN I2V conditions the video generation on a source image by encoding it via the
    same VAE and building a 36-channel model input:
      [noisy_video (16) | src_frame_latent (16) | mask (4)]

    This matches the channel layout the pretrained I2V transformer was trained with.
    CLIP is not used for conditioning — VAE latents only (confirmed by ai-toolkit).

    Compatible with:
      - Wan-AI/Wan2.1-I2V-14B-Diffusers
      - Wan-AI/Wan2.2-I2V-A14B-Diffusers
    """

    def load_model(self):
        """Load WAN I2V pipeline via WanImageToVideoPipeline."""
        pretrained_path = self.cfg.model.pretrained_path
        dtype = self.prepare_dtype()

        logger.info(f"Loading WAN I2V from: {pretrained_path}")

        try:
            from diffusers import WanImageToVideoPipeline

            pipe = WanImageToVideoPipeline.from_pretrained(
                pretrained_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            self.model = pipe.transformer
            self.vae = pipe.vae
            self.text_encoder = pipe.text_encoder
            self.tokenizer = pipe.tokenizer
            self.noise_scheduler = pipe.scheduler

        except (ImportError, Exception) as e:
            logger.warning(
                f"WanImageToVideoPipeline load failed: {e}. "
                "Falling back to WanPipeline (T2V weights, I2V forward pass)."
            )
            super().load_model()
            return self.model

        self.model.requires_grad_(False)
        if self.vae is not None:
            self.vae.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)

        logger.info(f"WAN I2V loaded: {self.param_count/1e9:.2f}B params")
        return self.model

    def _get_timestep_range(self) -> tuple:
        """I2V uses boundary 0.9 instead of 0.875 for regime clamping."""
        regime = str(self.cfg.model.get("noise_regime", "auto")).lower()
        if regime == "auto":
            regime = "both"
        boundary = _WAN_BOUNDARY_I2V
        if regime == "high":
            return (boundary, 0.999)
        elif regime == "low":
            return (0.001, boundary)
        else:
            return (0.001, 0.999)

    def encode_source_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a source image to VAE latents for I2V conditioning.

        Args:
            image: (B, C, H, W) pixel tensor in [-1, 1] on any device.
        Returns:
            (B, C_lat, H_lat, W_lat) latent tensor.
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded")
        with torch.no_grad():
            vae_param = next(self.vae.parameters())
            image = image.to(device=vae_param.device, dtype=vae_param.dtype)
            # WAN VAE expects 5D (B, C, T, H, W); lift single frame to T=1
            img5d = image.unsqueeze(2)  # (B, C, 1, H, W)
            latents = self.vae.encode(img5d).latent_dist.sample()
            latents = latents.squeeze(2)  # (B, C_lat, H_lat, W_lat)
            # Per-channel mean/std normalization (same as encode_video)
            if hasattr(self.vae, "latents_mean_tensor") and hasattr(self.vae, "latents_std_tensor"):
                mean = self.vae.latents_mean_tensor.to(device=latents.device, dtype=latents.dtype)
                std = self.vae.latents_std_tensor.to(device=latents.device, dtype=latents.dtype)
                latents = (latents - mean) / std
            else:
                sf = getattr(self.vae.config, "scaling_factor", 1.0)
                latents = latents * sf
            return latents

    def _build_i2v_model_input(
        self, noisy: torch.Tensor, source_latents: torch.Tensor
    ) -> torch.Tensor:
        """Build 36-channel I2V model input.

        Channel layout (from ai-toolkit wan22_14b_i2v_model.py):
          channels  0–15 : noisy video latents        (B, 16, T, H, W)
          channels 16–31 : source frame latent        (B, 16, T, H, W) — first frame expanded
          channels 32–35 : conditioning mask          (B,  4, T, H, W) — 1.0 at T=0

        Args:
            noisy:          (B, 16, T, H, W)
            source_latents: (B, 16, H, W)   — spatial-only, no temporal dim
        Returns:
            (B, 36, T, H, W)
        """
        B, C, T, H, W = noisy.shape
        # Expand source frame to match all temporal positions
        src = source_latents.unsqueeze(2).expand(-1, -1, T, -1, -1)  # (B, 16, T, H, W)
        # Mask: 1.0 for the conditioned frame (T=0), 0.0 elsewhere
        mask = torch.zeros(B, _I2V_MASK_CH, T, H, W, device=noisy.device, dtype=noisy.dtype)
        mask[:, :, 0] = 1.0
        model_input = torch.cat([noisy, src, mask], dim=1)   # (B, 36, T, H, W)
        logger.debug(
            "_build_i2v_model_input: noisy=%s src=%s → input=%s",
            tuple(noisy.shape), tuple(source_latents.shape), tuple(model_input.shape),
        )
        return model_input

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """WAN I2V flow-matching forward pass with 36-channel source conditioning."""
        latents = batch["latents"]               # (B, C, T, H, W)
        encoder_hidden_states = batch["encoder_hidden_states"]
        timesteps = batch["timesteps"]           # logit-normal floats [0,1]

        noisy, noise, t = self._add_noise(latents, timesteps)

        # Velocity target for flow-matching
        target = noise - latents

        # --- Source image conditioning ---
        source_latents = batch.get("source_latents")
        if source_latents is None and "source_frame" in batch:
            # Encode on-the-fly from raw pixel frame (C, H, W) or (B, C, H, W)
            src_frame = batch["source_frame"]
            if src_frame.ndim == 3:
                src_frame = src_frame.unsqueeze(0)
            source_latents = self.encode_source_image(src_frame)

        if source_latents is not None:
            model_input = self._build_i2v_model_input(noisy, source_latents)
        else:
            # Fallback: no source conditioning (T2V-style, wrong for I2V but safe)
            logger.warning(
                "WANi2vAdapter.forward: no source_latents or source_frame in batch — "
                "training without image conditioning. Check dataset has source_frame."
            )
            model_input = noisy

        pred = self.model(
            model_input,
            timestep=t,
            encoder_hidden_states=encoder_hidden_states,
        )
        if hasattr(pred, "sample"):
            pred = pred.sample

        loss = F.mse_loss(pred, target)
        return {"loss": loss, "pred": pred}
