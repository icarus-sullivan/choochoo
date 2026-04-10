"""WAN 2.2 model adapter with DiT/temporal support and FSDP-aware wrapping."""

from __future__ import annotations

import functools
import importlib
import importlib.machinery
import logging
import sys
from types import ModuleType
from typing import Any, Dict, List, Optional

# Only install flash_attn stubs if the real package is absent or ABI-broken.
# On compatible hardware (sm_120 + PyTorch 2.8 + flash_attn 3.x) the real
# package should be used. A functional flash_attn has a __version__ attribute;
# our stubs and partially-crashed imports do not.
#
# Stubs must survive deep attribute access: diffusers probes sub-attributes like
# `flash_attn_interface.flash_attn_gpu`. _StubAttr handles this by returning
# itself for any getattr and accepting (and ignoring) any call.
class _StubAttr:
    def __getattr__(self, _: str) -> "_StubAttr":
        return self
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None
    def __bool__(self) -> bool:
        return False
    # xformers reads flash_attn.__version__ and passes it to packaging.version.parse()
    # which requires a string. Return a parseable sentinel instead of self.
    def __str__(self) -> str:
        return "0.0.0"
    def __repr__(self) -> str:
        return "0.0.0"


def _make_stub_module(name: str) -> ModuleType:
    """Create a sys.modules stub that survives deep attribute/submodule probing."""
    is_pkg = "." not in name
    stub = ModuleType(name)
    stub.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=is_pkg)
    if is_pkg:
        stub.__path__ = []  # type: ignore[assignment]
    # __file__ must be a truthy string — see comment in flash_attn block below.
    stub.__file__ = f"<{name}_stub>"
    stub.__getattr__ = lambda attr: _StubAttr()  # type: ignore[method-assign]
    return stub


_needs_flash_stub = not hasattr(sys.modules.get("flash_attn"), "__version__")
if _needs_flash_stub:
    for _mod_name in (
        "flash_attn",
        "flash_attn.flash_attn_interface",
        "flash_attn.flash_attn_utils",
        "flash_attn_2",
        "flash_attn_2_cuda",
        "flash_attn_2_flash_fwd",
    ):
        # is_package=True sets submodule_search_locations=[] on the spec.
        # Without it, submodule_search_locations is None and Python classifies
        # the module as a built-in, raising "<module 'flash_attn'> is a
        # built-in module" when anything tries to descend into it.
        #
        # __file__ must be a truthy string. inspect.getfile() raises TypeError
        # for any module with __file__=None/missing, and torch.package patches
        # inspect.getfile without re-catching TypeError, causing inspect.getmodule
        # (which iterates all of sys.modules) to crash when bitsandbytes uses
        # @register_fake → inspect.getframeinfo during diffusers import.
        sys.modules[_mod_name] = _make_stub_module(_mod_name)
    del _mod_name
    # sys.modules stubs satisfy `import flash_attn.x` lookups but NOT attribute
    # access (`flash_attn.flash_attn_interface`). Set both.
    sys.modules["flash_attn"].flash_attn_interface = sys.modules["flash_attn.flash_attn_interface"]
    sys.modules["flash_attn"].flash_attn_utils = sys.modules["flash_attn.flash_attn_utils"]
    # xformers (and diffusers) read flash_attn.__version__ directly by attribute.
    # __getattr__ fallback returns _StubAttr, which parse_version() rejects.
    # Set a real string so version comparisons always succeed.
    sys.modules["flash_attn"].__version__ = "0.0.0"  # type: ignore[attr-defined]
del _needs_flash_stub

# xformers is often ABI-incompatible (e.g. built for PyTorch 2.10+cu128 but
# running on 2.6+cu124). Stub the entire package so its module-level C extension
# imports never execute, then tell diffusers it is unavailable.
_needs_xformers_stub = not hasattr(sys.modules.get("xformers"), "__version__")
if _needs_xformers_stub:
    for _mod_name in (
        "xformers",
        "xformers.ops",
        "xformers.ops.fmha",
        "xformers.ops.fmha.attn_bias",
    ):
        sys.modules.setdefault(_mod_name, _make_stub_module(_mod_name))
    del _mod_name
    sys.modules["xformers"].ops = sys.modules["xformers.ops"]
    sys.modules["xformers.ops"].fmha = sys.modules["xformers.ops.fmha"]
    sys.modules["xformers"].__version__ = "0.0.0"  # type: ignore[attr-defined]
del _needs_xformers_stub

# torchvision._meta_registrations: on older torchvision builds the module accesses
# torchvision.extension at decoration time (module-level code), before
# torchvision.__init__ has set that attribute, causing a circular-import
# AttributeError. An empty stub lets torchvision initialize cleanly; basic
# transforms still work and only custom-op meta-tensor registration is skipped.
# Only install if torchvision is not already fully loaded.
if not hasattr(sys.modules.get("torchvision"), "transforms"):
    _stub = ModuleType("torchvision._meta_registrations")
    _stub.__spec__ = importlib.machinery.ModuleSpec("torchvision._meta_registrations", None)
    sys.modules["torchvision._meta_registrations"] = _stub
    del _stub

import diffusers.utils.import_utils as _diffusers_utils
_diffusers_utils.is_flash_attn_available = lambda: False  # type: ignore[attr-defined]
# Patch the underlying bool too — some diffusers modules capture it directly
# at import time rather than calling is_flash_attn_available() at use time.
if hasattr(_diffusers_utils, "_flash_attn_available"):
    _diffusers_utils._flash_attn_available = False
# Disable xformers — the installed wheel may be ABI-incompatible.
_diffusers_utils.is_xformers_available = lambda: False  # type: ignore[attr-defined]
if hasattr(_diffusers_utils, "_xformers_available"):
    _diffusers_utils._xformers_available = False
importlib.invalidate_caches()

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .base import BaseModelAdapter

logger = logging.getLogger(__name__)

# Hardcoded WAN 2.2 dual-model noise boundaries (not user-configurable)
_WAN_BOUNDARY_T2V = 0.875
_WAN_BOUNDARY_I2V = 0.9


def wan_collate_fn(batch, min_t: float = 0.001, max_t: float = 0.999):
    """Collate function for WAN T2V and I2V datasets.

    Stacks tensors from VideoDataset/ImageDataset and injects logit-normal
    timestep samples in [min_t, max_t]. Flow-matching requires floats in [0,1],
    not DDPM integers in [0,1000].

    min_t / max_t are hardcoded by the adapter's noise_regime — never user-set.
    """
    from typing import Any, Dict
    collated: Dict[str, Any] = {}
    for key in batch[0]:
        vals = [item[key] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            collated[key] = torch.stack(vals)
        else:
            collated[key] = vals
    b = len(batch)
    collated["timesteps"] = torch.sigmoid(torch.randn(b)).clamp(min_t, max_t)
    return collated


class WANAdapter(BaseModelAdapter):
    """Adapter for WAN 2.1 and 2.2 T2V (video DiT architecture).

    WAN uses flow-matching with a DiT backbone:
    - Patchified spatial + temporal tokens
    - Full 3D attention with rotary position embeddings
    - Text conditioning via T5 cross-attention
    - Logit-normal timestep sampling in [0,1]

    Works with both WAN 2.1 (Wan-AI/Wan2.1-T2V-*) and WAN 2.2 (Wan-AI/Wan2.2-T2V-*)
    weights — both use the same WanPipeline class in diffusers. The only runtime
    difference is VAE normalization, handled automatically via buffer detection.
    """

    # Actual diffusers WanTransformer3DModel layer names

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._lora_injector = None

    def load_model(self) -> nn.Module:
        pretrained_path = self.cfg.model.pretrained_path
        dtype = self.prepare_dtype()

        logger.info(f"Loading WAN from: {pretrained_path}")

        try:
            from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanPipeline
            from diffusers import UniPCMultistepScheduler

            pipe = WanPipeline.from_pretrained(
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
            logger.warning(f"Could not load via WanPipeline: {e}. Trying generic DiT load.")
            self.model = self._load_generic_dit(pretrained_path, dtype)

        # Freeze everything except LoRA (injected later)
        self.model.requires_grad_(False)
        if self.vae is not None:
            self.vae.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)

        logger.info(f"WAN loaded: {self.param_count/1e9:.2f}B params")
        return self.model

    def _load_generic_dit(self, path: str, dtype: torch.dtype) -> nn.Module:
        """Fallback: load transformer checkpoint directly."""
        from safetensors.torch import load_file
        import os

        model = _WAN22DiT(
            hidden_dim=1536,
            num_heads=24,
            num_layers=28,
            patch_size=2,
            video_max_frames=64,
        ).to(dtype=dtype)

        # Try loading weights
        ckpt_path = os.path.join(path, "transformer", "diffusion_pytorch_model.safetensors")
        if os.path.exists(ckpt_path):
            sd = load_file(ckpt_path)
            model.load_state_dict(sd, strict=False)
            logger.info(f"Loaded DiT weights from {ckpt_path}")
        else:
            logger.warning(f"No weights found at {ckpt_path}, using random init")

        return model

    def inject_lora(self, injector: Any) -> None:
        """Inject LoRA into attention and MLP layers of the DiT."""
        self._lora_injector = injector

        injector.target_modules = self._resolve_target_modules()

        if self.model is None:
            raise RuntimeError("Call load_model() before inject_lora()")

        injector.inject(self.model)
        logger.info(
            f"LoRA injected: {injector.num_injected()} layers, "
            f"rank={injector.rank}, alpha={injector.alpha}"
        )
        logger.info(
            f"Trainable params: {self.trainable_param_count/1e6:.1f}M / "
            f"{self.param_count/1e9:.2f}B total"
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """WAN flow-matching training forward pass.

        Flow-matching target: velocity v = noise - x_0.
        Timesteps are logit-normal floats in [0,1] (sampled by WAN collate_fn).
        Supports dual-LoRA: splits batch by noise level, runs shared forward.
        """
        latents = batch["latents"]
        encoder_hidden_states = batch["encoder_hidden_states"]
        timesteps = batch["timesteps"]

        dual = self.cfg.lora.get("dual_lora", False)

        if dual and latents.shape[0] >= 2:
            return self._dual_forward(latents, encoder_hidden_states, timesteps, batch)

        noisy, noise, t = self._add_noise(latents, timesteps)

        # Velocity target for flow-matching: v = noise - x_0
        target = noise - latents

        pred = self.model(
            noisy,
            timestep=t,
            encoder_hidden_states=encoder_hidden_states,
        )
        # Transformers/DiT return a dataclass; extract .sample if present
        if hasattr(pred, "sample"):
            pred = pred.sample

        logger.debug(
            "forward: latent=%s t_range=[%.4f, %.4f]",
            tuple(latents.shape), t.min().item(), t.max().item(),
        )

        loss = F.mse_loss(pred, target)
        return {"loss": loss, "pred": pred}

    def _dual_forward(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Single forward pass for dual-LoRA training (high + low noise).

        Splits batch in half: first half = high noise [0.5, 0.999],
        second half = low noise [0.001, 0.5]. Both use flow-matching velocity target.
        Backprop once through combined loss.
        """
        b = latents.shape[0]
        half = b // 2

        # High-noise half: logit-normal floats biased toward 1 (t in [0.5, 0.999])
        t_high = torch.sigmoid(torch.randn(half, device=latents.device)).clamp(0.5, 0.999)
        # Low-noise half: logit-normal floats biased toward 0 (t in [0.001, 0.5])
        t_low = torch.sigmoid(torch.randn(half, device=latents.device)).clamp(0.001, 0.5)

        noisy_high, noise_high, _ = self._add_noise(latents[:half], t_high)
        noisy_low, noise_low, _ = self._add_noise(latents[half:], t_low)

        target_high = noise_high - latents[:half]
        target_low = noise_low - latents[half:]

        # Shared forward (single call) with combined batch
        noisy_combined = torch.cat([noisy_high, noisy_low], dim=0)
        t_combined = torch.cat([t_high, t_low], dim=0)
        enc_combined = torch.cat(
            [encoder_hidden_states[:half], encoder_hidden_states[half:]], dim=0
        )

        pred_combined = self.model(
            noisy_combined,
            timestep=t_combined,
            encoder_hidden_states=enc_combined,
        )
        if hasattr(pred_combined, "sample"):
            pred_combined = pred_combined.sample

        pred_high = pred_combined[:half]
        pred_low = pred_combined[half:]

        # Velocity loss for each regime
        loss_high = F.mse_loss(pred_high, target_high)
        loss_low = F.mse_loss(pred_low, target_low)

        w_high = self.cfg.lora.get("high_noise_weight", 1.0)
        w_low = self.cfg.lora.get("low_noise_weight", 1.0)
        loss = w_high * loss_high + w_low * loss_low

        return {
            "loss": loss,
            "loss_high": loss_high.detach(),
            "loss_low": loss_low.detach(),
            "pred": pred_combined,
        }

    def loss(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> torch.Tensor:
        pred = outputs["pred"]
        target = outputs.get("target", batch.get("target", pred))
        return F.mse_loss(pred, target)

    def _add_noise(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        """Flow-matching interpolation: x_t = (1-t)*x_0 + t*noise.

        timesteps must be floats in [0,1] (logit-normal sampled by WAN collate_fn).
        """
        noise = torch.randn_like(latents)
        t = timesteps.float().view(-1, *([1] * (latents.ndim - 1)))
        noisy = (1.0 - t) * latents + t * noise
        return noisy, noise, timesteps

    def get_fsdp_wrap_policy(self):
        """WAN-specific FSDP wrapping: wrap at DiT block level."""
        from torch.distributed.fsdp.wrap import (
            ModuleWrapPolicy,
            size_based_auto_wrap_policy,
        )

        # Prefer block-level wrapping for WAN DiT
        block_classes = self._get_block_classes()
        if block_classes:
            return ModuleWrapPolicy(block_classes)

        # Fallback: size-based
        min_params = int(self.cfg.distributed.fsdp.get("min_params", 1e7))
        return functools.partial(size_based_auto_wrap_policy, min_num_params=min_params)

    def _get_block_classes(self) -> set:
        """Collect DiT block classes from model for FSDP wrapping."""
        classes = set()
        for module in self.model.modules():
            name = type(module).__name__
            if any(kw in name for kw in ("Block", "Layer", "TransformerBlock")):
                if sum(p.numel() for p in module.parameters()) > 1e6:
                    classes.add(type(module))
        return classes

    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode video frames to latents using VAE.

        Applies per-channel mean/std normalization if the VAE exposes
        latents_mean_tensor / latents_std_tensor buffers (WAN 2.1 / 2.2 3D VAE).
        Falls back to scalar scaling_factor if those buffers are absent.
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded")
        with torch.no_grad():
            vae_param = next(self.vae.parameters())
            frames = frames.to(device=vae_param.device, dtype=vae_param.dtype)
            # frames: [B, C, T, H, W]
            b, c, t, h, w = frames.shape
            frames_flat = frames.view(b * t, c, h, w)
            latents_flat = self.vae.encode(frames_flat).latent_dist.sample()
            # Reshape back to video dims before normalization
            _, lc, lh, lw = latents_flat.shape
            latents = latents_flat.view(b, t, lc, lh, lw).permute(0, 2, 1, 3, 4)
            # Normalize: prefer per-channel buffers (WAN-style) over scalar scaling_factor
            if hasattr(self.vae, "latents_mean_tensor") and hasattr(self.vae, "latents_std_tensor"):
                mean = self.vae.latents_mean_tensor.to(device=latents.device, dtype=latents.dtype)
                std = self.vae.latents_std_tensor.to(device=latents.device, dtype=latents.dtype)
                latents = (latents - mean) / std
                logger.debug("encode_video: applied VAE mean/std normalization")
            else:
                sf = getattr(self.vae.config, "scaling_factor", 1.0)
                latents = latents * sf
                logger.debug("encode_video: applied scaling_factor=%.4f", sf)
            return latents

    def _get_timestep_range(self) -> tuple:
        """Resolve (min_t, max_t) from noise_regime config.

        noise_regime values:
          "high" → [boundary, 0.999]  — coarse-structure gradients only
          "low"  → [0.001, boundary]  — fine-detail gradients only
          "both" / "auto" → [0.001, 0.999]  — full range (default)

        Boundary is _WAN_BOUNDARY_T2V (0.875); I2V subclass overrides to _WAN_BOUNDARY_I2V.
        """
        regime = str(self.cfg.model.get("noise_regime", "auto")).lower()
        if regime == "auto":
            regime = "both"
        boundary = _WAN_BOUNDARY_T2V
        if regime == "high":
            return (boundary, 0.999)
        elif regime == "low":
            return (0.001, boundary)
        else:  # both
            return (0.001, 0.999)

    def get_collate_fn(self):
        min_t, max_t = self._get_timestep_range()
        logger.debug(
            "WANAdapter.get_collate_fn: noise_regime=%s → t∈[%.3f, %.3f]",
            self.cfg.model.get("noise_regime", "auto"), min_t, max_t,
        )
        return functools.partial(wan_collate_fn, min_t=min_t, max_t=max_t)

    def sample(self, sample_cfg) -> Optional[list]:
        from diffusers import WanPipeline

        s = sample_cfg
        results = []
        try:
            if self.vae is not None:
                self.vae = self.vae.to("cuda")
            if self.text_encoder is not None:
                self.text_encoder = self.text_encoder.to("cuda")
            pipe = WanPipeline(
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

    def encode_text(self, prompts: list) -> torch.Tensor:
        """Encode text prompts to embeddings."""
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not loaded")
        device = next(self.text_encoder.parameters()).device
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            return self.text_encoder(**tokens).last_hidden_state


# Backward-compatible alias — existing configs using wan22 still work
WAN22Adapter = WANAdapter


class _WAN22DiT(nn.Module):
    """Minimal WAN 2.2 DiT stub for testing without pretrained weights."""

    def __init__(
        self,
        hidden_dim: int = 1536,
        num_heads: int = 24,
        num_layers: int = 28,
        patch_size: int = 2,
        video_max_frames: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_embed = nn.Linear(patch_size * patch_size * 16, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.blocks = nn.ModuleList([
            _DiTBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, patch_size * patch_size * 16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self._timestep_embedding(timestep, 256).to(hidden_states.dtype)
        t_emb = self.time_embed(t_emb)

        # Flatten spatial dims for transformer
        b = hidden_states.shape[0]
        x = hidden_states.reshape(b, hidden_states.shape[1], -1).permute(0, 2, 1)
        x = self.patch_embed(x)

        for block in self.blocks:
            x = block(x, t_emb, encoder_hidden_states)

        x = self.norm_out(x)
        x = self.proj_out(x)
        return x.permute(0, 2, 1).reshape(hidden_states.shape)

    @staticmethod
    def _timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32, device=timesteps.device)
            * (torch.log(torch.tensor(10000.0)) / half)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.adaLN = nn.Linear(hidden_dim, 6 * hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale_shift = self.adaLN(t_emb).unsqueeze(1).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = scale_shift

        # Self-attention
        x_norm = self.norm1(x) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa * attn_out

        # Cross-attention
        if context is not None:
            x_norm2 = self.norm2(x)
            cross_out, _ = self.cross_attn(x_norm2, context, context)
            x = x + cross_out

        # MLP
        x_norm3 = self.norm3(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_norm3)

        return x
