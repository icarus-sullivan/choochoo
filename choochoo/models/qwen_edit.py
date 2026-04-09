"""Qwen Image Edit adapter for instruction-based image editing."""

from __future__ import annotations

import functools
import importlib
import importlib.machinery
import logging
import sys
from types import ModuleType
from typing import Any, Dict, Optional

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .base import BaseModelAdapter

logger = logging.getLogger(__name__)


class QwenEditAdapter(BaseModelAdapter):
    """Adapter for Qwen-based image editing models.

    Follows WAN-style DiT design:
    - Pipeline-extracted transformer as trainable backbone
    - Explicit latent-space diffusion training loop
    - LoRA injected only into transformer attention + MLP layers
    - Source image conditioning via channel-concatenated latents
    """


    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.tokenizer = None
        self._processor = None
        self._lora_injector = None
        self.latent_channels = 4

    def load_model(self) -> nn.Module:
        pretrained_path = self.cfg.model.pretrained_path
        dtype = self.prepare_dtype()

        logger.info(f"Loading Qwen Image Edit from: {pretrained_path}")

        try:
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(
                pretrained_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.model = pipe.transformer
            self.vae = pipe.vae
            self.text_encoder = pipe.text_encoder
            self.tokenizer = getattr(pipe, "tokenizer", None)
            self.noise_scheduler = pipe.scheduler
            self._processor = getattr(pipe, "processor", None)
            # Keep the full pipeline so encode_instruction can delegate to it
            # rather than hand-rolling the processor call (the pipeline knows its
            # own input format, we don't need to guess).
            self._pipeline = pipe
            logger.debug(
                "Pipeline type=%s  attrs=%s",
                type(pipe).__name__,
                [a for a in dir(pipe) if not a.startswith("_")],
            )

        except (ImportError, Exception) as e:
            logger.warning(f"Pipeline load failed: {e}. Using stub model.")
            logger.debug("Pipeline load traceback:", exc_info=True)
            self.model = _QwenEditDiT(hidden_dim=1024, num_heads=16, num_layers=24, latent_channels=self.latent_channels)
            self.model = self.model.to(dtype=dtype)
            self.vae = None
            self.text_encoder = None
            self.tokenizer = None
            self.noise_scheduler = None

        self.model.requires_grad_(False)
        if self.vae is not None:
            self.vae.requires_grad_(False)
            self.latent_channels = getattr(self.vae.config, "latent_channels", 4)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)

        logger.info(f"Qwen Edit loaded: {self.param_count/1e9:.2f}B params")
        return self.model

    def inject_lora(self, injector: Any) -> None:
        self._lora_injector = injector
        if self.model is None:
            raise RuntimeError("Call load_model() before inject_lora()")
        injector.target_modules = self.detect_lora_targets()
        injector.inject(self.model)
        logger.info(
            f"Qwen Edit LoRA injected: {injector.num_injected()} layers, "
            f"rank={injector.rank}, alpha={injector.alpha}"
        )
        logger.info(
            f"Trainable params: {self.trainable_param_count/1e6:.1f}M / "
            f"{self.param_count/1e9:.2f}B total"
        )

    def _is_flow_matching(self) -> bool:
        """True when the scheduler is flow-matching (no add_noise method)."""
        sched = self.noise_scheduler
        return sched is not None and not hasattr(sched, "add_noise")

    @staticmethod
    def _pack_latents(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
        """Pack (B, C, H, W) → (B, N, C*p^2) as the QwenImageEdit pipeline does.

        The transformer's img_in layer expects pre-packed tokens, not raw spatial
        latents. Packing with p=2 maps C=16 channels to 64-dim patch vectors that
        match img_in = Linear(64, 3072).
        """
        B, C, H, W = latents.shape
        latents = latents.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(B, (H // patch_size) * (W // patch_size), C * patch_size * patch_size)
        return latents

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Diffusion forward pass for single-image or source→target editing."""
        target_latents = batch["target_latents"]
        encoder_hidden_states = batch["encoder_hidden_states"]
        timesteps = batch["timesteps"]

        noisy, noise, t = self._add_noise(target_latents, timesteps)

        # Flow-matching target: velocity v = noise - x_0
        # DDPM target: noise itself
        if self._is_flow_matching():
            target = noise - target_latents
        else:
            target = noise

        B, C, H, W = target_latents.shape
        patch_size = 2

        # Pack noisy target: (B, C, H, W) → (B, N, C*patch_size^2) = (B, N, 64)
        # img_in = Linear(64, 3072) — tokens must be 64-dim, not 128-dim.
        noisy_packed = self._pack_latents(noisy)
        target_packed = self._pack_latents(target)

        source_latents = batch.get("source_latents")
        if source_latents is not None:
            # diffusers QwenEmbedRope.forward() always takes img_shapes[0] only.
            # Use frame=2 so RoPE covers the full concatenated sequence (noisy + source).
            # Frame 0 = noisy target tokens, frame 1 = source tokens.
            _shape = (2, H // patch_size, W // patch_size)
            source_packed = self._pack_latents(source_latents)
            model_input_packed = torch.cat([noisy_packed, source_packed], dim=1)
        else:
            _shape = (1, H // patch_size, W // patch_size)
            model_input_packed = noisy_packed
        img_shapes = [_shape] * B

        logger.debug(
            "forward: latent=%s packed=%s target_packed=%s t=%s",
            tuple(target_latents.shape), tuple(model_input_packed.shape),
            tuple(target_packed.shape), tuple(t.shape),
        )

        # Timesteps are already floats in [0,1] (logit-normal sampled in collate_fn).
        # The transformer expects [0,1] directly — do NOT divide by 1000.
        timestep_model = t.float()

        logger.debug(
            "forward: timestep_model range=[%.4f, %.4f]",
            timestep_model.min().item(), timestep_model.max().item(),
        )

        out = self.model(
            model_input_packed,
            timestep=timestep_model,
            encoder_hidden_states=encoder_hidden_states,
            img_shapes=img_shapes,
        )
        # Transformers/DiT return a dataclass; extract .sample if present
        pred = out.sample if hasattr(out, "sample") else out

        # When source latents are concatenated, pred is (B, 2N, C).
        # Only supervise the noisy target tokens (first N); discard the source half.
        N = target_packed.shape[1]
        pred_target = pred[:, :N, :]

        loss = self.loss({"pred": pred_target, "target": target_packed}, batch)
        return {"loss": loss, "pred": pred_target}

    def loss(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> torch.Tensor:
        pred = outputs["pred"]
        target = outputs.get("target", batch.get("noise", pred))
        return F.mse_loss(pred, target)

    def _add_noise(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        noise = torch.randn_like(latents)
        sched = self.noise_scheduler
        if sched is not None and hasattr(sched, "add_noise"):
            # DDPM-style scheduler (e.g. DDPMScheduler)
            noisy = sched.add_noise(latents, noise, timesteps)
        else:
            # Flow-matching: x_t = (1-t)*x_0 + t*noise
            # timesteps are already floats in [0,1] (logit-normal sampled in collate_fn).
            # Cast t to latents.dtype to avoid float32×bfloat16 promotion making noisy float32.
            t = timesteps.to(latents.dtype).view(-1, *([1] * (latents.ndim - 1)))
            noisy = (1.0 - t) * latents + t * noise
        return noisy, noise, timesteps

    def get_fsdp_wrap_policy(self):
        """Qwen Edit FSDP wrapping: wrap at DiT block level."""
        from torch.distributed.fsdp.wrap import (
            ModuleWrapPolicy,
            size_based_auto_wrap_policy,
        )

        block_classes = self._get_block_classes()
        if block_classes:
            return ModuleWrapPolicy(block_classes)

        min_params = int(self.cfg.distributed.fsdp.get("min_params", 1e7))
        return functools.partial(size_based_auto_wrap_policy, min_num_params=min_params)

    def _get_block_classes(self) -> set:
        classes = set()
        for module in self.model.modules():
            name = type(module).__name__
            if any(kw in name for kw in ("Block", "Layer", "TransformerBlock")):
                if sum(p.numel() for p in module.parameters()) > 1e6:
                    classes.add(type(module))
        return classes

    def get_collate_fn(self):
        """Return the qwen_edit collate function (handles latent cache + logit-normal ts)."""
        from pathlib import Path
        from choochoo.data.pipeline import edit_collate_fn
        cache_base = self.cfg.data.get("cache_dir") or str(
            Path(self.cfg.logging.get("output_dir", "./output")) / "cache"
        )
        cache_dir = Path(cache_base) / "encoded"
        adapter = self
        return lambda batch: edit_collate_fn(batch, cache_dir=cache_dir, adapter=adapter)

    def sample(self, sample_cfg) -> Optional[list]:
        from PIL import Image as PILImage

        pipe = getattr(self, "_pipeline", None)
        if pipe is None:
            logger.warning("QwenEditAdapter.sample: no _pipeline found, skipping")
            return None

        s = sample_cfg
        w, h = int(s.get("width", 480)), int(s.get("height", 480))
        blank_image = PILImage.new("RGB", (w, h), color=(0, 0, 0))
        images_cfg = s.get("images", [])

        # Move the entire pipeline to the transformer's device so diffusers' internal
        # image preprocessor creates tensors on the correct device before passing to the VAE.
        model_device = next(self.model.parameters()).device
        pipe.to(model_device)
        gen = torch.Generator(device=model_device).manual_seed(int(s.get("seed", 42)))

        results = []
        for i, prompt in enumerate(s.get("prompts", [])):
            src = blank_image
            if i < len(images_cfg) and images_cfg[i]:
                try:
                    src = PILImage.open(images_cfg[i]).convert("RGB")
                except Exception as e:
                    logger.warning("QwenEditAdapter.sample: could not load image %s: %s", images_cfg[i], e)
            with torch.inference_mode():
                result = pipe(
                    image=src,
                    prompt=prompt,
                    height=h,
                    width=w,
                    num_inference_steps=int(s.get("steps", 20)),
                    true_cfg_scale=float(s.get("cfg", 4.0)),
                    generator=gen,
                )
            images = getattr(result, "images", None) or result.frames[0]
            img = images[0] if isinstance(images, list) else images
            results.append({"mime": "image/png", "data": img})
        return results

    def prepare_batch(self, raw_batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert ImageDataset output (pixel_values + instruction) to adapter input."""
        pixel_values = raw_batch["pixel_values"]
        # Collate fn produces a list of (C,H,W) tensors; stack to (B,C,H,W).
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = torch.stack(pixel_values)
        instructions = raw_batch.get("instruction", raw_batch.get("caption", [""]))

        target_latents = self.encode_image(pixel_values)

        if isinstance(instructions, str):
            instructions = [instructions]
        encoded = [self.encode_instruction(inst, pixel_values[i:i+1]) for i, inst in enumerate(instructions)]
        encoder_hidden_states = torch.cat([hs for hs, _ in encoded], dim=0)
        masks = [m for _, m in encoded]
        encoder_attention_mask = torch.cat(masks, dim=0) if all(m is not None for m in masks) else None

        timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=pixel_values.device)
        out = {
            "target_latents": target_latents,
            "encoder_hidden_states": encoder_hidden_states,
            "timesteps": timesteps,
        }
        if encoder_attention_mask is not None:
            out["encoder_attention_mask"] = encoder_attention_mask
        return out

    def encode_instruction(self, instruction: str, source_image: Any) -> tuple:
        """Encode instruction + source image into conditioning embeddings.

        Returns (hidden_states, attention_mask) matching the pipeline's encode_prompt
        output. The mask marks valid (non-padding) token positions; the transformer
        uses it to ignore padding when attending to encoder hidden states.

        Delegates to the pipeline's own encode_prompt method when available.
        """
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not loaded")

        # Convert tensor (1,C,H,W) in [-1,1] to PIL.
        if isinstance(source_image, torch.Tensor):
            import numpy as _np
            from PIL import Image as _PIL
            img_t = source_image[0].float().cpu()
            logger.debug(
                "encode_instruction: source_image shape=%s dtype=%s range=[%.3f, %.3f]",
                tuple(img_t.shape), img_t.dtype, img_t.min().item(), img_t.max().item(),
            )
            img_np = (_np.clip((img_t.numpy() * 0.5 + 0.5) * 255, 0, 255)
                      .astype(_np.uint8).transpose(1, 2, 0))
            pil_image = _PIL.fromarray(img_np)
        else:
            pil_image = source_image

        device = next(self.text_encoder.parameters()).device

        # For Qwen VL models the correct encoding path is: format inputs via the
        # processor's chat template (which injects image placeholder tokens), then
        # run the text_encoder (the VLM) in encoder-only mode with output_hidden_states=True.
        #
        # pipe.encode_prompt is a pipeline-level wrapper designed for autoregressive
        # generation — it sets up KV caches and generation state that cause it to hang
        # when called in a training context. We skip it entirely.
        processor = self._processor
        if processor is None:
            raise RuntimeError(
                "No processor available on the pipeline. Cannot encode instruction. "
                "Ensure the pipeline loaded correctly and self._processor is set."
            )

        # Build inputs using apply_chat_template if the processor supports it
        # (required by vision-language models to inject image placeholder tokens).
        if hasattr(processor, "apply_chat_template"):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": instruction},
            ]}]
            formatted_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            logger.debug("encode_instruction: formatted_text[:120]=%r", formatted_text[:120])
            inputs = processor(text=[formatted_text], images=[pil_image], return_tensors="pt")
        else:
            inputs = processor(text=instruction, images=pil_image, return_tensors="pt")

        logger.debug(
            "encode_instruction: processor inputs — %s",
            {k: (tuple(v.shape), v.dtype) for k, v in inputs.items() if hasattr(v, "shape")},
        )
        # Move inputs to the encoder device, but keep image_grid_thw on CPU.
        # image_grid_thw is spatial metadata used only as integer indices (e.g. .prod(-1)
        # to compute split_sizes). Placing it on GPU forces PyTorch to JIT-compile a CUDA
        # reduction kernel via NVRTC, which fails in environments where libnvrtc-builtins
        # is missing. CPU is correct — the tensor is only used for Python-level splitting.
        _cpu_keys = {"image_grid_thw"}
        inputs = {k: (v if k in _cpu_keys else v.to(device)) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.text_encoder(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = inputs.get("attention_mask", None)
        logger.debug("encode_instruction: hidden_states[-1] shape=%s mask=%s", tuple(hidden.shape), tuple(mask.shape) if mask is not None else None)
        model_device = next(self.model.parameters()).device
        return hidden.to(model_device), (mask.to(model_device) if mask is not None else None)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latents using VAE.

        The Qwen-Image-Edit VAE is a 3D (WAN-style) VAE that expects
        (B, C, T, H, W). Lift 4-D image tensors to 5-D before encoding
        and squeeze the temporal dimension back out afterward.
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded")
        with torch.no_grad():
            # Encode on VAE's device; caller may have image on a different device.
            vae_param = next(self.vae.parameters())
            image = image.to(device=vae_param.device, dtype=vae_param.dtype)
            logger.debug(
                "encode_image: input shape=%s dtype=%s device=%s",
                tuple(image.shape), image.dtype, image.device,
            )
            needs_temporal = image.ndim == 4
            if needs_temporal:
                image = image.unsqueeze(2)  # (B,C,H,W) → (B,C,1,H,W)
                logger.debug("encode_image: added temporal dim → shape=%s", tuple(image.shape))
            latents = self.vae.encode(image).latent_dist.sample()
            if needs_temporal:
                latents = latents.squeeze(2)  # drop temporal dim
            logger.debug("encode_image: latents shape=%s", tuple(latents.shape))
            # Normalize latents using per-channel mean/std buffers (Wan-style VAE).
            # These are registered buffers on the VAE, not a scalar scaling_factor.
            # Diffusion-pipe reference: (latents - mean) / std
            if hasattr(self.vae, "latents_mean_tensor") and hasattr(self.vae, "latents_std_tensor"):
                mean = self.vae.latents_mean_tensor.to(device=latents.device, dtype=latents.dtype)
                std = self.vae.latents_std_tensor.to(device=latents.device, dtype=latents.dtype)
                latents = (latents - mean) / std
                logger.debug("encode_image: applied VAE mean/std normalization")
                logger.warning(
                    "VAE mean/std normalization applied — delete data/.cache/encoded/ "
                    "if this is the first run with normalization enabled."
                )
            else:
                logger.debug("encode_image: no latents_mean_tensor/latents_std_tensor on VAE — using raw latents")
            return latents


class _QwenEditDiT(nn.Module):
    """Minimal stub DiT for Qwen Edit without pretrained weights."""

    def __init__(self, hidden_dim: int = 1024, num_heads: int = 16, num_layers: int = 24, latent_channels: int = 4):
        super().__init__()
        self.latent_channels = latent_channels
        self.input_proj = nn.Linear(latent_channels, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.blocks = nn.ModuleList([
            _EditBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, latent_channels)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = x.shape[0]
        t_emb = _timestep_emb(timestep, 256).to(x.dtype)
        t_emb = self.time_embed(t_emb)

        x_flat = x.reshape(b, x.shape[1], -1).permute(0, 2, 1)
        x_flat = self.input_proj(x_flat)

        for block in self.blocks:
            x_flat = block(x_flat, t_emb, encoder_hidden_states)

        x_flat = self.norm(x_flat)
        return self.out(x_flat).permute(0, 2, 1).reshape(b, self.latent_channels, *x.shape[2:])


class _EditBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d)
        )

    def forward(self, x, t_emb, ctx=None):
        x = x + self.attn(*([self.norm1(x + t_emb.unsqueeze(1))] * 3))[0]
        if ctx is not None:
            x_n = self.norm2(x)
            x = x + self.cross_attn(x_n, ctx, ctx)[0]
        return x + self.mlp(self.norm3(x))


def _timestep_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, dtype=torch.float32, device=t.device)
        * (torch.log(torch.tensor(10000.0)) / half)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
