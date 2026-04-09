"""Qwen Image adapter for text-to-image generation (Qwen/Qwen-Image)."""

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
        sys.modules[_mod_name] = _make_stub_module(_mod_name)
    del _mod_name
    sys.modules["flash_attn"].flash_attn_interface = sys.modules["flash_attn.flash_attn_interface"]
    sys.modules["flash_attn"].flash_attn_utils = sys.modules["flash_attn.flash_attn_utils"]
    sys.modules["flash_attn"].__version__ = "0.0.0"  # type: ignore[attr-defined]
del _needs_flash_stub

# xformers is often ABI-incompatible. Stub the entire package.
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

if not hasattr(sys.modules.get("torchvision"), "transforms"):
    _stub = ModuleType("torchvision._meta_registrations")
    _stub.__spec__ = importlib.machinery.ModuleSpec("torchvision._meta_registrations", None)
    sys.modules["torchvision._meta_registrations"] = _stub
    del _stub

import diffusers.utils.import_utils as _diffusers_utils
_diffusers_utils.is_flash_attn_available = lambda: False  # type: ignore[attr-defined]
if hasattr(_diffusers_utils, "_flash_attn_available"):
    _diffusers_utils._flash_attn_available = False
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


class QwenAdapter(BaseModelAdapter):
    """Adapter for Qwen/Qwen-Image — pure text-to-image generation.

    Unlike QwenEditAdapter, this adapter requires only a caption + target image
    pair. No source image conditioning. Uses the same MMDiT architecture, VAE,
    packing, RoPE, and flow-matching schedule as Qwen-Image-Edit.
    """

    DEFAULT_LORA_TARGETS = [
        # Image stream attention
        r"attn\.to_q", r"attn\.to_k", r"attn\.to_v", r"attn\.to_out",
        # Text stream attention (MMDiT add_ projections)
        r"attn\.add_q_proj", r"attn\.add_k_proj", r"attn\.add_v_proj", r"attn\.to_add_out",
        # Image stream FFN
        r"img_mlp\.net\.0\.proj", r"img_mlp\.net\.2",
        # Text stream FFN
        r"txt_mlp\.net\.0\.proj", r"txt_mlp\.net\.2",
    ]

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.tokenizer = None
        self._processor = None
        self._lora_injector = None
        self.latent_channels = 4

    def load_model(self) -> nn.Module:
        pretrained_path = self.cfg.model.pretrained_path
        dtype = self.prepare_dtype()

        logger.info(f"Loading Qwen Image from: {pretrained_path}")

        try:
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(
                pretrained_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.model = pipe.transformer

            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    print(name)
            self.vae = pipe.vae
            self.text_encoder = pipe.text_encoder
            self.tokenizer = getattr(pipe, "tokenizer", None)
            self.noise_scheduler = pipe.scheduler
            self._processor = getattr(pipe, "processor", None)
            self._pipeline = pipe
            logger.debug(
                "Pipeline type=%s  attrs=%s",
                type(pipe).__name__,
                [a for a in dir(pipe) if not a.startswith("_")],
            )

        except (ImportError, Exception) as e:
            logger.warning(f"Pipeline load failed: {e}. No stub for QwenAdapter.")
            logger.debug("Pipeline load traceback:", exc_info=True)
            raise

        self.model.requires_grad_(False)
        if self.vae is not None:
            self.vae.requires_grad_(False)
            self.latent_channels = getattr(self.vae.config, "latent_channels", 4)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)

        logger.info(f"Qwen Image loaded: {self.param_count/1e9:.2f}B params")
        return self.model

    def inject_lora(self, injector: Any) -> None:
        self._lora_injector = injector
        if self.model is None:
            raise RuntimeError("Call load_model() before inject_lora()")
        target_modules = list(self.cfg.lora.get("target_modules", None) or self.DEFAULT_LORA_TARGETS)
        injector.target_modules = target_modules
        injector.inject(self.model)
        logger.info(
            f"Qwen Image LoRA injected: {injector.num_injected()} layers, "
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
        """Pack (B, C, H, W) → (B, N, C*p^2) as the Qwen pipeline does."""
        B, C, H, W = latents.shape
        latents = latents.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(B, (H // patch_size) * (W // patch_size), C * patch_size * patch_size)
        return latents

    def _add_noise(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        noise = torch.randn_like(latents)
        sched = self.noise_scheduler
        if sched is not None and hasattr(sched, "add_noise"):
            noisy = sched.add_noise(latents, noise, timesteps)
        else:
            # Flow-matching: x_t = (1-t)*x_0 + t*noise
            t = timesteps.to(latents.dtype).view(-1, *([1] * (latents.ndim - 1)))
            noisy = (1.0 - t) * latents + t * noise
        return noisy, noise, timesteps

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Flow-matching forward pass for text-to-image generation."""
        latents = batch["target_latents"]
        encoder_hidden_states = batch["encoder_hidden_states"]
        timesteps = batch["timesteps"]

        noisy, noise, t = self._add_noise(latents, timesteps)

        if self._is_flow_matching():
            target = noise - latents
        else:
            target = noise

        B, C, H, W = latents.shape
        patch_size = 2

        noisy_packed = self._pack_latents(noisy)
        target_packed = self._pack_latents(target)

        # Pure T2I: single image per sample, no source concatenation
        img_shapes = [(1, H // patch_size, W // patch_size)] * B

        logger.debug(
            "forward: latent=%s packed=%s target_packed=%s t=%s",
            tuple(latents.shape), tuple(noisy_packed.shape),
            tuple(target_packed.shape), tuple(t.shape),
        )

        timestep_model = t.float()

        out = self.model(
            noisy_packed,
            timestep=timestep_model,
            encoder_hidden_states=encoder_hidden_states,
            img_shapes=img_shapes,
        )
        pred = out.sample if hasattr(out, "sample") else out

        loss = F.mse_loss(pred, target_packed)
        return {"loss": loss, "pred": pred}

    def loss(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> torch.Tensor:
        pred = outputs["pred"]
        target = outputs.get("target", batch.get("noise", pred))
        return F.mse_loss(pred, target)

    def get_collate_fn(self):
        """Reuse edit_collate_fn — it already handles missing source_latents."""
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
            logger.warning("QwenAdapter.sample: no _pipeline found, skipping")
            return None

        s = sample_cfg
        w, h = int(s.get("width", 1024)), int(s.get("height", 1024))

        model_device = next(self.model.parameters()).device
        pipe.to(model_device)
        gen = torch.Generator(device=model_device).manual_seed(int(s.get("seed", 42)))

        negative_prompt = s.get("negative_prompt", " ")

        results = []
        for prompt in s.get("prompts", []):
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=h,
                    width=w,
                    num_inference_steps=int(s.get("steps", 50)),
                    true_cfg_scale=float(s.get("cfg", 4.0)),
                    generator=gen,
                )
            img = result.images[0]
            results.append({"mime": "image/png", "data": img})
        return results

    def encode_instruction(self, instruction: str, source_image: Any = None) -> tuple:
        """Encode text-only prompt into conditioning embeddings.

        Qwen-Image has no standalone processor — text is tokenized directly via
        the pipeline's tokenizer (pipe.tokenizer). The processor attr is checked
        first for forward-compatibility with pipeline variants that do expose one.
        """
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not loaded")

        device = next(self.text_encoder.parameters()).device

        # Prefer processor if available (future pipeline variants), else use tokenizer.
        tokenizer = self._processor or self.tokenizer
        if tokenizer is None:
            raise RuntimeError("No tokenizer or processor available. Cannot encode instruction.")

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": instruction}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = tokenizer(text=[text], return_tensors="pt")
        else:
            inputs = tokenizer(
                instruction,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

        inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
        with torch.no_grad():
            outputs = self.text_encoder(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = inputs.get("attention_mask", None)
        logger.debug(
            "encode_instruction: hidden_states[-1] shape=%s mask=%s",
            tuple(hidden.shape), tuple(mask.shape) if mask is not None else None,
        )
        model_device = next(self.model.parameters()).device
        return hidden.to(model_device), (mask.to(model_device) if mask is not None else None)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latents using the 3D (WAN-style) VAE."""
        if self.vae is None:
            raise RuntimeError("VAE not loaded")
        with torch.no_grad():
            vae_param = next(self.vae.parameters())
            image = image.to(device=vae_param.device, dtype=vae_param.dtype)
            logger.debug(
                "encode_image: input shape=%s dtype=%s device=%s",
                tuple(image.shape), image.dtype, image.device,
            )
            needs_temporal = image.ndim == 4
            if needs_temporal:
                image = image.unsqueeze(2)  # (B,C,H,W) → (B,C,1,H,W)
            latents = self.vae.encode(image).latent_dist.sample()
            if needs_temporal:
                latents = latents.squeeze(2)
            logger.debug("encode_image: latents shape=%s", tuple(latents.shape))
            if hasattr(self.vae, "latents_mean_tensor") and hasattr(self.vae, "latents_std_tensor"):
                mean = self.vae.latents_mean_tensor.to(device=latents.device, dtype=latents.dtype)
                std = self.vae.latents_std_tensor.to(device=latents.device, dtype=latents.dtype)
                latents = (latents - mean) / std
                logger.debug("encode_image: applied VAE mean/std normalization")
            return latents

    def prepare_batch(self, raw_batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert ImageDataset output (pixel_values + caption) to adapter input."""
        pixel_values = raw_batch["pixel_values"]
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = torch.stack(pixel_values)
        instructions = raw_batch.get("instruction", raw_batch.get("caption", [""]))

        target_latents = self.encode_image(pixel_values)

        if isinstance(instructions, str):
            instructions = [instructions]
        encoded = [self.encode_instruction(inst) for inst in instructions]
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

    def get_fsdp_wrap_policy(self):
        """FSDP wrapping: wrap at DiT block level."""
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
