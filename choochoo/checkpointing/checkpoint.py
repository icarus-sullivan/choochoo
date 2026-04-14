"""Fast, distributed-safe checkpointing with named LoRA output files.

Output naming convention:
  Single-path LoRA (all models):
    <prefix>-<name>-step<step>.safetensors

  Dual-LoRA (WAN with dual_lora: true):
    <prefix>-<name>-high-step<step>.safetensors   (primary A/B path, high-noise)
    <prefix>-<name>-low-step<step>.safetensors    (secondary A2/B2 path, low-noise)

Model-type → prefix:
    wan22      → wan
    qwen_edit  → qwen
    ltx2       → ltx2
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

_MODEL_PREFIX = {
    "wan22":      "wan",
    "wan21":      "wan",
    "wan_i2v":    "wan",
    "wan22_dual": "wan",
    "qwen_edit":  "qwen",
    "qwen":       "qwen",
    "ltx2":       "ltx2",
}


def _lora_filename(prefix: str, name: str, step: int, branch: Optional[str] = None) -> str:
    """Build the canonical LoRA output filename."""
    if branch:
        return f"{prefix}-{name}-{branch}-step{step}.safetensors"
    return f"{prefix}-{name}-step{step}.safetensors"


def _split_dual_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a dual-LoRA state dict into high-noise and low-noise dicts.

    Keys ending in 'lora_A' / 'lora_B'  → high-noise (primary path)
    Keys ending in 'lora_A2' / 'lora_B2' → low-noise  (secondary path),
                                             renamed to lora_A / lora_B so both
                                             output files are standard LoRA format.
    """
    high: Dict[str, torch.Tensor] = {}
    low: Dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        if k.endswith("lora_A2"):
            low[k.replace("lora_A2", "lora_A")] = v
        elif k.endswith("lora_B2"):
            low[k.replace("lora_B2", "lora_B")] = v
        else:
            high[k] = v

    return high, low


class CheckpointManager:
    """Manages training checkpoints with named LoRA safetensors output.

    Features:
    - Named LoRA safetensors using run name + step (e.g. wan-myrun-step500.safetensors)
    - Dual-LoRA split into separate high/low files
    - Full training state (optimizer, scheduler) for resumption
    - Top-K checkpoint rotation
    - FSDP-compatible saving via atomic rename
    """

    def __init__(
        self,
        output_dir: str,
        cfg: DictConfig,
        keep_last_n: int = 3,
        save_lora_only: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.cfg = cfg
        self.keep_last_n = keep_last_n
        self.save_lora_only = save_lora_only

        self._run_name: str = str(cfg.name)
        self._model_prefix: str = _MODEL_PREFIX.get(cfg.model.type, cfg.model.type)
        # Legacy dual_lora flag (single-model dual-path LoRA via lora_A2/B2 keys)
        self._dual: bool = bool(cfg.lora.get("dual_lora", False))
        # wan22_dual: two separate transformer objects → separate prefix keys
        self._dual_model: bool = cfg.model.type == "wan22_dual"
        # Models whose LoRA keys need diffusion_model. prefix for ComfyUI compatibility
        _COMFYUI_PREFIX_MODELS = {"qwen_edit", "qwen", "wan22", "wan21", "wan_i2v", "wan22_dual"}
        self._comfyui_prefix: bool = cfg.model.type in _COMFYUI_PREFIX_MODELS

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._saved: List[Tuple[int, Path]] = []  # (step, path)

    def save(
        self,
        step: int,
        model: nn.Module,
        injector: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        metrics: Optional[Dict[str, float]] = None,
        tag: Optional[str] = None,
        adapter: Optional[Any] = None,
    ) -> Path:
        """Save checkpoint. Returns path of saved directory."""
        dir_name = f"checkpoint-{step}" if tag is None else f"checkpoint-{tag}-{step}"
        ckpt_dir = self.output_dir / dir_name
        tmp_dir = self.output_dir / f".tmp-{dir_name}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        if injector is not None and self.save_lora_only:
            self._save_lora(injector, step, tmp_dir, tag=tag, adapter=adapter)
        else:
            self._save_full_model(model, tmp_dir)

        # Training state for resumption
        training_state: Dict[str, Any] = {"step": step, "metrics": metrics or {}}
        if optimizer is not None:
            training_state["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            training_state["scheduler"] = scheduler.state_dict()

        torch.save(training_state, tmp_dir / "training_state.pt")

        (tmp_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "metrics": metrics or {},
                    "tag": tag,
                    "name": self._run_name,
                    "model_prefix": self._model_prefix,
                    "dual_lora": self._dual,
                },
                indent=2,
            )
        )

        # Atomic rename
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        tmp_dir.rename(ckpt_dir)

        self._saved.append((step, ckpt_dir))
        self._rotate_checkpoints(tag)

        logger.info(f"Saved checkpoint: {ckpt_dir}")
        return ckpt_dir

    def _save_lora(
        self,
        injector: Any,
        step: int,
        directory: Path,
        tag: Optional[str] = None,
        adapter: Optional[Any] = None,
    ) -> None:
        """Save LoRA weights with canonical naming, splitting dual paths if needed.

        Regular interval saves (tag=None) write to both the checkpoint directory and
        the top-level output_dir so every save_every step accumulates as a standalone
        safetensors. Tagged saves (tag='best', tag='final') only write inside the
        checkpoint directory — they are not user-facing outputs.

        For wan22_dual adapters, LoRA state dicts from both transformers are extracted
        separately and saved as _high_noise / _low_noise safetensors files.
        """
        write_toplevel = tag is None

        base_meta = {
            "rank": str(injector.rank),
            "alpha": str(injector.alpha),
            "step": str(step),
            "name": self._run_name,
            "model_type": self._model_prefix,
        }

        # --- wan22_dual: separate LoRA injectors per transformer ---
        if self._dual_model and adapter is not None and hasattr(adapter, "_low_injector"):
            high_sd = {
                k: v.float().contiguous()
                for k, v in injector.get_lora_state_dict().items()
            }
            low_injector = adapter._low_injector
            low_sd = {
                k: v.float().contiguous()
                for k, v in low_injector.get_lora_state_dict().items()
            }
            self._validate_lora_save(injector, high_sd, label="high_noise")
            self._validate_lora_save(low_injector, low_sd, label="low_noise")
            if self._comfyui_prefix:
                high_sd = {"diffusion_model." + k: v for k, v in high_sd.items()}
                low_sd = {"diffusion_model." + k: v for k, v in low_sd.items()}
            if adapter is not None and hasattr(adapter, "remap_lora_keys"):
                high_sd = adapter.remap_lora_keys(high_sd)
                low_sd = adapter.remap_lora_keys(low_sd)

            high_name = _lora_filename(self._model_prefix, self._run_name, step, "high_noise")
            save_file(high_sd, str(directory / high_name), metadata={**base_meta, "branch": "high_noise"})
            if write_toplevel:
                save_file(high_sd, str(self.output_dir / high_name), metadata={**base_meta, "branch": "high_noise"})

            low_name = _lora_filename(self._model_prefix, self._run_name, step, "low_noise")
            save_file(low_sd, str(directory / low_name), metadata={**base_meta, "branch": "low_noise"})
            if write_toplevel:
                save_file(low_sd, str(self.output_dir / low_name), metadata={**base_meta, "branch": "low_noise"})

            logger.debug("Saved dual-model LoRA: %s, %s", high_name, low_name)
            return

        # --- Legacy dual_lora (single transformer, two LoRA paths via lora_A2/B2 keys) ---
        state_dict = injector.get_lora_state_dict()
        state_dict = {k: v.float().contiguous() for k, v in state_dict.items()}
        self._validate_lora_save(injector, state_dict)
        if self._comfyui_prefix:
            state_dict = {"diffusion_model." + k: v for k, v in state_dict.items()}
        if adapter is not None and hasattr(adapter, "remap_lora_keys"):
            state_dict = adapter.remap_lora_keys(state_dict)

        if self._dual and any(k.endswith("lora_A2") for k in state_dict):
            high_sd, low_sd = _split_dual_state_dict(state_dict)

            high_name = _lora_filename(self._model_prefix, self._run_name, step, "high")
            save_file(high_sd, str(directory / high_name), metadata={**base_meta, "branch": "high"})
            if write_toplevel:
                save_file(high_sd, str(self.output_dir / high_name), metadata={**base_meta, "branch": "high"})

            low_name = _lora_filename(self._model_prefix, self._run_name, step, "low")
            save_file(low_sd, str(directory / low_name), metadata={**base_meta, "branch": "low"})
            if write_toplevel:
                save_file(low_sd, str(self.output_dir / low_name), metadata={**base_meta, "branch": "low"})

            logger.debug("Saved legacy dual LoRA: %s, %s", high_name, low_name)
        else:
            fname = _lora_filename(self._model_prefix, self._run_name, step)
            save_file(state_dict, str(directory / fname), metadata=base_meta)
            if write_toplevel:
                save_file(state_dict, str(self.output_dir / fname), metadata=base_meta)
            logger.debug("Saved LoRA: %s", fname)

    @staticmethod
    def _validate_lora_save(injector: Any, state_dict: dict, label: str = "") -> None:
        """Validate that a LoRA state dict fully covers all injected modules.

        Checks:
          C — state dict is non-empty and has at least 2 tensors per injected module
          A — every injected module name appears in the saved keys
          D — logs a coverage summary
        """
        tag = f" ({label})" if label else ""
        n = len(state_dict)

        # Check C: non-empty and minimum tensor count
        if n == 0:
            raise RuntimeError(
                f"Saving empty LoRA state_dict{tag} — injection likely failed"
            )
        expected_min = injector.num_injected() * 2  # lora_A.weight + lora_B.weight per module
        if n < expected_min:
            raise RuntimeError(
                f"Incomplete LoRA save{tag}: expected at least {expected_min} tensors, got {n}"
            )

        # Check A: every injected module has saved keys
        injected_set: Set[str] = set(injector.injected_names())
        saved_modules: Set[str] = set()
        for k in state_dict:
            # Strip the trailing .lora_A.weight / .lora_B2.weight suffix to get module name
            base = re.sub(r"\.(lora_[AB]2?)(\.weight)?$", "", k)
            saved_modules.add(base)
        missing = injected_set - saved_modules
        extra = saved_modules - injected_set
        if missing:
            raise RuntimeError(f"LoRA save{tag} missing modules: {missing}")
        if extra:
            logger.warning("Unexpected LoRA modules in save%s (not injected): %s", tag, extra)

        # Check D: coverage summary
        logger.info(
            "LoRA Save Summary%s: injected=%d modules, saved=%d tensors, coverage=%.0f%%",
            tag,
            injector.num_injected(),
            n,
            n / max(expected_min, 1) * 100,
        )

    def _save_full_model(self, model: nn.Module, directory: Path) -> None:
        state_dict = {k: v.cpu().float().contiguous() for k, v in model.state_dict().items()}
        save_file(state_dict, str(directory / "model.safetensors"))

    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        injector: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
    ) -> Dict[str, Any]:
        """Load checkpoint. Returns training state dict."""
        ckpt_dir = Path(checkpoint_path)

        loaded = False
        if injector is not None:
            # Try named LoRA files first (new convention), fall back to legacy name
            candidates = list(ckpt_dir.glob("*.safetensors"))
            lora_files = [f for f in candidates if f.name != "model.safetensors"]

            if lora_files:
                # For dual: load both high and low into injector
                high_files = [f for f in lora_files if "-high-" in f.name]
                low_files = [f for f in lora_files if "-low-" in f.name]
                single_files = [f for f in lora_files if "-high-" not in f.name and "-low-" not in f.name]

                if high_files and low_files:
                    self._load_dual_lora(injector, high_files[0], low_files[0])
                    loaded = True
                elif single_files:
                    sd = load_file(str(single_files[0]))
                    # Strip ComfyUI prefix if present (handles diffusion-pipe-saved LoRAs)
                    sd = {k.removeprefix("diffusion_model."): v for k, v in sd.items()}
                    injector.load_lora_state_dict(sd, strict=False)
                    logger.info(f"Loaded LoRA from {single_files[0].name}")
                    loaded = True

        if not loaded:
            model_path = ckpt_dir / "model.safetensors"
            if model_path.exists():
                sd = load_file(str(model_path))
                model.load_state_dict(sd, strict=False)
                logger.info(f"Loaded model weights from {model_path}")

        training_state = {}
        state_path = ckpt_dir / "training_state.pt"
        if state_path.exists():
            training_state = torch.load(state_path, map_location="cpu", weights_only=False)
            if optimizer is not None and "optimizer" in training_state:
                optimizer.load_state_dict(training_state["optimizer"])
            if scheduler is not None and "scheduler" in training_state:
                scheduler.load_state_dict(training_state["scheduler"])

        return training_state

    def _load_dual_lora(self, injector: Any, high_path: Path, low_path: Path) -> None:
        """Load dual LoRA: high path → lora_A/B, low path → lora_A2/B2."""
        high_sd = load_file(str(high_path))
        low_sd = load_file(str(low_path))
        # Strip ComfyUI prefix if present
        high_sd = {k.removeprefix("diffusion_model."): v for k, v in high_sd.items()}
        low_sd = {k.removeprefix("diffusion_model."): v for k, v in low_sd.items()}

        # Remap low keys back to _A2/_B2 for the injector
        low_remapped = {}
        for k, v in low_sd.items():
            if k.endswith("lora_A"):
                low_remapped[k.replace("lora_A", "lora_A2")] = v
            elif k.endswith("lora_B"):
                low_remapped[k.replace("lora_B", "lora_B2")] = v
            else:
                low_remapped[k] = v

        combined = {**high_sd, **low_remapped}
        injector.load_lora_state_dict(combined, strict=False)
        logger.info(f"Loaded dual LoRA: {high_path.name} + {low_path.name}")

    def _rotate_checkpoints(self, tag: Optional[str] = None) -> None:
        if tag is not None:
            return
        regular = [(s, p) for s, p in self._saved if "best" not in p.name and "final" not in p.name]
        while len(regular) > self.keep_last_n:
            _, old_path = regular.pop(0)
            if old_path.exists():
                shutil.rmtree(old_path)
                logger.debug(f"Rotated out checkpoint: {old_path}")
            self._saved.remove((_, old_path))

    def list_checkpoints(self) -> List[Path]:
        return sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )

    def latest_checkpoint(self) -> Optional[Path]:
        ckpts = self.list_checkpoints()
        return ckpts[-1] if ckpts else None
