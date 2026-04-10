"""LoRA injection system — replaces target modules in-place."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .layers import LoRAConv2d, LoRALinear

logger = logging.getLogger(__name__)


class LoRAInjector:
    """Inject LoRA adapters into a model's target modules.

    Supports selective targeting via module name patterns (regex),
    and dual-LoRA mode for combined high/low-noise training.

    target_modules may be left empty at construction and assigned by the
    adapter's inject_lora() before calling inject().
    """

    def __init__(
        self,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        exclude_modules: Optional[List[str]] = None,
        dual: bool = False,
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or []
        self.exclude_modules = exclude_modules or []
        self.dual = dual
        self._injected: Dict[str, nn.Module] = {}
        # Compiled in inject() after target_modules is finalised by the adapter.
        self._compiled_patterns: List[re.Pattern] = []
        self._compiled_excludes: List[re.Pattern] = []

    def _should_target(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            return False
        for pat in self._compiled_excludes:
            if pat.search(name):
                return False
        for pat in self._compiled_patterns:
            if pat.search(name):
                return True
        return False

    def inject(self, model: nn.Module) -> nn.Module:
        """Replace targeted layers with LoRA wrappers. Returns modified model."""
        if not self.target_modules:
            raise ValueError("target_modules must be set before calling inject()")

        # (Re)compile — target_modules is typically assigned by the adapter after __init__
        self._compiled_patterns = [re.compile(p) for p in self.target_modules]
        self._compiled_excludes = [re.compile(p) for p in self.exclude_modules]

        self._injected.clear()
        replacements: List[Tuple[nn.Module, str, nn.Module]] = []

        for name, module in model.named_modules():
            if not self._should_target(name, module):
                continue
            parent, attr = self._get_parent(model, name)
            if isinstance(module, nn.Linear):
                lora_layer = LoRALinear(
                    module,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    dual=self.dual,
                )
            elif isinstance(module, nn.Conv2d):
                lora_layer = LoRAConv2d(
                    module,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
            else:
                continue
            replacements.append((parent, attr, lora_layer))
            self._injected[name] = lora_layer

        for parent, attr, lora_layer in replacements:
            setattr(parent, attr, lora_layer)

        if not self._injected:
            raise RuntimeError("LoRA injection failed: no modules matched any target pattern")

        # Per-pattern hit logging
        pattern_hits = {p: 0 for p in self.target_modules}
        for name in self._injected:
            for i, pat in enumerate(self._compiled_patterns):
                if pat.search(name):
                    pattern_hits[self.target_modules[i]] += 1
                    break
        logger.info("=== LoRA Injection ===")
        for pattern, count in pattern_hits.items():
            if count:
                logger.info("  %-40s -> %d layers", pattern, count)
        logger.info("  Total injected: %d", len(self._injected))

        return model

    @staticmethod
    def _get_parent(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Return only LoRA adapter parameters (not base model)."""
        params = []
        for layer in self._injected.values():
            if isinstance(layer, LoRALinear):
                params.append(layer.lora_A)
                params.append(layer.lora_B)
                if layer.dual:
                    params.append(layer.lora_A2)
                    params.append(layer.lora_B2)
            elif isinstance(layer, LoRAConv2d):
                params.extend(layer.lora_A.parameters())
                params.extend(layer.lora_B.parameters())
        return params

    def get_lora_state_dict(self) -> dict:
        """Return state dict containing only LoRA weights."""
        sd = {}
        for name, layer in self._injected.items():
            prefix = name + "."
            sd.update(layer.get_lora_state_dict(prefix=prefix))
        return sd

    def load_lora_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Load LoRA weights from a previously saved state dict."""
        lora_sd = self.get_lora_state_dict()
        if strict:
            missing = set(lora_sd.keys()) - set(state_dict.keys())
            unexpected = set(state_dict.keys()) - set(lora_sd.keys())
            if missing:
                raise KeyError(f"Missing LoRA keys: {missing}")
            if unexpected:
                raise KeyError(f"Unexpected LoRA keys: {unexpected}")

        for name, layer in self._injected.items():
            prefix = name + "."
            if isinstance(layer, LoRALinear):
                # Support both ComfyUI-compatible keys (lora_A.weight) and legacy bare keys
                a_key = f"{prefix}lora_A.weight" if f"{prefix}lora_A.weight" in state_dict else f"{prefix}lora_A"
                b_key = f"{prefix}lora_B.weight" if f"{prefix}lora_B.weight" in state_dict else f"{prefix}lora_B"
                if a_key in state_dict:
                    layer.lora_A.data.copy_(state_dict[a_key])
                if b_key in state_dict:
                    layer.lora_B.data.copy_(state_dict[b_key])
                if layer.dual:
                    a2_key = f"{prefix}lora_A2.weight" if f"{prefix}lora_A2.weight" in state_dict else f"{prefix}lora_A2"
                    b2_key = f"{prefix}lora_B2.weight" if f"{prefix}lora_B2.weight" in state_dict else f"{prefix}lora_B2"

                    if a2_key in state_dict:
                        layer.lora_A2.data.copy_(state_dict[a2_key])
                    if b2_key in state_dict:
                        layer.lora_B2.data.copy_(state_dict[b2_key])

    def num_injected(self) -> int:
        return len(self._injected)

    def injected_names(self) -> List[str]:
        return list(self._injected.keys())
