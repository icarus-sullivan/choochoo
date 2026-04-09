"""LoRA export, merging, and format conversion utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .injection import LoRAInjector
from .layers import LoRALinear, LoRAConv2d


class LoRAExporter:
    """Export LoRA weights to safetensors or merge into base model."""

    def __init__(self, injector: LoRAInjector):
        self.injector = injector

    def save(
        self,
        path: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Save LoRA weights to a safetensors file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self.injector.get_lora_state_dict()
        # Convert to float32 for maximum compatibility
        state_dict = {k: v.float().contiguous() for k, v in state_dict.items()}

        meta = {
            "rank": str(self.injector.rank),
            "alpha": str(self.injector.alpha),
            "target_modules": json.dumps(self.injector.target_modules),
        }
        if metadata:
            meta.update(metadata)

        save_file(state_dict, str(path), metadata=meta)

    def load(self, path: Union[str, Path]) -> None:
        """Load LoRA weights from a safetensors file into the injected model."""
        state_dict = load_file(str(path))
        state_dict = {k: v for k, v in state_dict.items()}
        self.injector.load_lora_state_dict(state_dict, strict=True)

    def merge_into_base(self, model: nn.Module) -> nn.Module:
        """Merge all injected LoRA layers back into base weights (for inference)."""
        for name, lora_layer in self.injector._injected.items():
            if not isinstance(lora_layer, LoRALinear):
                continue
            parent, attr = LoRAInjector._get_parent(model, name)
            merged = lora_layer.merge_weights()
            setattr(parent, attr, merged)
        return model

    @staticmethod
    def combine_loras(
        paths: list,
        weights: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """Weighted combination of multiple LoRA state dicts."""
        if weights is None:
            weights = [1.0 / len(paths)] * len(paths)
        assert len(paths) == len(weights), "paths and weights must have same length"

        combined: Dict[str, torch.Tensor] = {}
        for path, w in zip(paths, weights):
            sd = load_file(str(path))
            for k, v in sd.items():
                if k in combined:
                    combined[k] = combined[k] + v.float() * w
                else:
                    combined[k] = v.float() * w
        return combined
