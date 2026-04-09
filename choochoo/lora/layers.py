"""LoRA layer implementations with fused forward and dual-path support."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer wrapping a frozen nn.Linear.

    Implements W' = W + (B @ A) * (alpha / rank) with optional dropout.
    Supports dual-LoRA paths (high-noise / low-noise) in a single module.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        dual: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dual = dual

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        # Primary LoRA matrices
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, dtype=dtype, device=device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=dtype, device=device)
        )

        if dual:
            # Second LoRA path (e.g. low-noise regime)
            self.lora_A2 = nn.Parameter(
                torch.empty(rank, in_features, dtype=dtype, device=device)
            )
            self.lora_B2 = nn.Parameter(
                torch.zeros(out_features, rank, dtype=dtype, device=device)
            )
            nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B initialized to zero → identity at start

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor, lora_path: int = 0) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.
            lora_path: 0 = primary LoRA, 1 = secondary (dual mode only).
        """
        base_out = self.base_layer(x)

        if lora_path == 1 and self.dual:
            lora_out = (self.dropout(x) @ self.lora_A2.T) @ self.lora_B2.T
        else:
            lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T

        return base_out + lora_out * self.scaling

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA into base layer and return plain Linear (inference mode)."""
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype,
        )
        delta = (self.lora_B @ self.lora_A) * self.scaling
        merged.weight.data = self.base_layer.weight.data + delta
        if self.base_layer.bias is not None:
            merged.bias.data = self.base_layer.bias.data.clone()
        return merged

    def get_lora_state_dict(self, prefix: str = "") -> dict:
        return {
            f"{prefix}lora_A.weight": self.lora_A.data,
            f"{prefix}lora_B.weight": self.lora_B.data,
            **(
                {
                    f"{prefix}lora_A2.weight": self.lora_A2.data,
                    f"{prefix}lora_B2.weight": self.lora_B2.data,
                }
                if self.dual
                else {}
            ),
        }

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias


class LoRAConv2d(nn.Module):
    """LoRA for Conv2d layers using low-rank factored convolutions."""

    def __init__(
        self,
        base_layer: nn.Conv2d,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_ch = base_layer.in_channels
        out_ch = base_layer.out_channels
        k = base_layer.kernel_size
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        self.lora_A = nn.Conv2d(
            in_ch, rank, kernel_size=k,
            stride=base_layer.stride, padding=base_layer.padding,
            dilation=base_layer.dilation, bias=False,
        ).to(device=device, dtype=dtype)
        self.lora_B = nn.Conv2d(rank, out_ch, kernel_size=1, bias=False).to(
            device=device, dtype=dtype
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.base_layer.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    def get_lora_state_dict(self, prefix: str = "") -> dict:
        return {
            f"{prefix}lora_A.weight": self.lora_A.weight.data,
            f"{prefix}lora_B.weight": self.lora_B.weight.data,
        }
