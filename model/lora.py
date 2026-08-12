"""LoRA: Low-Rank Adaptation.

The idea: instead of updating every weight in a pretrained model (expensive,
and it means you now have a whole second copy of a huge model per task),
freeze the base model completely and, next to a handful of its linear layers,
add two small trainable matrices A and B. The layer's output becomes

    base(x) + (x @ A^T @ B^T) * scaling

A and B are "low rank" — narrow matrices, `rank` wide instead of the full
layer width — so together they have far fewer numbers in them than the layer
they sit next to. Training only touches A and B. The base weights never move.

That's the whole mechanism. This file is deliberately small enough to read in
one sitting: `LoRALinear` is the adapter, `apply_lora` is what swaps it into
an existing model in place.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.config import LoRAConfig
from model.gpt import GPT


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear, freezing it and adding a trainable low-rank adapter."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B starts at zero so the adapter contributes nothing until trained —
        # the model behaves exactly like the base model on step zero.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        adapter_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + adapter_out * self.scaling

    def num_trainable_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora(model: GPT, config: LoRAConfig) -> int:
    """Freezes the whole model, then swaps `config.target_modules` attention
    projections (in every block) for LoRALinear adapters. Returns how many
    adapters were added — used to compute the trainable-parameter comparison
    the app shows to make the "far fewer parameters" claim concrete.
    """
    for p in model.parameters():
        p.requires_grad = False

    added = 0
    for block in model.blocks:
        attn = block.attn
        for name in config.target_modules:
            current = getattr(attn, name)
            if isinstance(current, LoRALinear):
                continue
            setattr(attn, name, LoRALinear(current, rank=config.rank, alpha=config.alpha, dropout=config.dropout))
            added += 1
    return added


def lora_state_dict(model: GPT) -> dict[str, torch.Tensor]:
    """Just the adapter weights — this is the whole point: it's a small file,
    not a second copy of the base model."""
    return {name: p.detach().clone() for name, p in model.named_parameters() if "lora_" in name}


def load_lora_state_dict(model: GPT, state_dict: dict[str, torch.Tensor]) -> None:
    """Loads adapter weights into a model that has already had apply_lora() called on it."""
    own_state = dict(model.named_parameters())
    for name, tensor in state_dict.items():
        own_state[name].data.copy_(tensor)
