"""A minimal decoder-only Transformer (GPT-style), written from scratch.

No HuggingFace `transformers` import anywhere in this file on purpose — every
line here is something you can read top to bottom and understand, rather than
a black box. It's the same architecture (causal self-attention + MLP blocks)
that powers real large language models; the only difference from something
like GPT-3 is the numbers in GPTConfig.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Lets each token look at every token before it (and itself), never ahead.

    Query/key/value are kept as separate Linear layers (rather than one fused
    matrix) specifically so LoRA can target the query and value projections
    individually later, matching the original LoRA paper.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd)
        self.c_attn_k = nn.Linear(config.n_embd, config.n_embd)
        self.c_attn_v = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, C = x.shape

        q = self.c_attn_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_attn_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_attn_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        # captured post-softmax, pre-dropout — the actual "how much does token i
        # attend to token j" weights, before dropout randomly zeroes some of them
        attn_weights = att.detach() if return_attn else None
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(out)), attn_weights


class MLP(nn.Module):
    """The per-token feed-forward network that follows attention in every block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """One transformer block: attend to context, then think per-token, with residual connections."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, attn_weights = self.attn(self.ln_1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_attn: bool = False):
        B, T = idx.shape
        assert T <= self.config.block_size, "input longer than block_size"

        pos = torch.arange(0, T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        attn_maps = [] if return_attn else None
        for block in self.blocks:
            x, attn_weights = block(x, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn_weights)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_attn:
            return logits, loss, attn_maps
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
