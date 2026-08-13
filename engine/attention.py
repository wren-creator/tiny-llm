"""Pull the attention weights out of a trained model for a given prompt —
the same numbers `CausalSelfAttention` computes internally on every forward
pass, just returned instead of thrown away, so they can be looked at.
"""

from __future__ import annotations

import torch

from model.gpt import GPT
from model.tokenizer import CharTokenizer


def get_attention_maps(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    device: str = "cpu",
) -> tuple[list[str], torch.Tensor]:
    """Returns (tokens, attn).

    tokens is the prompt as the model actually saw it (unknown characters
    dropped, same as tokenizer.encode). attn has shape (n_layer, n_head, T, T):
    attn[l, h, i, j] is how much token i attends to token j in layer l, head h.
    """
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt contains no characters in the model's vocabulary")
    tokens = [tokenizer.itos[i] for i in ids]

    model.eval()
    model.to(device)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _, _, attn_maps = model(idx, return_attn=True)

    attn = torch.stack(attn_maps, dim=0)[:, 0]  # (n_layer, n_head, T, T) — drop the batch dim
    return tokens, attn
