"""The pretraining loop: teaches a GPT from a random initialization to predict
the next character of the training text, one gradient step at a time.

Written as a plain importable function with an `on_step` callback so the same
code drives a terminal run (which just prints) and the Streamlit app (which
updates a live chart) — there's no separate "demo" version of this loop.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

from model.config import GPTConfig, TrainConfig
from model.gpt import GPT
from model.tokenizer import CharTokenizer

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent

OnStep = Callable[[int, float, float | None], None]


def get_batch(ids: np.ndarray, block_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(ids) - block_size - 1
    if max_start < 1:
        # Without this the negative bound surfaces as an opaque torch RuntimeError
        # from randint, which tells nobody what actually went wrong.
        raise ValueError(
            f"not enough text: this split has {len(ids)} tokens, but a block_size of "
            f"{block_size} needs at least {block_size + 2}. Use a longer text, or a smaller block_size."
        )
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([torch.from_numpy(ids[s:s + block_size].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(ids[s + 1:s + block_size + 1].astype(np.int64)) for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPT, ids: np.ndarray, block_size: int, batch_size: int, iters: int, device: str) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(ids, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def lr_at_step(step: int, learning_rate: float, warmup_steps: int, max_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return learning_rate * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def train(
    model_config: GPTConfig,
    train_config: TrainConfig,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    device: str = "cpu",
    on_step: OnStep | None = None,
) -> GPT:
    torch.manual_seed(train_config.seed)
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    for step in range(train_config.max_steps):
        lr = lr_at_step(step, train_config.learning_rate, train_config.warmup_steps, train_config.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        last_loss = None
        for _ in range(train_config.grad_accum_steps):
            x, y = get_batch(train_ids, model_config.block_size, train_config.batch_size, device)
            _, loss = model(x, y)
            (loss / train_config.grad_accum_steps).backward()
            last_loss = loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

        val_loss = None
        if step % train_config.eval_interval == 0 or step == train_config.max_steps - 1:
            val_loss = estimate_loss(
                model, val_ids, model_config.block_size, train_config.batch_size, train_config.eval_iters, device
            )

        if on_step is not None:
            on_step(step, last_loss, val_loss)

    return model


def save_checkpoint(path: Path, model: GPT, model_config: GPTConfig, tokenizer: CharTokenizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config.__dict__,
            "tokenizer_chars": tokenizer.chars,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[GPT, CharTokenizer]:
    ckpt = torch.load(path, map_location="cpu")
    model_config = GPTConfig(**ckpt["model_config"])
    model = GPT(model_config)
    model.load_state_dict(ckpt["model_state"])
    tokenizer = CharTokenizer(chars=ckpt["tokenizer_chars"])
    return model, tokenizer


def _cli() -> None:
    from data.prepare import load_ids

    parser = argparse.ArgumentParser(description="Pretrain a tiny GPT from scratch.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "tiny_cpu.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "prepared" / "sample_corpus"))
    parser.add_argument("--out", default=str(ROOT / "checkpoints" / "base.pt"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_dir = Path(args.data)
    tokenizer = CharTokenizer.from_json((data_dir / "tokenizer.json").read_text())
    train_ids = load_ids(data_dir / "train.bin")
    val_ids = load_ids(data_dir / "val.bin")

    model_config = GPTConfig(vocab_size=tokenizer.vocab_size, **cfg["model"])
    train_config = TrainConfig(**cfg["train"])

    def on_step(step: int, train_loss: float, val_loss: float | None) -> None:
        if val_loss is not None:
            print(f"step {step:5d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}")

    model = train(model_config, train_config, train_ids, val_ids, on_step=on_step)
    save_checkpoint(Path(args.out), model, model_config, tokenizer)
    print(f"saved checkpoint to {args.out}  ({model.num_params():,} params)")


if __name__ == "__main__":
    _cli()
