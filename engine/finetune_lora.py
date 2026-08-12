"""LoRA fine-tuning: takes an already-pretrained base checkpoint, freezes it,
and trains only a small set of adapter parameters against a new text.

Reuses get_batch/estimate_loss/lr_at_step from engine/train.py — fine-tuning
is the same gradient-descent loop as pretraining, just aimed at a different
(tiny) slice of trainable parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

from model.config import LoRAConfig
from model.gpt import GPT
from model.lora import apply_lora, lora_state_dict, load_lora_state_dict
from model.tokenizer import CharTokenizer
from engine.train import estimate_loss, get_batch, lr_at_step, load_checkpoint

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent

OnStep = Callable[[int, float, float | None], None]


def finetune_lora(
    model: GPT,
    lora_config: LoRAConfig,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    device: str = "cpu",
    on_step: OnStep | None = None,
) -> tuple[GPT, int]:
    """Mutates `model` in place (adapters are swapped into its attention layers)
    and returns it, along with how many adapter layers were added."""
    torch.manual_seed(lora_config.seed)
    added = apply_lora(model, lora_config)
    model.to(device)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lora_config.learning_rate)

    for step in range(lora_config.max_steps):
        lr = lr_at_step(step, lora_config.learning_rate, lora_config.warmup_steps, lora_config.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        x, y = get_batch(train_ids, model.config.block_size, lora_config.batch_size, device)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, lora_config.grad_clip)
        optimizer.step()

        val_loss = None
        if step % lora_config.eval_interval == 0 or step == lora_config.max_steps - 1:
            val_loss = estimate_loss(
                model, val_ids, model.config.block_size, lora_config.batch_size, lora_config.eval_iters, device
            )

        if on_step is not None:
            on_step(step, loss.item(), val_loss)

    return model, added


def save_adapter_checkpoint(path: Path, model: GPT, lora_config: LoRAConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state": lora_state_dict(model),
            "lora_config": {**lora_config.__dict__, "target_modules": list(lora_config.target_modules)},
        },
        path,
    )


def load_adapter_onto(model: GPT, adapter_path: Path) -> LoRAConfig:
    """Applies a saved adapter to a freshly loaded base model (in place)."""
    ckpt = torch.load(adapter_path, map_location="cpu")
    lora_config = LoRAConfig(**{**ckpt["lora_config"], "target_modules": tuple(ckpt["lora_config"]["target_modules"])})
    apply_lora(model, lora_config)
    load_lora_state_dict(model, ckpt["adapter_state"])
    return lora_config


def _cli() -> None:
    from data.prepare import load_ids

    parser = argparse.ArgumentParser(description="LoRA fine-tune a pretrained tiny GPT.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "base.pt"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "lora_default.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "prepared" / "finetune_sample"))
    parser.add_argument("--out", default=str(ROOT / "checkpoints" / "adapter.pt"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["lora"]
    cfg["target_modules"] = tuple(cfg["target_modules"])
    lora_config = LoRAConfig(**cfg)

    model, tokenizer = load_checkpoint(Path(args.checkpoint))
    base_params = model.num_params()

    data_dir = Path(args.data)
    train_ids = load_ids(data_dir / "train.bin")
    val_ids = load_ids(data_dir / "val.bin")

    def on_step(step: int, train_loss: float, val_loss: float | None) -> None:
        if val_loss is not None:
            print(f"step {step:5d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}")

    model, added = finetune_lora(model, lora_config, train_ids, val_ids, on_step=on_step)
    adapter_params = model.num_trainable_params()

    save_adapter_checkpoint(Path(args.out), model, lora_config)
    pct = 100 * adapter_params / base_params
    print(f"added LoRA adapters to {added} layers")
    print(f"base model params: {base_params:,}  trainable adapter params: {adapter_params:,}  ({pct:.2f}%)")
    print(f"saved adapter to {args.out}")


if __name__ == "__main__":
    _cli()
