"""Generate text from a trained checkpoint — the qualitative "did it learn
anything" check, and the payoff step in the app: type a prompt, watch the
model continue it character by character.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model.gpt import GPT
from model.tokenizer import CharTokenizer
from engine.train import load_checkpoint

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent


def generate_text(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 20,
    device: str = "cpu",
    include_prompt: bool = True,
    no_repeat_ngram_size: int = 3,
) -> str:
    """Set include_prompt=False to get back only the newly generated text —
    useful when the prompt is long enough (e.g. retrieved context stuffed in
    front of it) that echoing it back would bury the model's actual output.

    no_repeat_ngram_size defaults on: low temperature/top_k make a small
    model more legible overall, but they also make it prone to getting stuck
    looping a short pattern ("ondondond...") instead of producing noise.
    Banning repeated 3-character sequences heads that off without needing to
    raise the randomness back up. Set to 0 to disable."""
    model.eval()
    model.to(device)

    ids = tokenizer.encode(prompt) or tokenizer.encode(" ")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    out_ids = out[0].tolist()
    if not include_prompt:
        out_ids = out_ids[len(ids):]
    return tokenizer.decode(out_ids)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Sample text from a trained checkpoint.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "base.pt"))
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter checkpoint to layer on top")
    parser.add_argument("--prompt", default="The lighthouse keeper")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3, help="0 disables")
    args = parser.parse_args()

    model, tokenizer = load_checkpoint(Path(args.checkpoint))

    if args.adapter:
        from engine.finetune_lora import load_adapter_onto

        load_adapter_onto(model, Path(args.adapter))
        print(f"(sampling with LoRA adapter from {args.adapter})")

    text = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print(text)


if __name__ == "__main__":
    _cli()
