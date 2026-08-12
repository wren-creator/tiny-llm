"""Turn raw text into the token id arrays the training loop reads.

This is the "tokenization" stage: every character in the input text becomes
an integer (via CharTokenizer), and the resulting sequence is split into a
training slice and a small held-out validation slice, so training can report
how well the model generalizes to text it wasn't trained on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model.tokenizer import CharTokenizer

THIS_DIR = Path(__file__).resolve().parent


def prepare_data(
    text: str,
    tokenizer: CharTokenizer | None = None,
    val_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, CharTokenizer]:
    if tokenizer is None:
        tokenizer = CharTokenizer.default()

    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    split = int(len(ids) * (1 - val_fraction))
    train_ids, val_ids = ids[:split], ids[split:]
    return train_ids, val_ids, tokenizer


def save_prepared(outdir: Path, train_ids: np.ndarray, val_ids: np.ndarray, tokenizer: CharTokenizer) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    train_ids.tofile(outdir / "train.bin")
    val_ids.tofile(outdir / "val.bin")
    (outdir / "tokenizer.json").write_text(tokenizer.to_json())


def load_ids(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint16)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Prepare a text file for training.")
    parser.add_argument("--input", default=str(THIS_DIR / "raw" / "sample_corpus.txt"))
    parser.add_argument("--outdir", default=str(THIS_DIR / "prepared" / "sample_corpus"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    text = Path(args.input).read_text()
    train_ids, val_ids, tokenizer = prepare_data(text, val_fraction=args.val_fraction)
    save_prepared(Path(args.outdir), train_ids, val_ids, tokenizer)

    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"train tokens: {len(train_ids)}  val tokens: {len(val_ids)}")
    print(f"wrote {args.outdir}/train.bin, val.bin, tokenizer.json")


if __name__ == "__main__":
    _cli()
