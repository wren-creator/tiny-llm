"""A character-level tokenizer.

The simplest possible tokenizer: every unique character in the training text
becomes one token. No merges, no subwords, no external vocabulary — the whole
"vocabulary" is just the set of characters the model has ever seen, which makes
it easy to inspect and explain. Real-world models use subword tokenizers (BPE)
so a fixed vocabulary can cover any language efficiently; see ROADMAP.md.
"""

from __future__ import annotations

import json
import string
from dataclasses import dataclass

# A fixed, practical character set (not just "whatever the first text contained")
# so that a base model's vocabulary already covers digits, punctuation, etc. that
# might only show up later in a fine-tuning corpus or a user's own upload.
# Characters outside this set are dropped silently when encoding.
DEFAULT_CHARS = sorted(set(string.ascii_letters + string.digits + string.punctuation + " \n\t"))


@dataclass
class CharTokenizer:
    chars: list[str]

    def __post_init__(self) -> None:
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def default(cls) -> "CharTokenizer":
        return cls(chars=list(DEFAULT_CHARS))

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Builds a vocabulary from only the characters seen in `text`. Handy for a
        quick standalone experiment, but prefer `default()` whenever the tokenizer
        will later see text it wasn't built from (e.g. a later LoRA fine-tune corpus)."""
        chars = sorted(set(text))
        return cls(chars=chars)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def to_json(self) -> str:
        return json.dumps({"chars": self.chars})

    @classmethod
    def from_json(cls, blob: str) -> "CharTokenizer":
        data = json.loads(blob)
        return cls(chars=data["chars"])
