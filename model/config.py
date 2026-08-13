from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Everything the model and training loop need to know about the model's shape.

    The same class describes a laptop-CPU toy run and a much bigger GPU run —
    scaling up means raising these numbers, not rewriting gpt.py.
    """

    vocab_size: int
    block_size: int = 128       # how many characters of context the model can see at once
    n_layer: int = 4            # how many transformer blocks stacked on top of each other
    n_head: int = 4             # how many attention heads per block
    n_embd: int = 128           # width of the model's internal representation
    dropout: float = 0.1


@dataclass
class TrainConfig:
    max_steps: int = 2000
    batch_size: int = 32
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    grad_accum_steps: int = 1
    eval_interval: int = 100
    eval_iters: int = 20
    grad_clip: float = 1.0
    seed: int = 1337


@dataclass
class LoRAConfig:
    rank: int = 4          # width of the low-rank adapter — the "how many new knobs" number
    alpha: float = 8.0     # scales the adapter's contribution
    dropout: float = 0.0
    target_modules: tuple = ("c_attn_q", "c_attn_v")  # which projections get adapters
    max_steps: int = 1000
    batch_size: int = 32
    learning_rate: float = 1e-3
    warmup_steps: int = 100
    eval_interval: int = 50
    eval_iters: int = 10
    grad_clip: float = 1.0
    seed: int = 1337
