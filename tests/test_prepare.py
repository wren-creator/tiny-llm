import numpy as np
import pytest

from data.prepare import prepare_data
from engine.train import get_batch, lr_at_step
from model.tokenizer import CharTokenizer


def test_split_ratio():
    text = "x" * 100
    train_ids, val_ids, _ = prepare_data(text, val_fraction=0.2)
    assert len(train_ids) == 80
    assert len(val_ids) == 20


def test_default_tokenizer_used_when_none_given():
    _, _, tok = prepare_data("hello")
    assert tok.vocab_size == len(tok.chars) > 0


def test_reuses_given_tokenizer_instead_of_building_a_new_one():
    tok = CharTokenizer.default()
    _, _, returned = prepare_data("hello", tokenizer=tok)
    assert returned is tok


def test_get_batch_shapes():
    ids = np.arange(500, dtype=np.uint16)
    x, y = get_batch(ids, block_size=32, batch_size=4, device="cpu")
    assert x.shape == (4, 32)
    assert y.shape == (4, 32)


def test_get_batch_targets_are_inputs_shifted_by_one():
    ids = np.arange(500, dtype=np.uint16)
    x, y = get_batch(ids, block_size=32, batch_size=4, device="cpu")
    assert (y[:, :-1] == x[:, 1:]).all()


def test_get_batch_rejects_too_short_split_with_a_readable_error():
    """A split smaller than block_size used to surface as an opaque torch
    RuntimeError about randint bounds."""
    ids = np.arange(50, dtype=np.uint16)
    with pytest.raises(ValueError, match="not enough text"):
        get_batch(ids, block_size=128, batch_size=4, device="cpu")


def test_lr_warmup_then_cosine_decay():
    lr, warmup, total = 1e-3, 10, 100
    assert lr_at_step(0, lr, warmup, total) < lr          # ramping up
    assert lr_at_step(warmup - 1, lr, warmup, total) == lr  # peak at end of warmup
    assert lr_at_step(total - 1, lr, warmup, total) < lr / 100  # decayed to ~zero


def test_lr_at_step_handles_zero_warmup():
    assert lr_at_step(0, 1e-3, 0, 100) > 0
