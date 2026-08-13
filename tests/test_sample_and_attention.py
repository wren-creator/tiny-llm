import pytest

from engine.attention import get_attention_maps
from engine.sample import generate_text
from model.config import GPTConfig
from model.gpt import GPT
from model.tokenizer import CharTokenizer

BLOCK_SIZE = 32


def make_model_and_tokenizer():
    tok = CharTokenizer.default()
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=BLOCK_SIZE, n_layer=2, n_head=2, n_embd=16))
    return model, tok


def test_attention_map_shape_matches_prompt():
    model, tok = make_model_and_tokenizer()
    tokens, attn = get_attention_maps(model, tok, "hello")
    assert tokens == list("hello")
    assert attn.shape == (2, 2, 5, 5)


def test_attention_crops_prompt_longer_than_block_size():
    """Used to raise a bare AssertionError from GPT.forward. Now it crops to the
    last block_size characters, the same way generate() does."""
    model, tok = make_model_and_tokenizer()
    tokens, attn = get_attention_maps(model, tok, "x" * (BLOCK_SIZE + 50))
    assert len(tokens) == BLOCK_SIZE
    assert attn.shape == (2, 2, BLOCK_SIZE, BLOCK_SIZE)


def test_attention_rejects_prompt_with_no_known_characters():
    model, tok = make_model_and_tokenizer()
    with pytest.raises(ValueError, match="no characters"):
        get_attention_maps(model, tok, "日本語")


def test_generate_includes_prompt_by_default():
    model, tok = make_model_and_tokenizer()
    out = generate_text(model, tok, "The", max_new_tokens=5)
    assert out.startswith("The")
    assert len(out) == len("The") + 5


def test_generate_can_return_continuation_only():
    model, tok = make_model_and_tokenizer()
    out = generate_text(model, tok, "The", max_new_tokens=5, include_prompt=False)
    assert len(out) == 5


def test_generate_handles_prompt_longer_than_block_size():
    model, tok = make_model_and_tokenizer()
    long_prompt = "y" * (BLOCK_SIZE + 20)
    out = generate_text(model, tok, long_prompt, max_new_tokens=5, include_prompt=False)
    assert len(out) == 5
