import torch

from model.config import GPTConfig
from model.gpt import GPT


def make_model(vocab_size=10, block_size=8, n_layer=2, n_head=2, n_embd=8):
    config = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    return GPT(config)


def test_forward_output_shapes():
    model = make_model()
    idx = torch.randint(0, 10, (2, 5))
    logits, loss = model(idx)
    assert logits.shape == (2, 5, 10)
    assert loss is None


def test_loss_computed_only_when_targets_given():
    model = make_model()
    idx = torch.randint(0, 10, (2, 5))
    targets = torch.randint(0, 10, (2, 5))
    _, loss = model(idx, targets)
    assert loss.item() > 0


def test_causal_mask_blocks_future_tokens():
    """Changing a later token must never change an earlier position's logits —
    that's the entire point of causal masking."""
    model = make_model()
    model.eval()
    idx = torch.randint(0, 10, (1, 5))
    logits_a, _ = model(idx)

    idx_changed = idx.clone()
    idx_changed[0, -1] = (idx_changed[0, -1] + 1) % 10
    logits_b, _ = model(idx_changed)

    assert torch.allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-5)


def test_generate_appends_requested_number_of_tokens():
    model = make_model()
    idx = torch.randint(0, 10, (1, 3))
    out = model.generate(idx, max_new_tokens=4)
    assert out.shape == (1, 7)


def test_no_repeat_ngram_size_does_not_change_output_length():
    model = make_model()
    idx = torch.randint(0, 10, (1, 3))
    out = model.generate(idx, max_new_tokens=20, no_repeat_ngram_size=3)
    assert out.shape == (1, 23)


def test_no_repeat_ngram_size_breaks_up_greedy_loops():
    """Near-greedy sampling (very low temperature, top_k=1) is the failure
    case that motivated this: a small, undertrained model tends to collapse
    into repeating the same short pattern forever ("ondondond...")."""
    torch.manual_seed(0)
    model = make_model(vocab_size=10, block_size=32, n_layer=2, n_head=2, n_embd=16)
    idx = torch.randint(0, 10, (1, 3))
    out = model.generate(idx, max_new_tokens=60, temperature=0.01, top_k=1, no_repeat_ngram_size=3)
    tokens = out[0].tolist()
    # the same 3-gram must never repeat three times in a row
    for i in range(len(tokens) - 8):
        a, b, c = tokens[i:i + 3], tokens[i + 3:i + 6], tokens[i + 6:i + 9]
        assert not (a == b == c), f"3-gram {a} repeated three times in a row at position {i}"


def test_no_repeat_ngram_size_never_bans_every_candidate():
    """A tiny vocabulary makes it easy to accidentally ban every remaining
    token, which would zero out the softmax distribution and crash
    multinomial sampling. Must never happen, however aggressive the ban."""
    torch.manual_seed(0)
    model = make_model(vocab_size=3, block_size=16, n_layer=1, n_head=1, n_embd=8)
    idx = torch.randint(0, 3, (1, 2))
    out = model.generate(idx, max_new_tokens=30, temperature=0.01, top_k=1, no_repeat_ngram_size=1)
    assert out.shape == (1, 32)


def test_return_attn_shapes():
    model = make_model(n_layer=2, n_head=2)
    idx = torch.randint(0, 10, (1, 5))
    _, _, attn_maps = model(idx, return_attn=True)
    assert len(attn_maps) == 2
    for attn in attn_maps:
        assert attn.shape == (1, 2, 5, 5)


def test_attention_rows_sum_to_one():
    """Each token's attention weights are a softmax over the tokens it can see —
    they should always sum to 1, regardless of layer, head, or position."""
    model = make_model()
    model.eval()
    idx = torch.randint(0, 10, (1, 5))
    _, _, attn_maps = model(idx, return_attn=True)
    row_sums = attn_maps[0].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_attention_never_looks_ahead():
    model = make_model()
    model.eval()
    idx = torch.randint(0, 10, (1, 5))
    _, _, attn_maps = model(idx, return_attn=True)
    attn = attn_maps[0][0]  # (n_head, T, T)
    upper_triangle = torch.triu(torch.ones(5, 5), diagonal=1).bool()
    assert (attn[:, upper_triangle] == 0).all()


def test_return_attn_default_off_keeps_two_tuple_return():
    model = make_model()
    idx = torch.randint(0, 10, (1, 5))
    result = model(idx)
    assert len(result) == 2
