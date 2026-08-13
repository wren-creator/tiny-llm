import torch

from model.config import GPTConfig, LoRAConfig
from model.gpt import GPT
from model.lora import apply_lora, load_lora_state_dict, lora_state_dict


def make_model():
    config = GPTConfig(vocab_size=10, block_size=8, n_layer=2, n_head=2, n_embd=8)
    return GPT(config)


def test_apply_lora_adds_one_adapter_per_target_module_per_block():
    model = make_model()
    config = LoRAConfig(target_modules=("c_attn_q", "c_attn_v"))
    added = apply_lora(model, config)
    assert added == model.config.n_layer * len(config.target_modules)


def test_base_weights_frozen_after_apply_lora():
    model = make_model()
    apply_lora(model, LoRAConfig())
    for name, p in model.named_parameters():
        if "lora_" not in name:
            assert not p.requires_grad, name


def test_lora_weights_trainable_after_apply_lora():
    model = make_model()
    apply_lora(model, LoRAConfig())
    lora_params = [p for name, p in model.named_parameters() if "lora_" in name]
    assert lora_params
    assert all(p.requires_grad for p in lora_params)


def test_trainable_param_count_matches_lora_state_dict():
    model = make_model()
    apply_lora(model, LoRAConfig(rank=4))
    assert model.num_trainable_params() == sum(p.numel() for p in lora_state_dict(model).values())


def test_zero_init_adapter_leaves_output_unchanged():
    """lora_B starts at zero, so a freshly-adapted model must behave identically
    to the un-adapted base model until it's actually trained."""
    torch.manual_seed(0)
    model = make_model()
    model.eval()
    idx = torch.randint(0, 10, (1, 5))
    base_logits, _ = model(idx)

    apply_lora(model, LoRAConfig())
    model.eval()
    adapted_logits, _ = model(idx)

    assert torch.allclose(base_logits, adapted_logits, atol=1e-5)


def test_load_lora_state_dict_restores_trained_weights():
    torch.manual_seed(0)
    model_a = make_model()
    apply_lora(model_a, LoRAConfig(rank=4))
    with torch.no_grad():
        for p in model_a.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)
    state = lora_state_dict(model_a)

    # same seed so model_b's base weights match model_a's exactly — only the
    # adapter weights should matter for this comparison, so hold base fixed
    torch.manual_seed(0)
    model_b = make_model()
    apply_lora(model_b, LoRAConfig(rank=4))
    load_lora_state_dict(model_b, state)

    idx = torch.randint(0, 10, (1, 5))
    model_a.eval()
    model_b.eval()
    out_a, _ = model_a(idx)
    out_b, _ = model_b(idx)
    assert torch.allclose(out_a, out_b, atol=1e-5)
