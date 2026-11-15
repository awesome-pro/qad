"""Tests for quantization wrapping of a real (tiny) Qwen3 model."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_qad.models import (
    count_fake_quant_modules,
    set_quantization_enabled,
    set_trainable_quantized_only,
    wrap_quantized_linears,
)
from qwen_qad.quantization import QuantConfig

CFG = QuantConfig(bits=3, group_size=64)


def tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=96,
        intermediate_size=160,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config).eval()


def test_wrap_replaces_only_quantizable_projections():
    model = tiny_model()
    replaced = wrap_quantized_linears(model, CFG)

    assert len(replaced) == 14
    assert count_fake_quant_modules(model) == 14
    # the tied lm_head remains an ordinary Linear
    assert isinstance(model.lm_head, nn.Linear)
    assert sum(1 for m in model.modules() if isinstance(m, nn.Linear)) == 1


def test_master_weights_are_fp32_and_preserve_values():
    model = tiny_model()
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    wrap_quantized_linears(model, CFG)

    for name, param in model.named_parameters():
        if name.endswith("q_proj.weight"):
            assert param.dtype == torch.float32
            assert torch.equal(param.detach(), original[name])


def test_forward_shape_and_quantized_change():
    torch.manual_seed(0)
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        before = model(ids).logits.clone()
    wrap_quantized_linears(model, CFG)
    with torch.no_grad():
        after = model(ids).logits

    assert after.shape == before.shape
    assert torch.isfinite(after).all()
    assert not torch.allclose(after, before, atol=1e-5)


def test_state_dict_keys_unchanged_and_roundtrip_reproduces_logits():
    torch.manual_seed(1)
    model_a = tiny_model()
    keys_before = set(model_a.state_dict())
    wrap_quantized_linears(model_a, CFG)
    assert set(model_a.state_dict()) == keys_before

    model_b = tiny_model()
    wrap_quantized_linears(model_b, CFG)
    model_b.load_state_dict(model_a.state_dict())

    ids = torch.tensor([[5, 6, 7, 8]])
    with torch.no_grad():
        assert torch.allclose(model_a(ids).logits, model_b(ids).logits, atol=1e-6)


def test_only_master_weights_are_trainable():
    model = tiny_model()
    wrap_quantized_linears(model, CFG)
    trainable = set_trainable_quantized_only(model)

    manual = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad
    )
    assert trainable == manual
    assert trainable > 0

    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert any("embed_tokens" in n for n in frozen)
    assert any("layernorm" in n for n in frozen)


def test_gradients_reach_only_trainable_masters():
    model = tiny_model()
    wrap_quantized_linears(model, CFG)
    set_trainable_quantized_only(model)

    ids = torch.tensor([[1, 2, 3, 4]])
    model(ids).logits.sum().backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all()
        else:
            assert param.grad is None, name


def test_include_and_exclude_filters():
    model = tiny_model()
    replaced = wrap_quantized_linears(model, CFG, include=[r".*mlp.*"])
    assert len(replaced) == 6  # 3 MLP projections x 2 layers

    model = tiny_model()
    replaced = wrap_quantized_linears(model, CFG, exclude=[r".*down_proj"])
    assert len(replaced) == 12  # 14 - 2 down_proj


def test_wrapping_is_idempotent_on_quantizable_modules():
    model = tiny_model()
    first = wrap_quantized_linears(model, CFG)
    second = wrap_quantized_linears(model, CFG)
    assert len(first) == 14
    assert second == []  # FakeQuantLinear is not nn.Linear

def test_quantization_disabled_reproduces_original_logits():
    torch.manual_seed(2)
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        before = model(ids).logits.clone()

    wrap_quantized_linears(model, CFG)
    disabled = set_quantization_enabled(model, False)
    assert disabled == 14
    with torch.no_grad():
        after = model(ids).logits

    max_diff = float((after - before).abs().max())
    assert max_diff == 0.0, f"disabled quantization changed logits by {max_diff}"
