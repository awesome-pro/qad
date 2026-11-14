"""Tests for the checkpoint-derived model inventory."""

from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_qad.models import (
    EXPECTED_PARAMETERS,
    build_inventory,
    is_quantizable_name,
    summarize_inventory,
)


def tiny_model() -> Qwen3ForCausalLM:
    # hidden 96 with 8 heads x head_dim 16 -> q_proj output 128 != hidden 96,
    # mirroring the Qwen3-0.6B shape asymmetry.
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


def test_quantizable_name_matching():
    assert is_quantizable_name("model.layers.0.self_attn.q_proj")
    assert is_quantizable_name("model.layers.27.mlp.down_proj")
    assert not is_quantizable_name("model.embed_tokens")
    assert not is_quantizable_name("lm_head")
    assert not is_quantizable_name("model.layers.0.input_layernorm")
    assert not is_quantizable_name("model.layers.0.self_attn.q_norm")


def test_inventory_counts_and_shapes():
    model = tiny_model()
    inventory = build_inventory(model, tokenizer=None, model_id="tiny", revision="test")

    total = sum(p.numel() for p in model.parameters())
    assert inventory["totals"]["parameters"] == total
    # 14 projections plus the tied lm_head Linear (not quantizable)
    assert inventory["linear_modules"]["count"] == 2 * 7 + 1
    assert inventory["linear_modules"]["quantized_count"] == 2 * 7
    assert inventory["tied_embeddings"] is True
    assert inventory["warnings"] == []

    by_name = {m["name"]: m for m in inventory["linear_modules"]["modules"]}
    assert by_name["model.layers.0.self_attn.q_proj"]["shape"] == [128, 96]
    assert by_name["model.layers.0.self_attn.k_proj"]["shape"] == [64, 96]
    assert by_name["model.layers.0.self_attn.o_proj"]["shape"] == [96, 128]
    assert by_name["model.layers.0.mlp.gate_proj"]["shape"] == [160, 96]
    assert inventory["totals"]["quantizable_parameters"] == sum(
        m["parameters"] for m in inventory["linear_modules"]["modules"] if m["quantizable"]
    )


def test_inventory_category_totals_sum_to_total():
    inventory = build_inventory(tiny_model(), None, "tiny", "test")
    assert sum(inventory["totals"]["categories"].values()) == inventory["totals"]["parameters"]


def test_inventory_warns_on_expectation_mismatch():
    inventory = build_inventory(tiny_model(), None, "Qwen/Qwen3-0.6B", "test")
    assert inventory["warnings"]
    assert any("total parameters" in w for w in inventory["warnings"])


def test_expected_parameters_are_positive():
    for model_id, expected in EXPECTED_PARAMETERS.items():
        assert expected["quantizable"] < expected["total"]
        assert expected["total"] > 0


def test_summary_is_human_readable():
    text = summarize_inventory(build_inventory(tiny_model(), None, "tiny", "rev"))
    assert "parameters" in text
    assert "linear modules" in text
    assert "14 quantizable" in text
