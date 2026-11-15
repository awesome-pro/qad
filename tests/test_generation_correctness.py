"""On-policy correctness checks: current weights, modes, alignment."""

from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_qad.losses import gather_scored_tokens, next_token_score_mask
from qwen_qad.models import set_quantization_enabled, set_trainable_quantized_only, wrap_quantized_linears
from qwen_qad.quantization import FakeQuantLinear, QuantConfig
from qwen_qad.rollouts import generate_rollouts


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 9


def tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config).eval()


def test_rollouts_use_current_master_weights_after_update():
    torch.manual_seed(0)
    model = tiny_model()
    wrap_quantized_linears(model, QuantConfig(bits=3, group_size=32))
    set_trainable_quantized_only(model)
    set_quantization_enabled(model, True)
    model.train()
    tokenizer = StubTokenizer()
    prompts = [[1, 2, 3, 4], [5, 6, 7, 8]]

    first = generate_rollouts(model, tokenizer, prompts, max_new_tokens=8, batch_size=2, seed=0)
    assert model.training is False  # generation puts the model in eval mode
    assert all(r["full_ids"][: len(r["prompt_ids"])] == r["prompt_ids"] for r in first)
    assert all(0 not in r["response_ids"] for r in first)

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, FakeQuantLinear):
                module.weight.add_(1.0)
                module.invalidate_cache()
    second = generate_rollouts(model, tokenizer, prompts, max_new_tokens=8, batch_size=2, seed=0)
    assert [r["response_ids"] for r in first] != [r["response_ids"] for r in second]


def test_rollout_scores_only_response_tokens():
    torch.manual_seed(1)
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    prompt_len, response_len = 3, 3
    mask = next_token_score_mask(
        torch.tensor([prompt_len]), torch.tensor([response_len]), ids.shape[1]
    )
    hidden = torch.randn(1, 6, 64)
    scored, targets = gather_scored_tokens(hidden, ids, mask)
    assert scored.shape[0] == response_len
    assert targets.tolist() == ids[0, prompt_len : prompt_len + response_len].tolist()
