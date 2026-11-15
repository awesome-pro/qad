"""Tests for offline-QAD batching, one KL step and checkpointing."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_qad.checkpoint import load_checkpoint, save_checkpoint
from qwen_qad.models import set_quantization_enabled, set_trainable_quantized_only, wrap_quantized_linears
from qwen_qad.offline import (
    build_step_batch,
    evaluate_heldout,
    example_to_batch,
    forward_kl_step,
    make_example,
)
from qwen_qad.quantization import QuantConfig

CFG = QuantConfig(bits=3, group_size=32)


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


def record(prompt_len: int, response_len: int) -> dict:
    return {
        "prompt_ids": list(range(1, prompt_len + 1)),
        "response_ids": list(range(20, 20 + response_len)),
    }


def test_make_example_truncates_to_max_seq_len():
    example = make_example(record(prompt_len=4, response_len=10), max_seq_len=8, index=0)
    assert example is not None
    assert example.prompt_ids == [1, 2, 3, 4]
    assert example.response_ids == list(range(20, 24))  # 4-token response budget
    assert example.total_len == 8


def test_make_example_response_cap():
    example = make_example(record(prompt_len=4, response_len=10), max_seq_len=32, index=0, max_response_tokens=3)
    assert example is not None
    assert len(example.response_ids) == 3


def test_make_example_rejects_too_long_prompt():
    assert make_example(record(10, 4), max_seq_len=8, index=0) is None


def test_build_step_batch_respects_budget_and_has_no_duplicates():
    examples = [make_example(record(2, 4), 32, i) for i in range(6)]
    batch, cursor, tokens = build_step_batch(examples, 0, response_token_budget=10)
    assert 0 < len(batch) <= 4
    assert tokens <= 10 or len(batch) == 1
    assert len({e.index for e in batch}) == len(batch)
    assert cursor == (0 + len(batch)) % len(examples)


def test_build_step_batch_wraps_around():
    examples = [make_example(record(2, 4), 32, i) for i in range(3)]
    batch, cursor, tokens = build_step_batch(examples, 2, response_token_budget=32)
    assert len(batch) == 3
    assert cursor == 2


def test_build_step_batch_rejects_bad_budget():
    examples = [make_example(record(2, 4), 32, 0)]
    with pytest.raises(ValueError):
        build_step_batch(examples, 0, 0)


def test_forward_kl_step_gradients_and_teacher_freeze():
    torch.manual_seed(0)
    teacher = tiny_model()
    student = tiny_model()
    student.load_state_dict(teacher.state_dict())
    wrap_quantized_linears(student, CFG)
    set_trainable_quantized_only(student)
    set_quantization_enabled(student, True)
    for p in teacher.parameters():
        p.requires_grad_(False)

    example = make_example(record(4, 6), 32, 0)
    ids, mask, _, _ = example_to_batch(example, torch.device("cpu"))
    before = next(m for m in student.modules() if m.__class__.__name__ == "FakeQuantLinear").weight.detach().clone()

    loss, n = forward_kl_step(student, teacher, ids, mask, chunk_size=4)
    assert n == 6
    assert torch.isfinite(loss)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    after = next(m for m in student.modules() if m.__class__.__name__ == "FakeQuantLinear").weight.detach()
    assert not torch.equal(before, after)
    assert all(p.grad is None for p in teacher.parameters())


def test_evaluate_heldout_returns_finite_metrics():
    torch.manual_seed(1)
    teacher = tiny_model()
    student = tiny_model()
    student.load_state_dict(teacher.state_dict())
    wrap_quantized_linears(student, CFG)
    blocks = torch.randint(0, 128, (2, 16)).numpy()
    metrics = evaluate_heldout(student, teacher, blocks, batch_size=1, chunk_size=4)
    assert metrics["tokens"] == 2 * 15
    assert metrics["nll"] >= 0
    assert metrics["kl"] > 0


def test_checkpoint_roundtrip_restores_logits():
    torch.manual_seed(2)
    model = tiny_model()
    wrap_quantized_linears(model, CFG)
    set_trainable_quantized_only(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    ids = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        before = model(ids).logits.clone()
    size = save_checkpoint("/tmp/qad_ckpt_test.pt", model, optimizer, step=3, meta={"x": 1})
    assert size > 0

    fresh = tiny_model()
    wrap_quantized_linears(fresh, CFG)
    set_trainable_quantized_only(fresh)
    fresh_optimizer = torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad], lr=1e-3)
    payload = load_checkpoint("/tmp/qad_ckpt_test.pt", fresh, fresh_optimizer)
    assert payload["step"] == 3
    assert payload["meta"] == {"x": 1}
    with torch.no_grad():
        after = fresh(ids).logits
    assert torch.allclose(before, after, atol=1e-6)


def test_inference_weight_export_roundtrip(tmp_path):
    from qwen_qad.checkpoint import export_inference_weights, import_inference_weights

    torch.manual_seed(3)
    model = tiny_model()
    wrap_quantized_linears(model, CFG)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        before = model(ids).logits.clone()

    path = tmp_path / "weights.safetensors"
    size = export_inference_weights(path, model, metadata={"step": "7"})
    assert size > 0

    fresh = tiny_model()
    wrap_quantized_linears(fresh, CFG)
    missing = import_inference_weights(path, fresh)
    assert all(key.startswith("lm_head.") for key in missing)
    with torch.no_grad():
        after = fresh(ids).logits
    assert torch.allclose(before, after, atol=1e-6)


def test_verified_checkpoint_rejects_wrong_metadata(tmp_path):
    from qwen_qad.checkpoint import load_checkpoint_verified, save_checkpoint

    model = tiny_model()
    wrap_quantized_linears(model, CFG)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    meta = {
        "revisions": {"resolved_model_sha": "abc"},
        "quantization": {"bits": 3, "group_size": 32, "scheme": "symmetric"},
    }
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, step=3, meta=meta)

    fresh = tiny_model()
    wrap_quantized_linears(fresh, CFG)
    got = load_checkpoint_verified(path, fresh, "abc", 3, 32)
    assert got["quantization"]["bits"] == 3

    for kwargs in (
        {"expected_model_sha": "wrong"},
        {"expected_model_sha": "abc", "expected_bits": 4, "expected_group_size": 32},
        {"expected_model_sha": "abc", "expected_bits": 3, "expected_group_size": 128},
    ):
        kwargs.setdefault("expected_model_sha", "abc")
        with pytest.raises(ValueError):
            load_checkpoint_verified(path, tiny_model(), **kwargs)
