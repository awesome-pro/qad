"""Tests for coverage, weight and activation diagnostics."""

from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_qad.diagnostics import (
    capture_hidden_states,
    compare_activations,
    quantization_coverage,
    relative_error,
    weight_diagnostics,
)
from qwen_qad.models import set_quantization_enabled, wrap_quantized_linears
from qwen_qad.quantization import QuantConfig, storage_bits

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


def test_coverage_matches_wrapped_modules():
    model = tiny_model()
    wrap_quantized_linears(model, CFG)
    coverage = quantization_coverage(model, CFG)

    expected = 0
    for module in model.modules():
        if hasattr(module, "config") and module.__class__.__name__ == "FakeQuantLinear":
            expected += module.weight.numel()
    assert coverage["quantized_modules"] == 14
    assert coverage["quantized_weight_parameters"] == expected
    assert 0 < coverage["fraction_of_model_parameters"] < 1
    assert coverage["total_parameters"] > coverage["quantized_weight_parameters"]


def test_coverage_effective_bits_use_exact_group_accounting():
    model = tiny_model()
    wrap_quantized_linears(model, CFG)
    coverage = quantization_coverage(model, CFG)

    total_bits = 0.0
    for module in model.modules():
        if module.__class__.__name__ == "FakeQuantLinear":
            total_bits += storage_bits(CFG, (module.out_features, module.in_features))["total_bits"]
    assert coverage["total_quantized_bytes"] == total_bits / 8.0
    assert coverage["effective_bits_per_quantized_weight"] > 3.0


def test_weight_diagnostics_are_finite():
    model = tiny_model()
    wrap_quantized_linears(model, CFG)
    diagnostics = weight_diagnostics(model, CFG)

    assert diagnostics["modules"] == 14
    assert diagnostics["mse_weighted"] > 0
    assert 0 <= diagnostics["endpoint_fraction"] <= 1
    assert diagnostics["clipped_fraction"] == 0.0  # symmetric max-abs never clips
    assert len(diagnostics["worst_modules"]) <= 5


def test_activation_capture_and_comparison():
    torch.manual_seed(0)
    teacher = tiny_model()
    student = tiny_model()
    student.load_state_dict(teacher.state_dict())
    wrap_quantized_linears(student, CFG)
    set_quantization_enabled(student, True)

    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    teacher_hidden, teacher_layers = capture_hidden_states(teacher, ids, [0, 1])
    student_hidden, student_layers = capture_hidden_states(student, ids, [0, 1])

    assert teacher_hidden.shape == student_hidden.shape
    assert set(teacher_layers) == {0, 1}
    comparisons = compare_activations(student_layers, teacher_layers)
    assert set(comparisons) == {"0", "1"}
    assert comparisons["0"]["relative_error"] > 0

    same = compare_activations(teacher_layers, teacher_layers)
    assert same["0"]["relative_error"] == 0.0
    assert relative_error(teacher_hidden, teacher_hidden) == 0.0


def test_quantization_disabled_reproduces_reference_activations():
    torch.manual_seed(1)
    teacher = tiny_model()
    student = tiny_model()
    student.load_state_dict(teacher.state_dict())
    wrap_quantized_linears(student, CFG)
    set_quantization_enabled(student, False)

    ids = torch.tensor([[1, 2, 3, 4]])
    teacher_hidden, teacher_layers = capture_hidden_states(teacher, ids, [0, 1])
    student_hidden, student_layers = capture_hidden_states(student, ids, [0, 1])

    assert torch.allclose(teacher_hidden, student_hidden, atol=0.0)
    assert torch.allclose(teacher_layers[0], student_layers[0], atol=0.0)
