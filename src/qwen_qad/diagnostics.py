"""Gate 1.5 diagnostics: coverage, weight reconstruction, activations, memory.

These are measurement helpers, not training code. They are deliberately small
and auditable: every number written to the diagnostics JSON comes from one of
the functions here or from ``qwen_qad.losses``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from .quantization import (
    FakeQuantLinear,
    QuantConfig,
    quant_diagnostics,
    quantize_weight,
    storage_bits,
)


def quantization_coverage(model: nn.Module, config: QuantConfig) -> dict[str, Any]:
    """Quantized weight coverage and exact theoretical storage for a model."""
    modules = [m for m in model.modules() if isinstance(m, FakeQuantLinear)]
    quantized_weights = 0
    total_bits = 0.0
    payload_bits = 0.0
    bias_params = 0
    for module in modules:
        quantized_weights += int(module.weight.numel())
        bits = storage_bits(config, (module.out_features, module.in_features))
        total_bits += bits["total_bits"]
        payload_bits += bits["payload_bits"]
        if module.bias is not None:
            bias_params += int(module.bias.numel())

    total_params = sum(p.numel() for p in model.parameters())
    return {
        "quantized_modules": len(modules),
        "quantized_weight_parameters": quantized_weights,
        "quantized_bias_parameters": bias_params,
        "total_parameters": int(total_params),
        "fraction_of_model_parameters": quantized_weights / total_params,
        "effective_bits_per_quantized_weight": total_bits / quantized_weights,
        "payload_bytes": payload_bits / 8.0,
        "metadata_bytes": (total_bits - payload_bits) / 8.0,
        "total_quantized_bytes": total_bits / 8.0,
    }


def weight_diagnostics(model: nn.Module, config: QuantConfig) -> dict[str, Any]:
    """Aggregate reconstruction diagnostics over all wrapped modules."""
    modules = [(name, m) for name, m in model.named_modules() if isinstance(m, FakeQuantLinear)]
    total_weights = 0
    weight_weighted_mse = 0.0
    max_abs_error = 0.0
    endpoint = 0.0
    clipped = 0.0
    zero_level = 0.0
    degenerate = 0
    scale_min = math.inf
    scale_max = 0.0
    scale_mean = 0.0
    per_module: list[tuple[str, float, int]] = []

    for name, module in modules:
        with torch.no_grad():
            diag = quant_diagnostics(module.weight, config)
        n = int(module.weight.numel())
        total_weights += n
        weight_weighted_mse += diag.mean_squared_error * n
        max_abs_error = max(max_abs_error, diag.max_abs_error)
        endpoint += diag.endpoint_fraction * n
        clipped += diag.clipped_fraction * n
        zero_level += diag.zero_level_fraction * n
        degenerate += diag.degenerate_groups
        scale_min = min(scale_min, diag.min_scale)
        scale_max = max(scale_max, diag.max_scale)
        scale_mean += diag.mean_scale * n
        per_module.append((name, diag.mean_squared_error, n))

    per_module.sort(key=lambda item: item[1], reverse=True)
    return {
        "modules": len(modules),
        "weights": total_weights,
        "mse_weighted": weight_weighted_mse / max(total_weights, 1),
        "max_abs_error": max_abs_error,
        "endpoint_fraction": endpoint / max(total_weights, 1),
        "clipped_fraction": clipped / max(total_weights, 1),
        "zero_level_fraction": zero_level / max(total_weights, 1),
        "degenerate_groups": degenerate,
        "scale_min": scale_min if modules else None,
        "scale_max": scale_max if modules else None,
        "scale_mean_weighted": scale_mean / max(total_weights, 1) if modules else None,
        "worst_modules": [
            {"name": name, "mse": mse, "weights": n} for name, mse, n in per_module[:5]
        ],
    }


def capture_hidden_states(
    model: nn.Module, input_ids: Tensor, layer_indices: Iterable[int]
) -> tuple[Tensor, dict[int, Tensor]]:
    """Run the base model once and return last hidden states plus selected layers."""
    captures: dict[int, Tensor] = {}
    handles = []

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captures[index] = hidden.detach()

        return hook

    for index in layer_indices:
        handles.append(model.model.layers[index].register_forward_hook(make_hook(index)))
    try:
        with torch.no_grad():
            output = model.model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return output.last_hidden_state.detach(), captures


def relative_error(student: Tensor, teacher: Tensor) -> float:
    """L2 distance relative to the teacher norm (a magnitude-relative metric)."""
    student_f = student.float()
    teacher_f = teacher.float()
    return float((student_f - teacher_f).norm() / teacher_f.norm().clamp_min(1e-12))


def cosine_similarity(student: Tensor, teacher: Tensor) -> float:
    """Cosine between flattened activations; measures direction only."""
    student_f = student.float().flatten()
    teacher_f = teacher.float().flatten()
    return float(torch.nn.functional.cosine_similarity(student_f, teacher_f, dim=0))


def max_abs_error(student: Tensor, teacher: Tensor) -> float:
    return float((student.float() - teacher.float()).abs().max())


def compare_activations(
    student_layers: dict[int, Tensor], teacher_layers: dict[int, Tensor]
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for index in sorted(teacher_layers):
        results[str(index)] = {
            "relative_error": relative_error(student_layers[index], teacher_layers[index]),
            "max_abs_error": max_abs_error(student_layers[index], teacher_layers[index]),
            "cosine_similarity": cosine_similarity(
                student_layers[index], teacher_layers[index]
            ),
        }
    return results


def layer_relative_errors(
    student_layers: dict[int, Tensor], teacher_layers: dict[int, Tensor]
) -> dict[int, float]:
    return {index: relative_error(student_layers[index], teacher_layers[index]) for index in teacher_layers}


def quantized_weight_tensors(model: nn.Module) -> dict[str, Tensor]:
    """Master weights of all wrapped modules (for external checks)."""
    return {
        name: module.weight.detach()
        for name, module in model.named_modules()
        if isinstance(module, FakeQuantLinear)
    }


def reconstruction_check(model: nn.Module, config: QuantConfig) -> dict[str, float]:
    """One-module spot check: fake forward equals explicit quantize-dequantize."""
    module = next(m for m in model.modules() if isinstance(m, FakeQuantLinear))
    w = module.weight.detach()
    explicit = quantize_weight(w, config).qdq
    with torch.no_grad():
        cached = module.dequantized_weight()
    return {"max_abs_diff": float((explicit - cached).abs().max())}
