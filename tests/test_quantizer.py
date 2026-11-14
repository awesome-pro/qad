"""Unit tests for the reference quantizer and STE wrapper.

The tests are intentionally independent of the implementation where practical:
hand-computed values for tiny cases and a naive per-group loop reference for
random tensors. They encode the correctness properties listed in
RESEARCH_BRIEF.md section 4 and the failure modes in section 9.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen_qad.quantization import (
    FakeQuantLinear,
    QuantConfig,
    effective_bits_per_weight,
    fake_quantize_weight,
    group_count,
    quant_diagnostics,
    quantize_weight,
    storage_bits,
)

W3 = QuantConfig(bits=3, group_size=4)


def loop_reference(w: torch.Tensor, cfg: QuantConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive per-group quantizer used as an independent reference."""
    w = w.detach().float()
    out, inn = w.shape
    gs = min(cfg.group_size, inn)
    qmin, qmax = cfg.code_min, cfg.code_max
    qmax_sym = cfg.qmax_symmetric
    dq = torch.empty_like(w)
    codes = torch.empty_like(w)
    for row in range(out):
        for start in range(0, inn, gs):
            stop = min(start + gs, inn)
            x = w[row, start:stop]
            if cfg.scheme == "symmetric":
                scale = max(x.abs().max().item() / qmax_sym, cfg.scale_floor)
                c = torch.clamp(torch.round(x / scale), -qmax_sym, qmax_sym)
                codes[row, start:stop] = c
                dq[row, start:stop] = c * scale
            else:
                lo, hi = x.min().item(), x.max().item()
                if hi - lo == 0:
                    scale = max(x.abs().max().item() / qmax_sym, cfg.scale_floor)
                    zp = float((qmax + qmin) // 2)
                else:
                    scale = max((hi - lo) / (qmax - qmin), cfg.scale_floor)
                    zp = float(min(max(round(qmin - lo / scale), qmin), qmax))
                c = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
                codes[row, start:stop] = c
                dq[row, start:stop] = (c - zp) * scale
    return codes, dq


# ---------------------------------------------------------------------------
# hand-computed cases
# ---------------------------------------------------------------------------


def test_symmetric_hand_computed_w3():
    w = torch.tensor([[1.0, -1.0, 0.75, -0.25]])
    result = quantize_weight(w, W3, with_codes=True)

    # qmax = 3, scale = 1/3
    assert result.scale.item() == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert result.zero_point.item() == 0.0
    assert result.codes is not None
    assert result.codes[0].tolist() == [3.0, -3.0, 2.0, -1.0]
    expected = torch.tensor([[1.0, -1.0, 2.0 / 3.0, -1.0 / 3.0]])
    assert torch.allclose(result.qdq, expected, atol=1e-6)


def test_symmetric_hand_computed_w2():
    cfg = QuantConfig(bits=2, group_size=3)
    w = torch.tensor([[0.6, -0.6, 0.2]])
    result = quantize_weight(w, cfg, with_codes=True)

    # qmax = 1 -> scale = 0.6, codes {-1, 0, 1}
    assert result.scale.item() == pytest.approx(0.6, rel=1e-6)
    assert result.codes is not None
    assert result.codes[0].tolist() == [1.0, -1.0, 0.0]
    assert torch.allclose(result.qdq, torch.tensor([[0.6, -0.6, 0.0]]), atol=1e-6)


def test_two_bit_convention_uses_one_level():
    cfg = QuantConfig(bits=2, group_size=8)
    assert cfg.code_min == -1 and cfg.code_max == 1
    assert cfg.qmax_symmetric == 1


# ---------------------------------------------------------------------------
# shapes, grouping, and leakage
# ---------------------------------------------------------------------------


def test_group_shape_and_count():
    w = torch.randn(4, 384)
    cfg = QuantConfig(bits=3, group_size=128)
    result = quantize_weight(w, cfg, with_codes=True)

    assert result.qdq.shape == w.shape
    assert result.scale.shape == (4, 3, 1)
    assert result.zero_point.shape == (4, 3, 1)
    assert result.codes is not None and result.codes.shape == w.shape


def test_no_cross_row_grouping():
    torch.manual_seed(0)
    row_a = torch.randn(1, 256) * 1.0
    row_b = torch.randn(1, 256) * 1000.0
    cfg = QuantConfig(bits=3, group_size=128)

    both = quantize_weight(torch.cat([row_a, row_b], dim=0), cfg)
    alone = quantize_weight(row_b, cfg)

    assert torch.allclose(both.qdq[1], alone.qdq[0], atol=1e-5)
    assert not torch.allclose(both.scale[0], both.scale[1])


def test_tail_group_is_quantized_without_padding():
    torch.manual_seed(1)
    w = torch.randn(3, 130)
    cfg = QuantConfig(bits=3, group_size=128)

    result = quantize_weight(w, cfg, with_codes=True)
    ref_codes, ref_dq = loop_reference(w, cfg)

    assert result.qdq.shape == w.shape
    assert result.scale.shape == (3, 2, 1)  # one 128 group + one 2 group
    assert torch.allclose(result.qdq, ref_dq, atol=1e-5)
    assert result.codes is not None
    assert torch.allclose(result.codes, ref_codes, atol=1e-5)


def test_group_size_larger_than_input_is_one_group():
    torch.manual_seed(2)
    w = torch.randn(2, 32)
    cfg = QuantConfig(bits=3, group_size=128)
    result = quantize_weight(w, cfg)
    assert result.scale.shape == (2, 1, 1)
    assert result.qdq.shape == w.shape


def test_per_row_and_per_tensor_shapes():
    w = torch.randn(5, 64)
    per_row = quantize_weight(w, QuantConfig(bits=3, granularity="per_row"))
    per_tensor = quantize_weight(w, QuantConfig(bits=3, granularity="per_tensor"))
    assert per_row.scale.shape == (5, 1, 1)
    assert per_tensor.scale.shape == (1, 1, 1)

    ref_codes, ref_dq = loop_reference(w, QuantConfig(bits=3, group_size=64))
    assert torch.allclose(per_row.qdq, ref_dq, atol=1e-5)


# ---------------------------------------------------------------------------
# degenerate and extreme values
# ---------------------------------------------------------------------------


def test_all_zero_group_is_finite_and_zero():
    w = torch.zeros(2, 256)
    cfg = QuantConfig(bits=3, group_size=128)
    result = quantize_weight(w, cfg, with_codes=True)

    assert torch.isfinite(result.scale).all()
    assert torch.isfinite(result.qdq).all()
    assert torch.equal(result.qdq, torch.zeros_like(w))
    assert torch.allclose(result.scale, torch.full_like(result.scale, cfg.scale_floor))
    assert result.codes is not None
    assert torch.equal(result.codes, torch.zeros_like(w))


def test_constant_group_recovers_exactly_symmetric():
    cfg = QuantConfig(bits=3, group_size=128)
    for value in (0.7, -0.7, 1e-3):
        w = torch.full((2, 128), value)
        result = quantize_weight(w, cfg)
        assert torch.allclose(result.qdq, w, atol=1e-6)


def test_outliers_saturate_instead_of_overflowing():
    cfg = QuantConfig(bits=3, group_size=8)
    w = torch.zeros(1, 8)
    w[0, 0] = 100.0
    w[0, 1] = -0.01
    result = quantize_weight(w, cfg, with_codes=True)
    assert result.codes is not None
    assert result.codes.min() >= -3 and result.codes.max() <= 3
    assert result.codes[0, 0].item() == 3.0  # positive outlier saturates
    assert torch.isfinite(result.qdq).all()


def test_codes_in_declared_range_random():
    torch.manual_seed(3)
    w = torch.randn(6, 512) * 0.02
    for cfg in (
        QuantConfig(bits=3, group_size=128, scheme="symmetric"),
        QuantConfig(bits=2, group_size=64, scheme="symmetric"),
        QuantConfig(bits=3, group_size=128, scheme="asymmetric"),
        QuantConfig(bits=2, group_size=64, scheme="asymmetric"),
    ):
        result = quantize_weight(w, cfg, with_codes=True)
        assert result.codes is not None
        assert result.codes.min().item() >= cfg.code_min
        assert result.codes.max().item() <= cfg.code_max


# ---------------------------------------------------------------------------
# asymmetric scheme
# ---------------------------------------------------------------------------


def test_asymmetric_matches_loop_reference():
    torch.manual_seed(4)
    w = torch.randn(4, 256) * 0.5
    cfg = QuantConfig(bits=3, group_size=128, scheme="asymmetric")
    result = quantize_weight(w, cfg, with_codes=True)
    ref_codes, ref_dq = loop_reference(w, cfg)

    assert result.codes is not None
    assert torch.allclose(result.codes, ref_codes, atol=1e-5)
    assert torch.allclose(result.qdq, ref_dq, atol=1e-5)


def test_asymmetric_reconstruction_formula():
    torch.manual_seed(5)
    w = torch.randn(3, 128)
    cfg = QuantConfig(bits=4, group_size=128, scheme="asymmetric")
    result = quantize_weight(w, cfg, with_codes=True)
    assert result.codes is not None
    expected = (result.codes - result.zero_point.squeeze(-1)) * result.scale.squeeze(-1)
    assert torch.allclose(result.qdq, expected, atol=1e-6)
    assert (result.zero_point >= cfg.code_min).all()
    assert (result.zero_point <= cfg.code_max).all()


def test_asymmetric_constant_groups_recover_exactly():
    cfg = QuantConfig(bits=3, group_size=128, scheme="asymmetric")
    for value in (0.7, -0.7):
        w = torch.full((1, 128), value)
        result = quantize_weight(w, cfg)
        assert result.degenerate_groups == 1
        assert torch.allclose(result.qdq, w, atol=1e-6)


def test_asymmetric_zero_group_is_zero():
    cfg = QuantConfig(bits=3, group_size=128, scheme="asymmetric")
    result = quantize_weight(torch.zeros(1, 128), cfg)
    assert torch.equal(result.qdq, torch.zeros(1, 128))
    assert torch.isfinite(result.qdq).all()


# ---------------------------------------------------------------------------
# straight-through estimator and wrapper
# ---------------------------------------------------------------------------


def test_ste_is_identity_gradient():
    torch.manual_seed(6)
    w = torch.randn(8, 128, requires_grad=True)
    cfg = QuantConfig(bits=3, group_size=128)
    out = fake_quantize_weight(w, cfg)
    out.sum().backward()

    assert w.grad is not None
    assert torch.isfinite(w.grad).all()
    assert torch.equal(w.grad, torch.ones_like(w))


def test_ste_does_not_modify_master_weight():
    torch.manual_seed(7)
    w = torch.randn(4, 128, requires_grad=True)
    before = w.detach().clone()
    cfg = QuantConfig(bits=3, group_size=128)
    out = fake_quantize_weight(w, cfg)

    assert torch.equal(w.detach(), before)
    assert not torch.equal(out.detach(), before)  # forward is actually quantized


def test_fake_forward_equals_explicit_quantize_dequantize():
    torch.manual_seed(8)
    w = torch.randn(4, 128)
    cfg = QuantConfig(bits=3, group_size=128)
    assert torch.equal(fake_quantize_weight(w, cfg), quantize_weight(w, cfg).qdq)


def test_fake_quantize_preserves_dtype():
    w = torch.randn(2, 128, dtype=torch.bfloat16)
    out = fake_quantize_weight(w, QuantConfig(bits=3, group_size=128))
    assert out.dtype == torch.bfloat16


def test_fake_quant_linear_matches_functional():
    torch.manual_seed(9)
    linear = torch.nn.Linear(64, 32, bias=True)
    cfg = QuantConfig(bits=3, group_size=32)
    module = FakeQuantLinear.from_linear(linear, cfg, name="probe")
    x = torch.randn(5, 64)

    expected = F.linear(x, module.dequantized_weight(), module.bias)
    assert torch.allclose(module(x), expected, atol=1e-6)
    assert torch.allclose(module.bias, linear.bias, atol=0)


def test_fake_quant_linear_can_be_disabled():
    torch.manual_seed(10)
    linear = torch.nn.Linear(64, 32, bias=False)
    cfg = QuantConfig(bits=3, group_size=32)
    module = FakeQuantLinear.from_linear(linear, cfg)
    x = torch.randn(5, 64)

    module.quantized = False
    assert torch.allclose(module(x), F.linear(x, linear.weight), atol=1e-6)
    module.quantized = True
    assert not torch.allclose(module(x), F.linear(x, linear.weight), atol=1e-6)


def test_forward_does_not_overwrite_master_weight():
    torch.manual_seed(11)
    linear = torch.nn.Linear(32, 8, bias=True)
    module = FakeQuantLinear.from_linear(linear, QuantConfig(bits=3, group_size=32))
    before = module.weight.detach().clone()
    module(torch.randn(2, 32))
    assert torch.equal(module.weight.detach(), before)


def test_bf16_activations_with_fp32_master_weight_backward():
    torch.manual_seed(13)
    module = FakeQuantLinear(64, 16, bias=True, config=QuantConfig(bits=3, group_size=64))
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    out = module(x)

    assert out.dtype == torch.bfloat16
    out.float().sum().backward()
    assert module.weight.grad is not None
    assert module.weight.grad.dtype == torch.float32
    assert torch.isfinite(module.weight.grad).all()


# ---------------------------------------------------------------------------
# config, accounting, diagnostics
# ---------------------------------------------------------------------------


def test_config_roundtrip():
    cfg = QuantConfig(bits=3, group_size=128, scheme="asymmetric")
    assert QuantConfig.from_dict(cfg.to_dict()) == cfg


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        QuantConfig(bits=1)
    with pytest.raises(ValueError):
        QuantConfig(bits=9)
    with pytest.raises(ValueError):
        QuantConfig(scheme="fp8")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        QuantConfig(granularity="per_block")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        QuantConfig(scale_floor=0.0)


def test_config_rejects_unknown_field():
    with pytest.raises(ValueError):
        QuantConfig.from_dict({"bits": 3, "learning_rate": 1e-5})


def test_effective_bits_estimate():
    shape = (256, 2048)
    sym = effective_bits_per_weight(QuantConfig(bits=3, group_size=128), shape)
    asym = effective_bits_per_weight(
        QuantConfig(bits=3, group_size=128, scheme="asymmetric"), shape
    )
    assert sym == pytest.approx(3 + 16 / 128)
    assert asym == pytest.approx(3 + 32 / 128)
    per_row = effective_bits_per_weight(QuantConfig(bits=3, granularity="per_row"), shape)
    assert per_row == pytest.approx(3 + 16 / 2048)  # one scale per row of 2048 weights


def test_diagnostics_are_finite_and_bounded():
    torch.manual_seed(12)
    w = torch.randn(8, 256)
    result = quant_diagnostics(w, QuantConfig(bits=3, group_size=128))

    assert math.isfinite(result.mean_squared_error)
    assert result.mean_squared_error >= 0
    assert 0.0 <= result.endpoint_fraction <= 1.0
    assert 0.0 <= result.clipped_fraction <= 1.0
    assert 0.0 <= result.zero_level_fraction <= 1.0
    assert result.min_scale <= result.mean_scale <= result.max_scale
    assert result.bits_per_weight == pytest.approx(3.125)

# ---------------------------------------------------------------------------
# diagnostics: endpoint occupancy vs clipping vs zero level
# ---------------------------------------------------------------------------


def test_diagnostics_zero_level_fraction_symmetric():
    w = torch.tensor([[0.0, 0.0, 1.0, -1.0]])
    cfg = QuantConfig(bits=3, group_size=4)
    result = quant_diagnostics(w, cfg)

    assert result.zero_level_fraction == pytest.approx(0.5)  # two exact zeros
    assert result.endpoint_fraction == pytest.approx(0.5)  # +1 and -1 sit on 3 / -3
    assert result.clipped_fraction == 0.0


def test_diagnostics_endpoint_occupancy_is_not_clipping():
    # max-abs scaling puts the group extreme exactly on an endpoint; that is
    # structural, not clipping. 100 and -100 both round to +/-3, nothing is cut.
    w = torch.tensor([[100.0, -100.0, 0.5, -0.5]])
    cfg = QuantConfig(bits=3, group_size=4)
    result = quant_diagnostics(w, cfg)

    assert result.endpoint_fraction == pytest.approx(0.5)
    assert result.clipped_fraction == 0.0


def test_diagnostics_asymmetric_clipping_is_counted():
    # b=2 -> codes 0..3; scale=0.6, zero_point=2; +0.9 maps to raw code 4 and
    # is actually clipped, while -0.9 lands on code 0 (an endpoint, not a clip).
    w = torch.tensor([[-0.9, 0.9]])
    cfg = QuantConfig(bits=2, group_size=2, scheme="asymmetric")
    result = quantize_weight(w, cfg, with_codes=True)

    assert result.codes_raw is not None and result.clipped is not None
    assert result.codes_raw[0].tolist() == [0.0, 4.0]
    assert result.codes.tolist() == [[0.0, 3.0]]
    assert result.clipped[0].tolist() == [False, True]

    diag = quant_diagnostics(w, cfg)
    assert diag.clipped_fraction == pytest.approx(0.5)
    assert diag.endpoint_fraction == pytest.approx(1.0)
    assert diag.zero_level_fraction == 0.0  # zero-point is code 2, unused here


def test_diagnostics_all_zero_group_zero_level():
    w = torch.zeros(1, 128)
    cfg = QuantConfig(bits=3, group_size=128)
    diag = quant_diagnostics(w, cfg)
    assert diag.zero_level_fraction == pytest.approx(1.0)
    assert diag.endpoint_fraction == 0.0
    assert diag.clipped_fraction == 0.0


# ---------------------------------------------------------------------------
# exact storage accounting (tail groups included)
# ---------------------------------------------------------------------------


def test_group_count_exact_with_tail():
    cfg = QuantConfig(bits=3, group_size=128)
    assert group_count(cfg, (4, 384)) == 12
    assert group_count(cfg, (4, 130)) == 8  # 1 full group + 1 tail group per row
    assert group_count(cfg, (4, 64)) == 4  # group larger than in_features
    assert group_count(QuantConfig(granularity="per_row"), (4, 130)) == 4
    assert group_count(QuantConfig(granularity="per_tensor"), (4, 130)) == 1


def test_tail_group_storage_accounting_exact():
    cfg = QuantConfig(bits=3, group_size=128)
    bits = storage_bits(cfg, (4, 130))

    assert bits["weights"] == 4 * 130
    assert bits["groups"] == 8
    assert bits["payload_bits"] == 4 * 130 * 3
    assert bits["scale_bits"] == 8 * 16
    assert bits["total_bits"] == 4 * 130 * 3 + 8 * 16
    assert bits["bits_per_weight"] == pytest.approx((4 * 130 * 3 + 8 * 16) / (4 * 130))

    # the tail makes metadata more expensive per weight than a full group
    assert effective_bits_per_weight(cfg, (4, 130)) > effective_bits_per_weight(cfg, (4, 256))


def test_storage_bits_per_tensor_and_per_row():
    row = storage_bits(QuantConfig(bits=3, granularity="per_row"), (4, 130))
    assert row["groups"] == 4
    assert row["bits_per_weight"] == pytest.approx(3 + 16 / 130)

    tensor = storage_bits(QuantConfig(bits=3, granularity="per_tensor"), (4, 130))
    assert tensor["groups"] == 1
    assert tensor["bits_per_weight"] == pytest.approx(3 + 16 / (4 * 130))

    asym = storage_bits(QuantConfig(bits=3, group_size=128, scheme="asymmetric"), (256, 2048))
    assert asym["groups"] == 256 * 16
    assert asym["scale_bits"] == asym["zero_point_bits"] == 256 * 16 * 16


# ---------------------------------------------------------------------------
# device placement and dequantized-weight cache
# ---------------------------------------------------------------------------


def test_from_linear_preserves_device_and_bias_dtype():
    linear = nn.Linear(32, 8, bias=True).to(torch.device("cpu"))
    module = FakeQuantLinear.from_linear(linear, QuantConfig(bits=3, group_size=32))

    assert module.weight.device == linear.weight.device
    assert module.weight.dtype == torch.float32
    assert module.bias is not None
    assert module.bias.device == linear.bias.device
    assert module.bias.dtype == linear.bias.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_from_linear_preserves_cuda_device():
    linear = nn.Linear(32, 8, bias=True).cuda()
    module = FakeQuantLinear.from_linear(linear, QuantConfig(bits=3, group_size=32))
    assert module.weight.device == linear.weight.device
    assert module.bias is not None and module.bias.device == linear.bias.device


def test_dequant_cache_invalidated_by_in_place_update():
    torch.manual_seed(14)
    module = FakeQuantLinear(32, 8, bias=False, config=QuantConfig(bits=3, group_size=32))
    first = module.dequantized_weight().clone()
    assert module._dequant_cache is not None

    with torch.no_grad():
        module.weight.add_(0.05)
    second = module.dequantized_weight()

    assert not torch.equal(first, second)
    assert module._dequant_cache is not None
    assert module._dequant_cache[0] == module.weight._version


def test_invalidate_cache_after_data_assignment():
    torch.manual_seed(15)
    module = FakeQuantLinear(32, 8, bias=False, config=QuantConfig(bits=3, group_size=32))
    first = module.dequantized_weight().clone()

    with torch.no_grad():
        module.weight.data = module.weight.data + 0.1
    module.invalidate_cache()
    second = module.dequantized_weight()

    assert not torch.equal(first, second)


def test_no_grad_forward_uses_cache_and_refreshes_after_update():
    torch.manual_seed(16)
    module = FakeQuantLinear(32, 8, bias=False, config=QuantConfig(bits=3, group_size=32))
    x = torch.randn(2, 32)

    with torch.no_grad():
        out1 = module(x).clone()
        out1b = module(x)
        assert torch.equal(out1, out1b)

        version_before = module.weight._version
        module.weight.mul_(0.5)
        assert module.weight._version != version_before
        out2 = module(x)

    assert not torch.allclose(out1, out2)


def test_grad_enabled_forward_ignores_cache_and_keeps_ste():
    torch.manual_seed(17)
    module = FakeQuantLinear(32, 8, bias=False, config=QuantConfig(bits=3, group_size=32))
    with torch.no_grad():
        module(torch.randn(1, 32))
    assert module._dequant_cache is not None

    x = torch.randn(1, 32)
    out = module(x)
    assert out.requires_grad
    out.sum().backward()
    assert module.weight.grad is not None
    assert torch.isfinite(module.weight.grad).all()


def test_to_device_clears_cache():
    module = FakeQuantLinear(32, 8, bias=False, config=QuantConfig(bits=3, group_size=32))
    with torch.no_grad():
        module(torch.randn(1, 32))
    assert module._dequant_cache is not None
    module.to(torch.device("cpu"))
    assert module._dequant_cache is None
