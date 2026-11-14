"""Reference weight-only quantizer with fake quantization and STE.

Pilot convention (RESEARCH_BRIEF.md sections 3-4):

* signed symmetric, integer codes in ``[-qmax, qmax]``, ``qmax = 2**(b-1) - 1``
  (so one signed code is unused, e.g. W3 uses -3..3);
* group size 128 along ``in_features``, independent per output row;
* ``scale = max|W_group| / qmax`` with a numerical floor;
* no zero-point; scales are computed from the current master weight and are
  not learned.

An asymmetric affine variant (codes in ``[0, 2**b - 1]`` plus a zero-point) is
provided for comparison only. It is never selected implicitly.

This is a *reference* implementation. It does not pack bits and gives no
inference speedup; see brief section 4 ("Memory accounting"). Weight matrices
are assumed to have PyTorch layout ``[out_features, in_features]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

Granularity = Literal["per_tensor", "per_row", "per_group"]
Scheme = Literal["symmetric", "asymmetric"]

_SCHEMES: tuple[str, ...] = ("symmetric", "asymmetric")
_GRANULARITIES: tuple[str, ...] = ("per_tensor", "per_row", "per_group")


@dataclass(frozen=True)
class QuantConfig:
    """Declarative quantization state. Must survive save/reload unchanged."""

    bits: int = 3
    group_size: int = 128
    granularity: Granularity = "per_group"
    scheme: Scheme = "symmetric"
    scale_floor: float = 1e-8

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 8:
            raise ValueError(f"bits must be in [2, 8], got {self.bits}")
        if self.scheme not in _SCHEMES:
            raise ValueError(f"scheme must be one of {_SCHEMES}, got {self.scheme!r}")
        if self.granularity not in _GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {_GRANULARITIES}, got {self.granularity!r}"
            )
        if self.granularity == "per_group" and self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if not self.scale_floor > 0:
            raise ValueError(f"scale_floor must be > 0, got {self.scale_floor}")

    # ---- code ranges -----------------------------------------------------
    @property
    def qmax_symmetric(self) -> int:
        return 2 ** (self.bits - 1) - 1

    @property
    def code_min(self) -> int:
        return 0 if self.scheme == "asymmetric" else -self.qmax_symmetric

    @property
    def code_max(self) -> int:
        return 2**self.bits - 1 if self.scheme == "asymmetric" else self.qmax_symmetric

    @property
    def dequant_formula(self) -> str:
        if self.scheme == "symmetric":
            return "w_hat = code * scale"
        return "w_hat = (code - zero_point) * scale"

    # ---- serialization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QuantConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown QuantConfig field(s): {sorted(unknown)}")
        return cls(**{k: v for k, v in payload.items() if k in allowed})


def effective_bits_per_weight(cfg: QuantConfig, weight_shape: tuple[int, int]) -> float:
    """Upper-bound storage estimate: b + metadata bits / weights per group.

    Assumes one 16-bit scale per group (plus one 16-bit integer zero-point for
    the asymmetric scheme). Ignores tail groups and packing/alignment overhead.
    """
    out_features, in_features = weight_shape
    if cfg.granularity == "per_tensor":
        weights_per_group = out_features * in_features
    elif cfg.granularity == "per_row":
        weights_per_group = in_features
    else:
        weights_per_group = min(cfg.group_size, in_features)
    metadata_bits = 16.0 if cfg.scheme == "symmetric" else 32.0
    return cfg.bits + metadata_bits / weights_per_group


@dataclass
class QuantResult:
    """Output of :func:`quantize_weight`. All tensors are detached float32."""

    qdq: Tensor
    scale: Tensor
    zero_point: Tensor
    codes: Tensor | None = None
    degenerate_groups: int = 0


def _quantize_blocks(blocks: Tensor, cfg: QuantConfig) -> QuantResult:
    """Quantize ``[rows, n_groups, group_len]`` blocks along the last dim."""
    qmin, qmax = cfg.code_min, cfg.code_max

    if cfg.scheme == "symmetric":
        qmax_sym = cfg.qmax_symmetric
        amax = blocks.abs().amax(dim=-1, keepdim=True)
        scale = (amax / qmax_sym).clamp_min(cfg.scale_floor)
        zero = torch.zeros_like(scale)
        codes = torch.round(blocks / scale).clamp_(qmin, qmax)
        return QuantResult(qdq=codes * scale, scale=scale, zero_point=zero, codes=codes)

    # asymmetric affine: codes in [0, 2**b - 1], one zero-point per group
    wmin = blocks.amin(dim=-1, keepdim=True)
    wmax = blocks.amax(dim=-1, keepdim=True)
    span = wmax - wmin
    scale = (span / (qmax - qmin)).clamp_min(cfg.scale_floor)
    zero = (qmin - wmin / scale).round().clamp_(qmin, qmax)

    # A group with zero dynamic range cannot be represented by an affine grid
    # anchored at 0 (a constant negative group would dequantize to 0). Fall
    # back to a symmetric grid centred in the code range, which reproduces
    # constant groups exactly. Counted so it is visible in diagnostics.
    degenerate = span == 0
    n_degenerate = int(degenerate.sum().item())
    if n_degenerate:
        sym_scale = (blocks.abs().amax(dim=-1, keepdim=True) / cfg.qmax_symmetric).clamp_min(
            cfg.scale_floor
        )
        center = float((qmax + qmin) // 2)
        scale = torch.where(degenerate, sym_scale, scale)
        zero = torch.where(degenerate, torch.full_like(zero, center), zero)

    codes = (blocks / scale).round().add_(zero).clamp_(qmin, qmax)
    qdq = (codes - zero) * scale
    return QuantResult(
        qdq=qdq, scale=scale, zero_point=zero, codes=codes, degenerate_groups=n_degenerate
    )


def quantize_weight(
    weight: Tensor, cfg: QuantConfig, *, with_codes: bool = False
) -> QuantResult:
    """Quantize-dequantize a ``[out_features, in_features]`` weight matrix.

    Returns detached tensors; for a differentiable wrapper use
    :func:`fake_quantize_weight`. ``with_codes=True`` also returns integer
    codes (stored as float32) for diagnostics only.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    w = weight.detach().to(torch.float32)
    out_features, in_features = w.shape

    if cfg.granularity == "per_tensor":
        result = _quantize_blocks(w.reshape(1, 1, out_features * in_features), cfg)
        qdq = result.qdq.reshape(out_features, in_features)
        codes = result.codes.reshape(out_features, in_features) if result.codes is not None else None
    elif cfg.granularity == "per_row":
        result = _quantize_blocks(w.reshape(out_features, 1, in_features), cfg)
        qdq = result.qdq.reshape(out_features, in_features)
        codes = result.codes.reshape(out_features, in_features) if result.codes is not None else None
    else:
        gs = cfg.group_size
        if gs >= in_features:
            result = _quantize_blocks(w.reshape(out_features, 1, in_features), cfg)
            qdq = result.qdq.reshape(out_features, in_features)
            codes = (
                result.codes.reshape(out_features, in_features) if result.codes is not None else None
            )
        elif in_features % gs == 0:
            result = _quantize_blocks(w.reshape(out_features, in_features // gs, gs), cfg)
            qdq = result.qdq.reshape(out_features, in_features)
            codes = (
                result.codes.reshape(out_features, in_features) if result.codes is not None else None
            )
        else:
            # Tail group: quantize the full groups and the remainder separately
            # so that no padding value can leak into a group's max-abs.
            full = in_features // gs * gs
            head = w[:, :full].reshape(out_features, full // gs, gs)
            tail = w[:, full:].reshape(out_features, 1, in_features - full)
            r_head = _quantize_blocks(head, cfg)
            r_tail = _quantize_blocks(tail, cfg)
            qdq = torch.cat(
                [
                    r_head.qdq.reshape(out_features, full),
                    r_tail.qdq.reshape(out_features, in_features - full),
                ],
                dim=1,
            )
            scale = torch.cat([r_head.scale, r_tail.scale], dim=1)
            zero = torch.cat([r_head.zero_point, r_tail.zero_point], dim=1)
            codes = None
            if with_codes and r_head.codes is not None and r_tail.codes is not None:
                codes = torch.cat(
                    [
                        r_head.codes.reshape(out_features, full),
                        r_tail.codes.reshape(out_features, in_features - full),
                    ],
                    dim=1,
                )
            return QuantResult(
                qdq=qdq,
                scale=scale,
                zero_point=zero,
                codes=codes,
                degenerate_groups=r_head.degenerate_groups + r_tail.degenerate_groups,
            )

    return QuantResult(
        qdq=qdq,
        scale=result.scale,
        zero_point=result.zero_point,
        codes=codes if with_codes else None,
        degenerate_groups=result.degenerate_groups,
    )


def fake_quantize_weight(weight: Tensor, cfg: QuantConfig) -> Tensor:
    """Quantize-dequantize with a straight-through estimator (identity STE).

    The forward value is exactly the quantize-dequantize value; the backward
    pass is the identity, so gradients reach the high-precision master weight.
    """
    qdq = quantize_weight(weight, cfg).qdq.to(dtype=weight.dtype)
    if not weight.requires_grad:
        return qdq
    return weight + (qdq - weight).detach()


@dataclass
class QuantDiagnostics:
    mean_squared_error: float
    max_abs_error: float
    saturation_fraction: float
    zero_level_fraction: float
    mean_scale: float
    min_scale: float
    max_scale: float
    degenerate_groups: int
    bits_per_weight: float


def quant_diagnostics(weight: Tensor, cfg: QuantConfig) -> QuantDiagnostics:
    """Numerical diagnostics for one weight matrix (for reports, not training)."""
    w = weight.detach().to(torch.float32)
    result = quantize_weight(w, cfg, with_codes=True)
    assert result.codes is not None
    codes = result.codes
    error = result.qdq - w
    if cfg.scheme == "symmetric":
        saturated = codes.abs() == cfg.qmax_symmetric
        zero_level = torch.zeros_like(codes, dtype=torch.bool)
    else:
        saturated = (codes == cfg.code_min) | (codes == cfg.code_max)
        zero_level = codes == result.zero_point
    return QuantDiagnostics(
        mean_squared_error=float(error.pow(2).mean().item()),
        max_abs_error=float(error.abs().max().item()),
        saturation_fraction=float(saturated.float().mean().item()),
        zero_level_fraction=float(zero_level.float().mean().item()),
        mean_scale=float(result.scale.mean().item()),
        min_scale=float(result.scale.min().item()),
        max_scale=float(result.scale.max().item()),
        degenerate_groups=result.degenerate_groups,
        bits_per_weight=effective_bits_per_weight(cfg, tuple(w.shape)),
    )


class FakeQuantLinear(nn.Module):
    """Drop-in ``nn.Linear`` replacement holding an fp32 master weight.

    ``forward`` computes ``F.linear(x, fake_quantize(weight), bias)``. The
    master weight is never overwritten with its quantized version.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: QuantConfig | None = None,
        name: str = "",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantConfig()
        self.name = name
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, config: QuantConfig, name: str = ""
    ) -> "FakeQuantLinear":
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=config,
            name=name,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight.detach().float())
            if linear.bias is not None:
                module.bias.copy_(linear.bias.detach().float())
        return module

    @property
    def quantized(self) -> bool:
        return bool(getattr(self, "_quantized", True))

    @quantized.setter
    def quantized(self, value: bool) -> None:
        self._quantized = bool(value)

    def forward(self, x: Tensor) -> Tensor:
        weight = fake_quantize_weight(self.weight, self.config) if self.quantized else self.weight
        weight = weight.to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)

    @torch.no_grad()
    def dequantized_weight(self) -> Tensor:
        """Return the fake-quantized weight (detached, fp32)."""
        return quantize_weight(self.weight, self.config).qdq

    def extra_repr(self) -> str:
        bits = self.config.bits
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={bits}, scheme={self.config.scheme}, "
            f"group_size={self.config.group_size}, name={self.name!r}"
        )
