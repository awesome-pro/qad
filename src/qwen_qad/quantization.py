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

import math
from dataclasses import asdict, dataclass
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


def group_count(cfg: QuantConfig, weight_shape: tuple[int, int]) -> int:
    """Exact number of quantization groups for a ``[out, in]`` weight.

    Tail groups count as full groups for metadata purposes: a shape of
    ``(4, 130)`` with group size 128 has **two** groups per row (128 + 2).
    """
    out_features, in_features = weight_shape
    if out_features <= 0 or in_features <= 0:
        raise ValueError(f"weight shape must be positive, got {weight_shape}")
    if cfg.granularity == "per_tensor":
        return 1
    if cfg.granularity == "per_row":
        return out_features
    return out_features * math.ceil(in_features / cfg.group_size)


def storage_bits(cfg: QuantConfig, weight_shape: tuple[int, int]) -> dict[str, float]:
    """Exact bit accounting for one weight matrix (no packing alignment).

    ``payload_bits`` is ``weights * bits``; metadata is one 16-bit scale per
    group and, for the asymmetric scheme, one 16-bit zero-point per group.
    Tail groups are counted exactly via :func:`group_count`.
    """
    out_features, in_features = weight_shape
    weights = out_features * in_features
    groups = group_count(cfg, weight_shape)
    payload_bits = weights * cfg.bits
    scale_bits = groups * 16.0
    zero_point_bits = 0.0 if cfg.scheme == "symmetric" else groups * 16.0
    total_bits = payload_bits + scale_bits + zero_point_bits
    return {
        "weights": float(weights),
        "groups": float(groups),
        "payload_bits": float(payload_bits),
        "scale_bits": float(scale_bits),
        "zero_point_bits": float(zero_point_bits),
        "total_bits": float(total_bits),
        "bits_per_weight": float(total_bits / weights),
    }


def effective_bits_per_weight(cfg: QuantConfig, weight_shape: tuple[int, int]) -> float:
    """Bits per weight including scale (and zero-point) metadata, exactly."""
    return storage_bits(cfg, weight_shape)["bits_per_weight"]


@dataclass
class QuantResult:
    """Output of :func:`quantize_weight`. All tensors are detached float32.

    ``codes`` are the clamped integer codes; ``codes_raw`` are the pre-clamp
    rounded codes and ``clipped`` marks positions that were actually clipped.
    The last two are populated only when ``with_codes=True`` and exist so that
    endpoint occupancy is not confused with clipping (with max-abs scaling the
    group extreme always sits on an endpoint, which is not a clipping event).
    """

    qdq: Tensor
    scale: Tensor
    zero_point: Tensor
    codes: Tensor | None = None
    codes_raw: Tensor | None = None
    clipped: Tensor | None = None
    degenerate_groups: int = 0


def _quantize_blocks(blocks: Tensor, cfg: QuantConfig, with_codes: bool) -> QuantResult:
    """Quantize ``[rows, n_groups, group_len]`` blocks along the last dim.

    When ``with_codes`` is False, pre-clamp codes and the clipping mask are not
    materialized: the training forward only needs the dequantized weight.
    """
    qmin, qmax = cfg.code_min, cfg.code_max

    if cfg.scheme == "symmetric":
        qmax_sym = cfg.qmax_symmetric
        amax = blocks.abs().amax(dim=-1, keepdim=True)
        scale = (amax / qmax_sym).clamp_min(cfg.scale_floor)
        zero = torch.zeros_like(scale)
        raw = torch.round(blocks / scale)
        clipped = raw.abs() > qmax_sym if with_codes else None
        codes = raw.clamp(qmin, qmax)
        return QuantResult(
            qdq=codes * scale,
            scale=scale,
            zero_point=zero,
            codes=codes if with_codes else None,
            codes_raw=raw if with_codes else None,
            clipped=clipped,
        )

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

    raw = (blocks / scale).round().add_(zero)
    clipped = ((raw < qmin) | (raw > qmax)) if with_codes else None
    codes = raw.clamp(qmin, qmax)
    qdq = (codes - zero) * scale
    return QuantResult(
        qdq=qdq,
        scale=scale,
        zero_point=zero,
        codes=codes if with_codes else None,
        codes_raw=raw if with_codes else None,
        clipped=clipped,
        degenerate_groups=n_degenerate,
    )


def _flatten(result: QuantResult, out_features: int, in_features: int) -> QuantResult:
    def _2d(t: Tensor | None) -> Tensor | None:
        return None if t is None else t.reshape(out_features, in_features)

    return QuantResult(
        qdq=_2d(result.qdq),  # type: ignore[arg-type]
        scale=result.scale,
        zero_point=result.zero_point,
        codes=_2d(result.codes),
        codes_raw=_2d(result.codes_raw),
        clipped=_2d(result.clipped),
        degenerate_groups=result.degenerate_groups,
    )


def quantize_weight(
    weight: Tensor, cfg: QuantConfig, *, with_codes: bool = False
) -> QuantResult:
    """Quantize-dequantize a ``[out_features, in_features]`` weight matrix.

    Returns detached tensors; for a differentiable wrapper use
    :func:`fake_quantize_weight`. ``with_codes=True`` also returns integer
    codes, pre-clamp codes and the clipping mask (for diagnostics only).
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    w = weight.detach().to(torch.float32)
    out_features, in_features = w.shape

    if cfg.granularity == "per_tensor":
        blocks = w.reshape(1, 1, out_features * in_features)
    elif cfg.granularity == "per_row":
        blocks = w.reshape(out_features, 1, in_features)
    else:
        gs = cfg.group_size
        if gs >= in_features:
            blocks = w.reshape(out_features, 1, in_features)
        elif in_features % gs == 0:
            blocks = w.reshape(out_features, in_features // gs, gs)
        else:
            # Tail group: quantize the full groups and the remainder separately
            # so that no padding value can leak into a group's max-abs.
            full = in_features // gs * gs
            head = w[:, :full].reshape(out_features, full // gs, gs)
            tail = w[:, full:].reshape(out_features, 1, in_features - full)
            r_head = _quantize_blocks(head, cfg, with_codes)
            r_tail = _quantize_blocks(tail, cfg, with_codes)
            return QuantResult(
                qdq=torch.cat(
                    [
                        r_head.qdq.reshape(out_features, full),
                        r_tail.qdq.reshape(out_features, in_features - full),
                    ],
                    dim=1,
                ),
                scale=torch.cat([r_head.scale, r_tail.scale], dim=1),
                zero_point=torch.cat([r_head.zero_point, r_tail.zero_point], dim=1),
                codes=_cat2d(r_head.codes, r_tail.codes, out_features, full, in_features, with_codes),
                codes_raw=_cat2d(
                    r_head.codes_raw, r_tail.codes_raw, out_features, full, in_features, with_codes
                ),
                clipped=_cat2d(
                    r_head.clipped, r_tail.clipped, out_features, full, in_features, with_codes
                ),
                degenerate_groups=r_head.degenerate_groups + r_tail.degenerate_groups,
            )

    result = _quantize_blocks(blocks, cfg, with_codes)
    return _flatten(
        QuantResult(
            qdq=result.qdq,
            scale=result.scale,
            zero_point=result.zero_point,
            codes=result.codes if with_codes else None,
            codes_raw=result.codes_raw if with_codes else None,
            clipped=result.clipped if with_codes else None,
            degenerate_groups=result.degenerate_groups,
        ),
        out_features,
        in_features,
    )


def _cat2d(
    head: Tensor | None,
    tail: Tensor | None,
    out_features: int,
    full: int,
    in_features: int,
    with_codes: bool,
) -> Tensor | None:
    if not with_codes or head is None or tail is None:
        return None
    return torch.cat(
        [
            head.reshape(out_features, full),
            tail.reshape(out_features, in_features - full),
        ],
        dim=1,
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
    endpoint_fraction: float
    clipped_fraction: float
    zero_level_fraction: float
    mean_scale: float
    min_scale: float
    max_scale: float
    degenerate_groups: int
    bits_per_weight: float


def quant_diagnostics(weight: Tensor, cfg: QuantConfig) -> QuantDiagnostics:
    """Numerical diagnostics for one weight matrix (for reports, not training).

    ``endpoint_fraction`` is the share of codes sitting on the lowest/highest
    representable code. With max-abs scaling this is structural (each group's
    extreme maps to an endpoint) and is **not** clipping. ``clipped_fraction``
    is the share of positions whose pre-clamp rounded code fell outside the
    representable range. ``zero_level_fraction`` is the share of codes equal to
    the zero level (code 0 symmetric, zero-point asymmetric).
    """
    w = weight.detach().to(torch.float32)
    result = quantize_weight(w, cfg, with_codes=True)
    assert result.codes is not None and result.codes_raw is not None and result.clipped is not None
    codes = result.codes
    error = result.qdq - w
    endpoint = (codes == cfg.code_min) | (codes == cfg.code_max)
    if cfg.scheme == "symmetric":
        zero_level = codes == 0
    else:
        zero_level = codes == result.zero_point
    return QuantDiagnostics(
        mean_squared_error=float(error.pow(2).mean().item()),
        max_abs_error=float(error.abs().max().item()),
        endpoint_fraction=float(endpoint.float().mean().item()),
        clipped_fraction=float(result.clipped.float().mean().item()),
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

    Device placement is preserved by :meth:`from_linear`; no-grad forwards use
    a dequantized-weight cache keyed by the parameter's version counter, so an
    in-place optimizer update invalidates it automatically. Trainers should
    also call :meth:`invalidate_cache` after ``optimizer.step()`` because
    ``param.data`` assignments may not bump the version counter.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: QuantConfig | None = None,
        name: str = "",
        device: torch.device | str | None = None,
        bias_dtype: torch.dtype = torch.float32,
        init: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantConfig()
        self.name = name
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32, device=device)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=bias_dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)
        if init:
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self._dequant_cache: tuple[int, Tensor] | None = None

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
            device=linear.weight.device,
            bias_dtype=linear.bias.dtype if linear.bias is not None else torch.float32,
            init=False,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight.detach().to(torch.float32))
            if linear.bias is not None:
                module.bias.copy_(linear.bias.detach())
        return module

    @property
    def quantized(self) -> bool:
        return bool(getattr(self, "_quantized", True))

    @quantized.setter
    def quantized(self, value: bool) -> None:
        self._quantized = bool(value)

    def _apply(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        self._dequant_cache = None
        return super()._apply(fn, *args, **kwargs)

    def invalidate_cache(self) -> None:
        """Drop the cached dequantized weight (call after an optimizer step)."""
        self._dequant_cache = None

    @torch.no_grad()
    def dequantized_weight(self) -> Tensor:
        """Return the fake-quantized weight, cached until the master changes."""
        version = self.weight._version
        cache = self._dequant_cache
        if (
            cache is not None
            and cache[0] == version
            and cache[1].device == self.weight.device
        ):
            return cache[1]
        weight = quantize_weight(self.weight, self.config).qdq
        self._dequant_cache = (version, weight)
        return weight

    def forward(self, x: Tensor) -> Tensor:
        if not self.quantized:
            weight = self.weight
        elif torch.is_grad_enabled():
            weight = fake_quantize_weight(self.weight, self.config)
        else:
            weight = self.dequantized_weight()
        weight = weight.to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.config.bits}, "
            f"scheme={self.config.scheme}, group_size={self.config.group_size}, "
            f"name={self.name!r}"
        )
