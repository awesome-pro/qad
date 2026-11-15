"""Checkpoint save/load for wrapped Quantized students.

The state dict uses the original module names, so a wrapped model saves and
reloads with the same keys. The payload also stores the optimizer, step,
quantization config and run metadata. Atomic write via a temporary file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    meta: dict[str, Any] | None = None,
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "meta": meta or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path.stat().st_size


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        if payload.get("optimizer") is None:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def export_inference_weights(
    path: str | Path,
    model: nn.Module,
    metadata: dict[str, str] | None = None,
) -> int:
    """Save portable master weights (no optimizer) as safetensors.

    Qwen3 ties ``lm_head.weight`` to the embedding; the duplicate key is omitted
    because safetensors rejects shared storage. The optimizer state is kept
    separately by :func:`save_checkpoint` for training resumption.
    """
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if not key.startswith("lm_head.")
    }
    save_file(state, str(path), metadata=metadata or {})
    return path.stat().st_size


def import_inference_weights(path: str | Path, model: nn.Module) -> list[str]:
    """Restore portable weights; returns keys intentionally missing (tied lm_head)."""
    from safetensors.torch import load_file

    state = load_file(str(path))
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise ValueError(f"unexpected weights: {unexpected}")
    return list(result.missing_keys)


def file_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_verified(
    path: str | Path,
    model: nn.Module,
    expected_model_sha: str,
    expected_bits: int | None = None,
    expected_group_size: int | None = None,
    expected_scheme: str = "symmetric",
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a checkpoint only if its metadata matches the requested model/quantizer."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload.get("meta") or {}
    revisions = meta.get("revisions") or {}
    model_sha = revisions.get("resolved_model_sha")
    if model_sha != expected_model_sha:
        raise ValueError(
            f"checkpoint model revision mismatch: {model_sha!r} != {expected_model_sha!r}"
        )
    quant = meta.get("quantization") or {}
    if expected_bits is not None and quant.get("bits") != expected_bits:
        raise ValueError(f"checkpoint bits mismatch: {quant.get('bits')!r} != {expected_bits}")
    if expected_group_size is not None and quant.get("group_size") != expected_group_size:
        raise ValueError(
            f"checkpoint group_size mismatch: {quant.get('group_size')!r} != {expected_group_size}"
        )
    if quant.get("scheme") != expected_scheme:
        raise ValueError(f"checkpoint scheme mismatch: {quant.get('scheme')!r} != {expected_scheme}")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        if payload.get("optimizer") is None:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    return meta
