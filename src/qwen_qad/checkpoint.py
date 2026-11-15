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
