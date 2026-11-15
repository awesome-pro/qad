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
