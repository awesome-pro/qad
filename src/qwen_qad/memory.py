"""Device memory measurement helpers.

CUDA exposes a true peak (``max_memory_allocated``). MPS only exposes current
allocated memory, so the peak is sampled from a background thread and is a
lower bound. Process RSS is also recorded for CPU runs. None of these numbers
are comparable across processes that hold different numbers of models; the
diagnostics report parameter bytes separately.
"""

from __future__ import annotations

import resource
import threading
import time
from typing import Any

import torch


def current_allocated_bytes(device: torch.device | str) -> int | None:
    device = torch.device(device)
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    return None


def driver_allocated_bytes(device: torch.device | str) -> int | None:
    device = torch.device(device)
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.memory_reserved(device))
    return None


def parameter_bytes(model: torch.nn.Module) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


class MemorySampler:
    """Context manager that records a process-level memory peak.

    ``peak_bytes`` is exact on CUDA and a sampled lower bound on MPS.
    """

    def __init__(self, device: torch.device | str, interval_s: float = 0.02) -> None:
        self.device = torch.device(device)
        self.interval_s = interval_s
        self.start_allocated: int | None = None
        self.end_allocated: int | None = None
        self.peak_bytes: int | None = None
        self.method: str = "unsupported"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = 0

    def _poll(self) -> None:
        while not self._stop.is_set():
            value = current_allocated_bytes(self.device)
            if value is not None and value > self._peak:
                self._peak = value
            time.sleep(self.interval_s)

    def __enter__(self) -> "MemorySampler":
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.method = "torch.cuda.max_memory_allocated"
        elif self.device.type == "mps":
            self.method = f"sampled torch.mps.current_allocated_memory every {self.interval_s * 1000:.0f} ms"
        else:
            self.method = "resource.ru_maxrss"
        self.start_allocated = current_allocated_bytes(self.device)
        self._peak = self.start_allocated or 0
        if self.device.type == "mps":
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.end_allocated = current_allocated_bytes(self.device)
        if self.device.type == "cuda":
            self.peak_bytes = int(torch.cuda.max_memory_allocated(self.device))
        elif self.device.type == "mps":
            self.peak_bytes = int(self._peak)
        else:
            self.peak_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    def result(self) -> dict[str, Any]:
        return {
            "device_type": self.device.type,
            "start_allocated_bytes": self.start_allocated,
            "end_allocated_bytes": self.end_allocated,
            "peak_bytes": self.peak_bytes,
            "peak_method": self.method,
            "driver_allocated_bytes_at_end": driver_allocated_bytes(self.device),
            "note": (
                "Process-level number. CUDA is a true peak; MPS is a sampled lower bound. "
                "Processes holding teacher+student are not comparable to single-model runs; "
                "compare parameter_bytes instead."
            ),
        }
