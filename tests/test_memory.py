"""Tests for memory measurement helpers."""

from __future__ import annotations

import torch
from torch import nn

from qwen_qad.memory import MemorySampler, parameter_bytes


def test_parameter_bytes():
    module = nn.Linear(10, 5)
    expected = (10 * 5 + 5) * 4  # fp32 weight and bias
    assert parameter_bytes(module) == expected


def test_memory_sampler_cpu_reports_rss():
    with MemorySampler(torch.device("cpu")) as sampler:
        pass
    result = sampler.result()
    assert result["peak_bytes"] is not None and result["peak_bytes"] > 0
    assert result["peak_method"] == "resource.ru_maxrss"
    assert "not comparable" in result["note"]
