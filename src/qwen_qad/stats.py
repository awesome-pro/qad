"""Small statistical helpers for benchmark comparisons."""

from __future__ import annotations

import math
import random
from typing import Sequence


def accuracy(successes: Sequence[bool]) -> float:
    if not successes:
        raise ValueError("empty results")
    return sum(bool(x) for x in successes) / len(successes)


def bootstrap_ci(
    successes: Sequence[bool], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for accuracy; returns (point, low, high)."""
    values = [1 if x else 0 for x in successes]
    if not values:
        raise ValueError("empty results")
    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    low = estimates[int((alpha / 2) * n_boot)]
    high = estimates[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return sum(values) / n, low, high


def paired_difference_ci(
    q1: Sequence[bool], q2: Sequence[bool], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap CI for (Q2 accuracy - Q1 accuracy) on paired questions."""
    if len(q1) != len(q2) or not q1:
        raise ValueError("paired sequences must be non-empty and equal length")
    rng = random.Random(seed)
    n = len(q1)
    diffs = []
    for _ in range(n_boot):
        indices = [rng.randrange(n) for _ in range(n)]
        a = sum(bool(q1[i]) for i in indices) / n
        b = sum(bool(q2[i]) for i in indices) / n
        diffs.append(b - a)
    diffs.sort()
    point = accuracy(q2) - accuracy(q1)
    low = diffs[int((alpha / 2) * n_boot)]
    high = diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return point, low, high


def mcnemar_exact(q1: Sequence[bool], q2: Sequence[bool]) -> tuple[int, int, float]:
    """Exact McNemar test for paired correctness; returns (q1_only, q2_only, p)."""
    if len(q1) != len(q2):
        raise ValueError("paired sequences must have equal length")
    q1_only = sum(bool(a) and not bool(b) for a, b in zip(q1, q2))
    q2_only = sum(bool(b) and not bool(a) for a, b in zip(q1, q2))
    n = q1_only + q2_only
    if n == 0:
        return 0, 0, 1.0
    k = min(q1_only, q2_only)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return q1_only, q2_only, min(1.0, 2 * p)
