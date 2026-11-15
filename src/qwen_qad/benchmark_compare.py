"""Paired per-question comparison for two evaluation records."""

from __future__ import annotations

from typing import Sequence

from .stats import bootstrap_ci, mcnemar_exact, paired_difference_ci


def paired_compare(q1_records: Sequence[dict], q2_records: Sequence[dict]) -> dict:
    """Compare correctness on the same questions; records must align by id."""
    if len(q1_records) != len(q2_records) or not q1_records:
        raise ValueError("record sets must be non-empty and the same length")
    ids1 = [r["example_id"] for r in q1_records]
    ids2 = [r["example_id"] for r in q2_records]
    if ids1 != ids2:
        raise ValueError("record sets are not aligned by example_id")

    q1 = [bool(r["correct"]) for r in q1_records]
    q2 = [bool(r["correct"]) for r in q2_records]
    point1, low1, high1 = bootstrap_ci(q1, seed=1)
    point2, low2, high2 = bootstrap_ci(q2, seed=2)
    diff, diff_low, diff_high = paired_difference_ci(q1, q2, seed=3)
    q1_only, q2_only, p = mcnemar_exact(q1, q2)
    return {
        "count": len(q1),
        "q1_accuracy": point1,
        "q1_ci95": [low1, high1],
        "q2_accuracy": point2,
        "q2_ci95": [low2, high2],
        "difference_q2_minus_q1": diff,
        "difference_ci95": [diff_low, diff_high],
        "paired_counts": {"q1_only": q1_only, "q2_only": q2_only, "both": sum(a and b for a, b in zip(q1, q2)), "neither": sum(not a and not b for a, b in zip(q1, q2))},
        "mcnemar_p": p,
    }
