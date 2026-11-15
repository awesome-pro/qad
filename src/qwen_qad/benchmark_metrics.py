"""Shared benchmark metric definitions (strict marker terminology)."""

from __future__ import annotations

from typing import Sequence

from .stats import bootstrap_ci

MARKER_STATUSES = {"ok", "ambiguous", "malformed"}


def summarize_records(records: Sequence[dict], thinking: bool, seed: int = 1234) -> dict:
    """Summarize per-example records without conflating marker/parse/accuracy."""
    if not records:
        raise ValueError("empty records")
    n = len(records)
    correct = [bool(r["correct"]) for r in records]
    status_counts: dict[str, int] = {}
    for record in records:
        status = record["parse_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    completed = sum(bool(r.get("finished_naturally")) for r in records)
    truncated = sum(bool(r.get("hit_cap")) for r in records)
    closed = sum(bool(r.get("thinking_closed")) for r in records)
    marker_present = sum(r["parse_status"] in MARKER_STATUSES for r in records)
    parsed = sum(r["parse_status"] == "ok" for r in records)
    point, low, high = bootstrap_ci(correct, seed=seed)
    return {
        "accuracy": point,
        "accuracy_ci95": [low, high],
        "correct": int(sum(correct)),
        "count": n,
        "marker_present_rate": marker_present / n,
        "parsed_answer_rate": parsed / n,
        "completion_rate": completed / n,
        "truncation_rate": truncated / n,
        "thinking_closed_rate": closed / n,
        "parse_status_counts": status_counts,
        "mean_response_tokens": sum(len(r.get("response_ids", [])) for r in records) / n,
    }
