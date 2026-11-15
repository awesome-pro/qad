from __future__ import annotations

import pytest

from qwen_qad.benchmark_compare import paired_compare


def records(correct: list[bool]) -> list[dict]:
    return [{"example_id": f"q{i}", "correct": value} for i, value in enumerate(correct)]


def test_paired_compare_identical():
    result = paired_compare(records([True, False, True]), records([True, False, True]))
    assert result["difference_q2_minus_q1"] == 0.0
    assert result["paired_counts"]["both"] == 2


def test_paired_compare_requires_alignment():
    with pytest.raises(ValueError):
        paired_compare(records([True]), records([True, False]))
    with pytest.raises(ValueError):
        paired_compare(
            [{"example_id": "a", "correct": True}],
            [{"example_id": "b", "correct": True}],
        )
