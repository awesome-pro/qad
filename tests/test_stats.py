"""Tests for benchmark statistics."""

from __future__ import annotations

import pytest

from qwen_qad.stats import accuracy, bootstrap_ci, mcnemar_exact, paired_difference_ci


def test_accuracy_and_bootstrap_bounds():
    point, low, high = bootstrap_ci([True] * 20, n_boot=200, seed=1)
    assert point == 1.0
    assert low == 1.0 and high == 1.0
    point, low, high = bootstrap_ci([True, False] * 10, n_boot=200, seed=1)
    assert 0.2 <= low <= point <= high <= 0.8


def test_paired_difference_zero_for_identical():
    q = [True, False, True, True]
    point, low, high = paired_difference_ci(q, q, n_boot=200, seed=1)
    assert point == 0.0
    assert low <= 0.0 <= high


def test_mcnemar_counts_and_p():
    q1 = [True, True, False, False, True]
    q2 = [True, False, True, False, True]
    a, b, p = mcnemar_exact(q1, q2)
    assert (a, b) == (1, 1)
    assert 0.0 <= p <= 1.0
    a, b, p = mcnemar_exact([True] * 10, [False] * 10)
    assert (a, b) == (10, 0)
    assert p < 0.01


def test_empty_sequences_rejected():
    with pytest.raises(ValueError):
        accuracy([])
    with pytest.raises(ValueError):
        paired_difference_ci([], [])
