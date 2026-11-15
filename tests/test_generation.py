"""Tests for generation-quality metrics."""

from __future__ import annotations

import pytest

from qwen_qad.generation import (
    distinct_ratio,
    generation_summary,
    max_repeated_ngram_count,
    ngram_repetition_ratio,
)


def test_ngram_repetition_ratio_no_repeats():
    ids = [1, 2, 3, 4, 5, 6]
    assert ngram_repetition_ratio(ids, n=4) == 0.0


def test_ngram_repetition_ratio_detects_loop():
    ids = [1, 2, 3, 4] * 3
    ratio = ngram_repetition_ratio(ids, n=4)
    assert ratio > 0.5
    assert max_repeated_ngram_count(ids, n=4) == 3


def test_short_sequence_has_zero_repetition():
    assert ngram_repetition_ratio([1, 2], n=4) == 0.0
    assert max_repeated_ngram_count([1, 2], n=4) == 0


def test_generation_summary_complete_sequence():
    prompt = [10, 11]
    generated = list(range(1, 40)) + [99]  # 39 content tokens then EOS
    summary = generation_summary(prompt + generated, prompt_len=2, eos_token_id=99)
    assert summary["eos_generated"] is True
    assert summary["eos_position"] == 39
    assert summary["premature_eos"] is False
    assert summary["hit_max_tokens"] is False
    assert summary["content_tokens"] == 39


def test_generation_summary_premature_eos():
    summary = generation_summary([10, 1, 2, 99], prompt_len=1, eos_token_id=99)
    assert summary["premature_eos"] is True
    assert summary["content_tokens"] == 2


def test_generation_summary_no_eos_hits_cap():
    summary = generation_summary([10, 1, 2, 3], prompt_len=1, eos_token_id=99)
    assert summary["hit_max_tokens"] is True
    assert summary["eos_generated"] is False
    assert summary["premature_eos"] is False


def test_generation_summary_empty_after_eos():
    summary = generation_summary([10, 99], prompt_len=1, eos_token_id=99)
    assert summary["empty"] is True
    assert summary["eos_position"] == 0


def test_distinct_ratio_bounds():
    assert distinct_ratio([1, 2, 3], n=1) == 1.0
    assert distinct_ratio([1, 1, 1], n=1) == pytest.approx(1 / 3)
