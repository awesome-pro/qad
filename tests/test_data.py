"""Tests for data preprocessing helpers."""

from __future__ import annotations

import pytest

from qwen_qad.data import truncate_response


def test_complete_response_gets_eos():
    ids, truncated = truncate_response([1, 2, 3], max_tokens=8, eos_token_id=99)
    assert truncated is False
    assert ids == [1, 2, 3, 99]


def test_truncated_response_does_not_get_eos():
    ids, truncated = truncate_response([1, 2, 3, 4], max_tokens=2, eos_token_id=99)
    assert truncated is True
    assert ids == [1, 2]
    assert ids[-1] != 99


def test_exact_length_is_complete_and_gets_eos():
    ids, truncated = truncate_response([1, 2], max_tokens=2, eos_token_id=99)
    assert truncated is False
    assert ids == [1, 2, 99]


def test_rejects_invalid_max_tokens():
    with pytest.raises(ValueError):
        truncate_response([1], max_tokens=0, eos_token_id=99)
