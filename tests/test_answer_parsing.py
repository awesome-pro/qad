"""Tests for strict final-answer parsing and Qwen3 thinking markers."""

from __future__ import annotations

import pytest

from qwen_qad.answer_parsing import (
    THINK_CLOSE,
    THINK_OPEN,
    normalize_number,
    parse_final_answer,
    split_thinking,
)


def test_qwen3_thinking_markers_are_ascii_added_tokens():
    # Verified against the pinned Qwen3 tokenizer: ids 151667 / 151668.
    assert THINK_OPEN == "<think>"
    assert THINK_CLOSE == "</think>"


def test_non_thinking_explicit_answer():
    parsed = parse_final_answer("18 + 4 = 22. #### 22", thinking_mode=False)
    assert parsed.status == "ok"
    assert parsed.value == "22"


def test_non_thinking_last_number_is_not_an_answer():
    parsed = parse_final_answer("The intermediate result is 40, then 12 more...", thinking_mode=False)
    assert parsed.status == "missing"
    assert parsed.value is None


def test_thinking_intermediate_numbers_are_ignored():
    parsed = parse_final_answer("3*4 = 12 and 12+6 = 18. Wait, 18 seems wrong.", thinking_mode=True)
    assert parsed.status == "incomplete_thinking"
    assert parsed.value is None


def test_thinking_closed_with_answer():
    parsed = parse_final_answer("reasoning 12 -> 18.</think>\n\n#### 42", thinking_mode=True)
    assert parsed.status == "ok"
    assert parsed.value == "42"
    assert parsed.thinking_closed


def test_marker_inside_thinking_is_ignored():
    parsed = parse_final_answer("I could write #### 7 but that is a guess.</think>No final answer.", thinking_mode=True)
    assert parsed.status == "missing"


def test_thinking_closed_without_marker():
    parsed = parse_final_answer("reasoning... 18</think>\nI think the answer is 18.", thinking_mode=True)
    assert parsed.status == "missing"


def test_ambiguous_conflicting_markers():
    parsed = parse_final_answer("</think>\n#### 12\n#### 13", thinking_mode=True)
    assert parsed.status == "ambiguous"
    assert parsed.value is None


def test_malformed_marker():
    assert parse_final_answer("#### not-a-number", thinking_mode=False).status == "malformed"


def test_negative_decimal_commas_dollar_and_fraction():
    assert normalize_number("-3.5") == "-3.5"
    assert normalize_number("1,234") == "1234"
    assert normalize_number("$540") == "540"
    assert normalize_number("3/4") == "0.75"
    assert normalize_number("1/0") is None


def test_boxed_and_answer_line():
    assert parse_final_answer("work \\boxed{7}", thinking_mode=False).value == "7"
    assert parse_final_answer("work\nAnswer: -12", thinking_mode=False).value == "-12"


def test_answer_marker_must_be_on_its_own_line():
    # An inline "answer: 5" inside reasoning is not an Answer: line.
    parsed = parse_final_answer("the answer: 5 is a guess", thinking_mode=False)
    assert parsed.status == "missing"


def test_split_thinking_uses_last_close():
    reasoning, region, closed = split_thinking("a<think>b</think>c</think>#### 5")
    assert closed
    assert reasoning.endswith("c")
    assert region.strip() == "#### 5"


def test_missing_answer_region_after_close():
    parsed = parse_final_answer("reasoning</think>  ", thinking_mode=True)
    assert parsed.status == "missing"
    assert parsed.thinking_closed
