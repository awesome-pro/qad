"""Tests for strict final-answer parsing and Qwen3 thinking markers."""

from __future__ import annotations

import pytest

from qwen_qad.answer_parsing import (
    PARSER_VERSION,
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
    assert PARSER_VERSION == "3.0.0"


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


def test_conflicting_hash_markers_are_ambiguous():
    parsed = parse_final_answer("#### 12\n#### 13", thinking_mode=False)
    assert parsed.status == "ambiguous"
    assert parsed.value is None


def test_conflicting_valid_and_malformed_markers_are_rejected():
    parsed = parse_final_answer("#### 42\n#### not-a-number", thinking_mode=False)
    assert parsed.status == "ambiguous"
    assert parsed.value is None


def test_multiple_same_value_markers_are_ok():
    parsed = parse_final_answer("#### 42\nAnswer: 42", thinking_mode=False)
    assert parsed.status == "ok"
    assert parsed.value == "42"


def test_hash_does_not_consume_newline():
    # The old regex `####\s*` would consume the newline and accept 42.
    parsed = parse_final_answer("####\n42", thinking_mode=False)
    assert parsed.status != "ok"
    assert parsed.value is None


def test_intermediate_boxed_is_ignored_final_boxed_wins():
    text = "First compute \\boxed{12}.\nSo the final answer is \\boxed{42}."
    assert parse_final_answer(text, thinking_mode=False).value == "42"


def test_final_line_boxed_is_accepted():
    assert parse_final_answer("work\n\\boxed{7}", thinking_mode=False).value == "7"
    assert parse_final_answer("work\nAnswer: \\boxed{7}", thinking_mode=False).value == "7"


def test_inline_boxed_calculation_is_not_accepted():
    parsed = parse_final_answer("\\boxed{12} + 3 = 15", thinking_mode=False)
    assert parsed.status == "missing"
    assert parsed.value is None


def test_boxed_on_final_line_wins_over_intermediate_boxed():
    text = "intermediate \\boxed{12}\n\\boxed{42}"
    assert parse_final_answer(text, thinking_mode=False).value == "42"


def test_boxed_with_prose_on_final_line_is_accepted():
    assert parse_final_answer("final answer \\boxed{42}", thinking_mode=False).value == "42"


def test_malformed_marker():
    assert parse_final_answer("#### not-a-number", thinking_mode=False).status == "malformed"


def test_negative_decimal_commas_dollar_and_fraction():
    assert normalize_number("-3.5") == "-3.5"
    assert normalize_number("1,234") == "1234"
    assert normalize_number("$540") == "540"
    assert normalize_number("3/4") == "0.75"
    assert normalize_number("1/0") is None


def test_answer_line():
    assert parse_final_answer("work\nAnswer: -12", thinking_mode=False).value == "-12"


def test_answer_marker_must_be_on_its_own_line():
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


def test_marker_present_flag_separates_parse_from_presence():
    parsed = parse_final_answer("#### 12", thinking_mode=False)
    assert parsed.status == "ok"
    assert parsed.marker_present and parsed.explicit
    ambiguous = parse_final_answer("#### 12\n#### 13", thinking_mode=False)
    assert ambiguous.status == "ambiguous"
    assert ambiguous.marker_present and not ambiguous.explicit


def test_qwen3_tokenizer_markers_when_cached():
    """Verify marker token ids against the pinned tokenizer when it is local."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B",
            revision="c1899de289a04d12100db370d81485cdf75e47ca",
            local_files_only=True,
        )
    except Exception:
        pytest.skip("pinned tokenizer not in the local cache")
    assert tok.encode("<think>", add_special_tokens=False) == [151667]
    assert tok.encode("</think>", add_special_tokens=False) == [151668]


def test_boxed_inside_display_math_is_accepted():
    text = "work\n\n$$\n\\boxed{18}\n$$"
    assert parse_final_answer(text, thinking_mode=False).value == "18"
    text2 = "intermediate \\boxed{12}\n$$\n\\boxed{42}\n$$"
    assert parse_final_answer(text2, thinking_mode=False).value == "42"


def test_display_math_decorations_and_units():
    assert parse_final_answer("$$ \\boxed{125} $$", thinking_mode=False).value == "125"
    assert parse_final_answer("3 + 4 = \\boxed{7} \\text{ DVDs}", thinking_mode=False).value == "7"
    assert parse_final_answer("**Answer:** $ \\boxed{5} $", thinking_mode=False).value == "5"


def test_boxed_followed_by_arithmetic_is_not_final():
    assert parse_final_answer("\\boxed{12} + 3 = 15", thinking_mode=False).status == "missing"
    assert parse_final_answer("\\boxed{12} = 12", thinking_mode=False).status == "missing"


def test_multiple_distinct_boxes_are_ambiguous():
    assert parse_final_answer("\\boxed{12} and \\boxed{13}", thinking_mode=False).status == "ambiguous"


def test_nested_hash_boxed_is_one_answer():
    assert parse_final_answer(r"#### $\boxed{8}$", thinking_mode=False).value == "8"
    assert parse_final_answer(r"#### **\boxed{8}**", thinking_mode=False).value == "8"


def test_markdown_answer_lines():
    assert parse_final_answer("**Answer: $694**", thinking_mode=False).value == "694"
    assert parse_final_answer("**Answer: $233.00**", thinking_mode=False).value == "233"


def test_answer_regex_does_not_consume_next_line():
    parsed = parse_final_answer("Answer: 5\nnot the answer", thinking_mode=False)
    assert parsed.status == "missing"
    assert parsed.marker_seen
    assert not parsed.terminal_marker_present


def test_marker_present_vs_terminal_marker():
    earlier = parse_final_answer("I tried #### 8\nbut no final marker", thinking_mode=False)
    assert earlier.status == "missing"
    assert earlier.marker_seen
    assert not earlier.terminal_marker_present
    terminal = parse_final_answer("#### 8", thinking_mode=False)
    assert terminal.marker_seen and terminal.terminal_marker_present


def test_contradictory_terminal_answers_rejected():
    assert parse_final_answer("#### 8\nAnswer: 9", thinking_mode=False).status == "ambiguous"


def test_nested_boxed_with_arithmetic_is_rejected():
    assert parse_final_answer(r"#### \boxed{8} + 3", thinking_mode=False).status != "ok"


def test_nested_markers_after_answer_prefix():
    assert parse_final_answer("**Answer:** #### 225", thinking_mode=False).value == "225"
    assert parse_final_answer("**Answer:** $\\boxed{10}$ games.", thinking_mode=False).value == "10"
    assert parse_final_answer("Answer: #### 8\nAnswer: 9", thinking_mode=False).status == "ambiguous"
    assert parse_final_answer("Answer: $\\boxed{12}$ + 3 = 15", thinking_mode=False).status != "ok"


def test_answers_with_units_are_numeric_but_extra_numbers_are_not():
    assert normalize_number("14 bags.") == "14"
    assert normalize_number("120 minutes") == "120"
    assert normalize_number("8 and 9") is None
    assert normalize_number("12 + 3") is None
    text = "**Answer:** 14 bags.\\nFinal answer: #### 14."
    assert parse_final_answer(text, thinking_mode=False).value == "14"
