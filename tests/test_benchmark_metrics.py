from __future__ import annotations

import pytest

from qwen_qad.benchmark_metrics import summarize_records


def rec(status: str, correct: bool = False, complete: bool = True, closed: bool = True,
        seen: bool = True, terminal: bool = True) -> dict:
    return {
        "parse_status": status,
        "correct": correct,
        "finished_naturally": complete,
        "hit_cap": not complete,
        "thinking_closed": closed,
        "marker_seen": seen,
        "terminal_marker_present": terminal,
        "response_ids": [1, 2, 3],
    }


def test_marker_presence_parse_and_accuracy_are_distinct():
    records = [
        rec("ok", True),
        rec("ambiguous"),
        rec("malformed"),
        rec("missing", seen=False, terminal=False),
    ]
    m = summarize_records(records, thinking=False)
    assert m["accuracy"] == 0.25
    assert m["marker_present_rate"] == 0.75
    assert m["terminal_marker_present_rate"] == 0.75
    assert m["parsed_answer_rate"] == 0.25
    assert m["completion_rate"] == 1.0


def test_completion_and_truncation_are_reported_separately():
    records = [rec("missing", complete=False), rec("ok", True, complete=True)]
    m = summarize_records(records, thinking=True)
    assert m["completion_rate"] == 0.5
    assert m["truncation_rate"] == 0.5
    assert m["thinking_closed_rate"] == 1.0


def test_marker_seen_without_terminal_marker():
    records = [rec("missing", seen=True, terminal=False), rec("ok", True)]
    m = summarize_records(records, thinking=False)
    assert m["marker_present_rate"] == 1.0
    assert m["terminal_marker_present_rate"] == 0.5
    assert m["parsed_answer_rate"] == 0.5


def test_empty_records_rejected():
    with pytest.raises(ValueError):
        summarize_records([], thinking=False)
