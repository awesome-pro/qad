"""Tests for cross-artifact evaluation integrity checks."""

from __future__ import annotations

from qwen_qad.benchmark_metrics import summarize_records
from qwen_qad.eval_integrity import check_provenance, check_records, check_summary


def records():
    return [
        {"example_id": "a", "parse_status": "ok", "parsed_value": "8", "gold_answer": "8",
         "correct": True, "finished_naturally": True, "hit_cap": False,
         "thinking_closed": True, "marker_seen": True, "terminal_marker_present": True,
         "response_ids": [1, 2]},
        {"example_id": "b", "parse_status": "missing", "parsed_value": None, "gold_answer": "9",
         "correct": False, "finished_naturally": False, "hit_cap": True,
         "thinking_closed": False, "marker_seen": False, "terminal_marker_present": False,
         "response_ids": [1]},
    ]


def summary(records_):
    return {"parser_version": "3.0.0", "source_sha256": "abc",
            "metrics": summarize_records(records_, thinking=False)}


def test_summary_and_record_consistency_pass():
    recs = records()
    assert check_records(recs) == []
    assert check_summary(summary(recs), recs, "nonthinking") == []


def test_record_correct_flag_is_checked():
    recs = records()
    recs[0]["correct"] = False
    assert any("correct flag" in problem for problem in check_records(recs))


def test_stale_provenance_is_rejected():
    recs = records()
    canonical = summary(recs)
    stale = {"metrics": {**canonical["metrics"], "correct": 0},
             "parser_version": "3.0.0", "source_sha256": "abc",
             "checkpoint": {"sha256": "ckpt"}}
    run = {"best_checkpoint": {"sha256": "ckpt"}}
    problems = check_provenance(stale, canonical, run)
    assert any("metrics" in problem for problem in problems)
