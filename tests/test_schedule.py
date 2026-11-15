"""Tests for the shared matched prompt schedule."""

from __future__ import annotations

import json

from qwen_qad.schedule import build_schedule, prompt_hash, step_indices, step_prompt_hashes


def records(n: int = 40, prompt_len: int = 10, response_len: int = 20) -> list[dict]:
    return [
        {"prompt_ids": list(range(100 + i, 100 + i + prompt_len)),
         "response_ids": list(range(i, i + response_len))}
        for i in range(n)
    ]


def test_schedule_is_deterministic_and_identical_for_both_arms():
    recs = records()
    a = build_schedule(recs, max_seq_len=256, continuation_cap=32, seed=42)
    b = build_schedule(recs, max_seq_len=256, continuation_cap=32, seed=42)
    c = build_schedule(recs, max_seq_len=256, continuation_cap=32, seed=43)
    assert a == b
    assert a != c


def test_schedule_filters_too_long_prompts():
    recs = records(5, prompt_len=250, response_len=10)
    try:
        build_schedule(recs, max_seq_len=128, continuation_cap=32, seed=42)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected no eligible prompts")


def test_step_indices_are_identical_across_arms_and_wrap():
    recs = records(10)
    schedule = build_schedule(recs, max_seq_len=256, continuation_cap=32, seed=7)
    q1 = [step_indices(schedule, step, 3) for step in (1, 2, 3, 4)]
    q2 = [step_indices(schedule, step, 3) for step in (1, 2, 3, 4)]
    assert q1 == q2
    assert q1[0] == schedule[:3]
    assert q1[1] == schedule[3:6]
    assert len(set(q1[0])) == 3  # no duplicate within a step


def test_step_indices_reject_too_many_prompts():
    recs = records(4)
    schedule = build_schedule(recs, max_seq_len=256, continuation_cap=32, seed=1)
    try:
        step_indices(schedule, 1, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_prompt_hashes_match_for_same_indices():
    recs = records(6)
    schedule = build_schedule(recs, max_seq_len=256, continuation_cap=32, seed=1)
    indices = step_indices(schedule, 1, 2)
    h1 = step_prompt_hashes(recs, indices)
    h2 = step_prompt_hashes(recs, indices)
    assert h1 == h2
    assert len(h1) == 2
    assert prompt_hash([1, 2, 3]) != prompt_hash([1, 2, 4])
