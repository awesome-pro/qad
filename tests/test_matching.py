"""Tests for the matched-run comparison."""

from __future__ import annotations

import pytest

from qwen_qad.matching import compare_matched_runs


def run(arm: str, hashes: list[list[str]], tokens: int = 4096, unique: int = 300) -> dict:
    steps = [
        {"step": i + 1, "prompt_hashes": h, "response_tokens": tokens}
        for i, h in enumerate(hashes)
    ]
    return {
        "arm": arm,
        "model": {"id": "Qwen/Qwen3-0.6B", "revisions": {"resolved_model_sha": "abc"}},
        "quantization": {"bits": 4, "group_size": 128},
        "schedule": {
            "seed": 42,
            "prompts_per_step": 2,
            "continuation_cap": 128,
            "prompt_instances": 4,
            "unique_prompts_seen": unique,
        },
        "optimizer": {"lr": 1e-5, "warmup_steps": 5, "steps": 2},
        "data": {"val_sha256": "val"},
        "steps_log": steps,
    }


def test_matched_runs_pass():
    hashes = [["a", "b"], ["c", "d"]]
    result = compare_matched_runs(run("q1", hashes), run("q2", hashes))
    assert result["matched"]
    assert result["checks"]["prompt_hashes_match"]
    assert result["token_ratio"] == pytest.approx(1.0)


def test_prompt_hash_mismatch_is_reported():
    result = compare_matched_runs(run("q1", [["a", "b"]]), run("q2", [["a", "x"]]))
    assert not result["matched"]
    assert result["prompt_hash_mismatch_steps"] == [1]


def test_token_budget_mismatch_is_reported():
    result = compare_matched_runs(
        run("q1", [["a"]], tokens=4096), run("q2", [["a"]], tokens=3500)
    )
    assert not result["checks"]["token_budget_within_5pct"]
    assert not result["matched"]
    assert result["token_ratio"] > 1.05
    assert any("token budget" in item for item in result["mismatches"])


def test_missing_required_field_fails_matched():
    q1 = run("q1", [["a"]])
    del q1["schedule"]["seed"]
    result = compare_matched_runs(q1, run("q2", [["a"]]))
    assert not result["matched"]
    assert not result["checks"]["schedule.seed"]
    assert any("schedule.seed" in item for item in result["mismatches"])


def test_missing_steps_log_fails_matched():
    q1 = run("q1", [["a"]])
    q1["steps_log"] = []
    result = compare_matched_runs(q1, run("q2", [["a"]]))
    assert not result["matched"]
    assert not result["checks"]["same_step_count"]


def test_missing_step_field_fails_matched():
    q1 = run("q1", [["a"]])
    del q1["steps_log"][0]["response_tokens"]
    result = compare_matched_runs(q1, run("q2", [["a"]]))
    assert not result["matched"]


def test_wrong_arm_is_reported():
    result = compare_matched_runs(run("q1", [["a"]]), run("q1", [["a"]]))
    assert not result["matched"]
    assert any("arms" in item for item in result["mismatches"])
