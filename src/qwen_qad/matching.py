"""Automated checks that the matched Q1/Q2 runs really are matched."""

from __future__ import annotations

from typing import Any


def compare_matched_runs(q1: dict[str, Any], q2: dict[str, Any]) -> dict[str, Any]:
    """Compare two run manifests; return checks, mismatches and token ratios."""
    mismatches: list[str] = []
    checks: dict[str, bool] = {}

    if q1.get("arm") != "q1" or q2.get("arm") != "q2":
        mismatches.append("arms must be q1 and q2")

    def same(path: str, getter=lambda run: run) -> Any:
        try:
            a = getter(q1)
            b = getter(q2)
        except Exception:
            return None
        checks[path] = a == b
        if a != b:
            mismatches.append(f"{path}: {a!r} != {b!r}")
        return a

    same("model.id", lambda r: r["model"]["id"])
    same("model.revisions.resolved_model_sha", lambda r: r["model"]["revisions"]["resolved_model_sha"])
    same("quantization", lambda r: r["quantization"])
    same("schedule.seed", lambda r: r["schedule"]["seed"])
    same("schedule.prompts_per_step", lambda r: r["schedule"]["prompts_per_step"])
    same("schedule.continuation_cap", lambda r: r["schedule"]["continuation_cap"])
    same("optimizer.lr", lambda r: r["optimizer"]["lr"])
    same("optimizer.warmup_steps", lambda r: r["optimizer"]["warmup_steps"])
    same("optimizer.steps", lambda r: r["optimizer"]["steps"])
    same("data.val_sha256", lambda r: r["data"]["val_sha256"])

    q1_steps = {s["step"]: s for s in q1["steps_log"]}
    q2_steps = {s["step"]: s for s in q2["steps_log"]}
    checks["same_step_count"] = set(q1_steps) == set(q2_steps)
    if not checks["same_step_count"]:
        mismatches.append("different optimizer step sets")

    hash_mismatch_steps = []
    for step in sorted(set(q1_steps) & set(q2_steps)):
        if q1_steps[step]["prompt_hashes"] != q2_steps[step]["prompt_hashes"]:
            hash_mismatch_steps.append(step)
    checks["prompt_hashes_match"] = not hash_mismatch_steps
    if hash_mismatch_steps:
        mismatches.append(f"prompt hash mismatch on {len(hash_mismatch_steps)} steps")

    q1_tokens = sum(s["response_tokens"] for s in q1["steps_log"])
    q2_tokens = sum(s["response_tokens"] for s in q2["steps_log"])
    token_ratio = (q1_tokens / q2_tokens) if q2_tokens else None
    checks["token_budget_within_5pct"] = bool(token_ratio and abs(token_ratio - 1.0) <= 0.05)

    checks["prompt_instances_match"] = (
        q1["schedule"]["prompt_instances"] == q2["schedule"]["prompt_instances"]
    )
    if not checks["prompt_instances_match"]:
        mismatches.append("prompt instance counts differ")

    return {
        "matched": not mismatches,
        "mismatches": mismatches,
        "checks": checks,
        "prompt_hash_mismatch_steps": hash_mismatch_steps,
        "q1_response_tokens": q1_tokens,
        "q2_response_tokens": q2_tokens,
        "token_ratio": token_ratio,
        "q1_unique_prompts": q1["schedule"]["unique_prompts_seen"],
        "q2_unique_prompts": q2["schedule"]["unique_prompts_seen"],
    }
