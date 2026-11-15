"""Automated checks that the matched Q1/Q2 runs really are matched."""

from __future__ import annotations

from typing import Any

_REQUIRED_PATHS = (
    "model.id",
    "model.revisions.resolved_model_sha",
    "quantization",
    "schedule.seed",
    "schedule.prompts_per_step",
    "schedule.continuation_cap",
    "schedule.prompt_instances",
    "schedule.unique_prompts_seen",
    "optimizer.lr",
    "optimizer.warmup_steps",
    "optimizer.steps",
    "data.val_sha256",
)


def _get(run: dict, path: str) -> Any:
    value: Any = run
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def _steps(run: dict) -> dict[int, dict]:
    steps = run.get("steps_log")
    if not isinstance(steps, list) or not steps:
        raise KeyError("steps_log")
    mapped = {}
    for step in steps:
        for key in ("step", "prompt_hashes", "response_tokens"):
            if key not in step:
                raise KeyError(f"steps_log.{key}")
        mapped[int(step["step"])] = step
    return mapped


def compare_matched_runs(q1: dict[str, Any], q2: dict[str, Any]) -> dict[str, Any]:
    """Compare two run manifests; missing fields or token mismatch is a failure."""
    mismatches: list[str] = []
    checks: dict[str, bool] = {}

    checks["arms"] = q1.get("arm") == "q1" and q2.get("arm") == "q2"
    if not checks["arms"]:
        mismatches.append("arms must be q1 and q2")

    for path in _REQUIRED_PATHS:
        try:
            a = _get(q1, path)
            b = _get(q2, path)
        except KeyError:
            checks[path] = False
            mismatches.append(f"{path}: missing or invalid")
            continue
        checks[path] = a == b
        if a != b:
            mismatches.append(f"{path}: {a!r} != {b!r}")

    try:
        q1_steps = _steps(q1)
        q2_steps = _steps(q2)
    except KeyError as error:
        checks["same_step_count"] = False
        checks["prompt_hashes_match"] = False
        mismatches.append(f"steps_log missing required field: {error}")
        q1_steps, q2_steps = {}, {}

    checks["same_step_count"] = bool(q1_steps) and set(q1_steps) == set(q2_steps)
    if not checks["same_step_count"]:
        mismatches.append("different or empty optimizer step sets")

    hash_mismatch_steps = []
    for step in sorted(set(q1_steps) & set(q2_steps)):
        if q1_steps[step]["prompt_hashes"] != q2_steps[step]["prompt_hashes"]:
            hash_mismatch_steps.append(step)
    checks["prompt_hashes_match"] = bool(q1_steps) and not hash_mismatch_steps
    if not checks["prompt_hashes_match"]:
        mismatches.append(f"prompt hash mismatch on {len(hash_mismatch_steps)} steps")

    q1_tokens = sum(int(s["response_tokens"]) for s in q1_steps.values())
    q2_tokens = sum(int(s["response_tokens"]) for s in q2_steps.values())
    token_ratio = (q1_tokens / q2_tokens) if q2_tokens else None
    token_ok = bool(token_ratio is not None and abs(token_ratio - 1.0) <= 0.05)
    checks["token_budget_within_5pct"] = token_ok
    if not token_ok:
        mismatches.append(f"response token budget mismatch: ratio={token_ratio!r}")

    try:
        checks["prompt_instances_match"] = (
            _get(q1, "schedule.prompt_instances") == _get(q2, "schedule.prompt_instances")
        )
    except KeyError:
        checks["prompt_instances_match"] = False
    if not checks["prompt_instances_match"]:
        mismatches.append("prompt instance counts differ or missing")

    try:
        q1_unique = _get(q1, "schedule.unique_prompts_seen")
        q2_unique = _get(q2, "schedule.unique_prompts_seen")
    except KeyError:
        q1_unique = q2_unique = None

    return {
        "matched": not mismatches,
        "mismatches": mismatches,
        "checks": checks,
        "prompt_hash_mismatch_steps": hash_mismatch_steps,
        "q1_response_tokens": q1_tokens,
        "q2_response_tokens": q2_tokens,
        "token_ratio": token_ratio,
        "q1_unique_prompts": q1_unique,
        "q2_unique_prompts": q2_unique,
    }
