"""Shared prompt schedule for matched offline/on-policy comparisons.

Both arms must see the same prompt identifiers at the same optimizer step. The
schedule is a seeded permutation of records that satisfy the prompt-length and
continuation-cap constraints; steps read fixed-size slices with wrap-around.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Sequence


def load_prompt_records(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open() as handle:
        for line in handle:
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"no prompt records in {path}")
    return records


def build_schedule(
    records: Sequence[dict],
    max_seq_len: int,
    continuation_cap: int,
    seed: int,
) -> list[int]:
    """Indices eligible under both arms, shuffled with the given seed."""
    eligible = []
    for index, record in enumerate(records):
        prompt_len = len(record["prompt_ids"])
        if len(record.get("response_ids", [])) < 2:
            continue
        if prompt_len + continuation_cap + 2 > max_seq_len:
            continue
        eligible.append(index)
    if not eligible:
        raise RuntimeError("no eligible prompts for the requested length cap")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible


def step_indices(schedule: Sequence[int], step: int, prompts_per_step: int) -> list[int]:
    """Prompt indices for 1-based ``step``; deterministic wrap-around."""
    if prompts_per_step < 1:
        raise ValueError("prompts_per_step must be >= 1")
    if prompts_per_step > len(schedule):
        raise ValueError("prompts_per_step exceeds schedule size")
    start = ((step - 1) * prompts_per_step) % len(schedule)
    return [schedule[(start + offset) % len(schedule)] for offset in range(prompts_per_step)]


def prompt_hash(prompt_ids: Sequence[int]) -> str:
    payload = json.dumps([int(t) for t in prompt_ids]).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def step_prompt_hashes(records: Sequence[dict], indices: Sequence[int]) -> list[str]:
    return [prompt_hash(records[i]["prompt_ids"]) for i in indices]
