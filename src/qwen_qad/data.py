"""Data helpers shared by preparation and training code."""

from __future__ import annotations

from typing import Sequence


def truncate_response(
    full_ids: Sequence[int], max_tokens: int, eos_token_id: int
) -> tuple[list[int], bool]:
    """Truncate a completion and append EOS only when it is complete.

    Appending EOS after an artificial cutoff would teach the model that the
    reasoning trace ends mid-thought, which is wrong for QAD training. Returns
    ``(ids, truncated)``.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    truncated = len(full_ids) > max_tokens
    ids = [int(t) for t in full_ids[:max_tokens]]
    if not truncated:
        ids.append(int(eos_token_id))
    return ids, truncated
