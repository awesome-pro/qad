"""Small, model-agnostic generation-quality metrics.

These are descriptive diagnostics, not benchmark scores: repetition is measured
on token 4-grams, premature EOS is an EOS emitted before ``min_new_tokens``
content tokens, and examples are saved verbatim for human inspection.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def _as_ints(token_ids: Sequence[int]) -> list[int]:
    return [int(t) for t in token_ids]


def ngram_repetition_ratio(token_ids: Sequence[int], n: int = 4) -> float:
    """Fraction of n-gram occurrences that are repeats (0 = all distinct)."""
    ids = _as_ints(token_ids)
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def max_repeated_ngram_count(token_ids: Sequence[int], n: int = 4) -> int:
    ids = _as_ints(token_ids)
    if len(ids) < n:
        return 0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return max(Counter(grams).values())


def distinct_ratio(token_ids: Sequence[int], n: int = 2) -> float:
    ids = _as_ints(token_ids)
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def generation_summary(
    token_ids: Sequence[int],
    prompt_len: int,
    eos_token_id: int,
    min_new_tokens: int = 32,
) -> dict[str, float | int | bool | None]:
    """Summarize one generated sequence (prompt included in ``token_ids``)."""
    ids = _as_ints(token_ids)
    if prompt_len < 0 or prompt_len > len(ids):
        raise ValueError(f"invalid prompt_len {prompt_len} for {len(ids)} tokens")
    generated = ids[prompt_len:]
    eos_position = next(
        (index for index, token in enumerate(generated) if token == eos_token_id), None
    )
    content = generated if eos_position is None else generated[:eos_position]
    return {
        "generated_tokens": len(generated),
        "content_tokens": len(content),
        "eos_generated": eos_position is not None,
        "eos_position": eos_position,
        "premature_eos": eos_position is not None and eos_position < min_new_tokens,
        "hit_max_tokens": eos_position is None,
        "empty": len(content) == 0,
        "repetition_4gram": ngram_repetition_ratio(content, n=4),
        "max_repeated_4gram": max_repeated_ngram_count(content, n=4),
        "distinct_1": distinct_ratio(content, n=1),
        "distinct_2": distinct_ratio(content, n=2),
    }
