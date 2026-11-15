"""Offline-QAD data batching and one teacher-student KL step.

Kept separate from the CLI so the batching and loss plumbing can be unit-tested
on a tiny model. The model forward is done here so the training script only
owns the optimizer/schedule/logging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

from .losses import (
    chunked_forward_kl,
    gather_scored_tokens,
    next_token_score_mask,
)


@dataclass
class Example:
    prompt_ids: list[int]
    response_ids: list[int]
    index: int

    @property
    def total_len(self) -> int:
        return len(self.prompt_ids) + len(self.response_ids)


def make_example(
    record: dict,
    max_seq_len: int,
    index: int,
    max_response_tokens: int | None = None,
) -> Example | None:
    """Prompt + truncated response that fits ``max_seq_len``, or None."""
    prompt_ids = [int(t) for t in record["prompt_ids"]]
    response_ids = [int(t) for t in record["response_ids"]]
    if len(prompt_ids) + 2 > max_seq_len:
        return None
    response_budget = max_seq_len - len(prompt_ids)
    if max_response_tokens is not None:
        response_budget = min(response_budget, max_response_tokens)
    response_ids = response_ids[:response_budget]
    if len(response_ids) < 2:
        return None
    return Example(prompt_ids=prompt_ids, response_ids=response_ids, index=index)


def build_step_batch(
    examples: Sequence[Example],
    cursor: int,
    response_token_budget: int,
) -> tuple[list[Example], int, int]:
    """Pack complete examples up to a response-token budget.

    No example is duplicated or padded; if the next example would exceed the
    budget, the step stops (the caller advances ``cursor``). Returns
    ``(batch, next_cursor, response_tokens)``.
    """
    if response_token_budget < 1:
        raise ValueError("response_token_budget must be >= 1")
    if not examples:
        raise ValueError("no examples")
    batch: list[Example] = []
    tokens = 0
    position = cursor % len(examples)
    while True:
        example = examples[position]
        if batch and tokens + len(example.response_ids) > response_token_budget:
            break
        batch.append(example)
        tokens += len(example.response_ids)
        position = (position + 1) % len(examples)
        if len(batch) == len(examples):
            break
        if tokens >= response_token_budget:
            break
    return batch, position, tokens


def example_to_batch(example: Example, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    ids = torch.tensor(
        [example.prompt_ids + example.response_ids], dtype=torch.long, device=device
    )
    mask = next_token_score_mask(
        torch.tensor([len(example.prompt_ids)], device=device),
        torch.tensor([len(example.response_ids)], device=device),
        ids.shape[1],
        device=device,
    )
    return ids, mask, len(example.prompt_ids), len(example.response_ids)


def forward_kl_step(
    student: nn.Module,
    teacher: nn.Module,
    ids: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, int]:
    """One differentiable forward-KL loss on a single example."""
    student_lm = student.get_output_embeddings().weight
    teacher_lm = teacher.get_output_embeddings().weight
    with torch.no_grad():
        teacher_hidden = teacher.model(input_ids=ids, use_cache=False).last_hidden_state
        h_teacher, targets = gather_scored_tokens(teacher_hidden, ids, mask)
    student_hidden = student.model(input_ids=ids, use_cache=False).last_hidden_state
    h_student, student_targets = gather_scored_tokens(student_hidden, ids, mask)
    if not torch.equal(targets, student_targets):
        raise RuntimeError("teacher/student target mismatch")
    loss = chunked_forward_kl(
        h_student, h_teacher, student_lm, teacher_lm, chunk_size=chunk_size
    )
    return loss, int(targets.numel())


@torch.no_grad()
def evaluate_heldout(
    student: nn.Module,
    teacher: nn.Module,
    blocks,
    batch_size: int,
    chunk_size: int = 256,
) -> dict[str, float]:
    """Teacher-forced NLL and KL(teacher||student) on fixed token blocks."""
    import numpy as np

    student_lm = student.get_output_embeddings().weight
    teacher_lm = teacher.get_output_embeddings().weight
    nll_sum = 0.0
    kl_sum = 0.0
    tokens = 0
    for start in range(0, blocks.shape[0], batch_size):
        input_ids = torch.tensor(
            blocks[start : start + batch_size], dtype=torch.long, device=next(teacher.parameters()).device
        )
        mask = next_token_score_mask(
            torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device),
            torch.full(
                (input_ids.shape[0],), input_ids.shape[1] - 1,
                dtype=torch.long, device=input_ids.device,
            ),
            input_ids.shape[1],
            device=input_ids.device,
        )
        teacher_hidden = teacher.model(input_ids=input_ids, use_cache=False).last_hidden_state
        student_hidden = student.model(input_ids=input_ids, use_cache=False).last_hidden_state
        h_teacher, targets = gather_scored_tokens(teacher_hidden, input_ids, mask)
        h_student, _ = gather_scored_tokens(student_hidden, input_ids, mask)
        from .losses import chunked_nll

        nll = chunked_nll(h_student, targets, student_lm, chunk_size=chunk_size)
        kl = chunked_forward_kl(
            h_student, h_teacher, student_lm, teacher_lm, chunk_size=chunk_size
        )
        count = int(targets.numel())
        nll_sum += float(nll) * count
        kl_sum += float(kl) * count
        tokens += count
    return {
        "nll": nll_sum / tokens,
        "perplexity": math.exp(nll_sum / tokens),
        "kl": kl_sum / tokens,
        "tokens": tokens,
    }
