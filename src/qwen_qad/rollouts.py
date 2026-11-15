"""Student rollout generation for on-policy QAD.

Prompts are left-padded for batched generation, then each rollout is extracted
back into an unpadded prompt+response sequence for teacher/student scoring.
Padding is never scored and is never passed to the loss.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .generation import ngram_repetition_ratio


def left_pad_prompts(
    prompt_ids_list: Sequence[Sequence[int]], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Left-pad prompts; returns input_ids, attention_mask, pad lengths."""
    lengths = [len(ids) for ids in prompt_ids_list]
    max_len = max(lengths)
    input_ids = torch.full((len(lengths), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(prompt_ids_list):
        start = max_len - len(ids)
        input_ids[row, start:] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, start:] = 1
    return input_ids, attention_mask, [max_len - length for length in lengths]


def extract_rollouts(
    generated: torch.Tensor,
    prompt_lens: Sequence[int],
    pad_lens: Sequence[int],
    eos_token_id: int,
    pad_token_id: int,
) -> list[list[int]]:
    """Strip left padding and trailing pad; keep a generated EOS as the last token."""
    responses: list[list[int]] = []
    for row, prompt_len, pad_len in zip(generated, prompt_lens, pad_lens):
        start = int(pad_len) + int(prompt_len)
        response: list[int] = []
        for token in row[start:].tolist():
            if token == pad_token_id:
                break
            response.append(int(token))
            if token == eos_token_id:
                break
        while response and response[-1] == pad_token_id:
            response.pop()
        responses.append(response)
    return responses


@torch.no_grad()
def generate_rollouts(
    model: nn.Module,
    tokenizer,
    prompt_ids_list: Sequence[Sequence[int]],
    max_new_tokens: int = 512,
    batch_size: int = 8,
    seed: int = 42,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
) -> list[dict]:
    """Sample one rollout per prompt with identical settings for every batch."""
    device = next(model.parameters()).device
    model.eval()
    results: list[dict | None] = [None] * len(prompt_ids_list)
    order = sorted(range(len(prompt_ids_list)), key=lambda i: -len(prompt_ids_list[i]))

    for batch_start in range(0, len(order), batch_size):
        indices = order[batch_start : batch_start + batch_size]
        prompts = [prompt_ids_list[i] for i in indices]
        input_ids, attention_mask, pad_lens = left_pad_prompts(
            prompts, tokenizer.pad_token_id, device
        )
        torch.manual_seed(seed + batch_start)
        generated = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        responses = extract_rollouts(
            generated,
            [len(p) for p in prompts],
            pad_lens,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
        for idx, prompt, response in zip(indices, prompts, responses):
            full = list(prompt) + list(response)
            results[idx] = {
                "index": idx,
                "prompt_ids": list(prompt),
                "response_ids": response,
                "full_ids": full,
                "prompt_len": len(prompt),
                "response_len": len(response),
                "eos_generated": bool(response) and response[-1] == tokenizer.eos_token_id,
                "repetition_4gram": ngram_repetition_ratio(response, n=4),
            }
    return [r for r in results if r is not None]
