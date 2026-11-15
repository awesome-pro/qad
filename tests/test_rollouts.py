"""Tests for rollouts padding/extraction and on-policy batching."""

from __future__ import annotations

import torch

from qwen_qad.rollouts import extract_rollouts, left_pad_prompts


def test_left_pad_prompts_shapes_and_mask():
    ids, mask, pads = left_pad_prompts([[1, 2], [3, 4, 5]], pad_token_id=0, device=torch.device("cpu"))
    assert ids.shape == (2, 3)
    assert pads == [1, 0]
    assert ids[0].tolist() == [0, 1, 2]
    assert ids[1].tolist() == [3, 4, 5]
    assert mask[0].tolist() == [0, 1, 1]
    assert mask[1].tolist() == [1, 1, 1]


def test_extract_rollouts_strips_padding_and_keeps_eos():
    # row 0: pad 1, prompt 2, response [7, 8, EOS, pad]
    # row 1: pad 0, prompt 3, response [9, EOS, pad, pad]
    generated = torch.tensor(
        [
            [0, 1, 2, 7, 8, 99, 0],
            [3, 4, 5, 9, 99, 0, 0],
        ]
    )
    responses = extract_rollouts(
        generated, prompt_lens=[2, 3], pad_lens=[1, 0], eos_token_id=99, pad_token_id=0
    )
    assert responses[0] == [7, 8, 99]
    assert responses[1] == [9, 99]


def test_extract_rollouts_without_eos_hits_cap():
    generated = torch.tensor([[1, 2, 3, 4, 5, 0]])
    responses = extract_rollouts(
        generated, prompt_lens=[2], pad_lens=[0], eos_token_id=99, pad_token_id=0
    )
    assert responses[0] == [3, 4, 5]
