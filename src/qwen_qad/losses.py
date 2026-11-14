"""Masked NLL, forward KL and sampled reverse KL with hidden-state chunking.

Design (plan.md section 6.3): the transformer runs once and produces hidden
states at the scored positions. The LM head is applied chunk-by-chunk inside
``torch.utils.checkpoint``, so the full ``[N_resp, V]`` logits tensor is never
materialized, and the transformer is traversed once in forward and once in
backward. Slicing already-materialized logits would save nothing and is not
used.

All functions operate on *already gathered* hidden states (``[N, H]``) and
target token ids (``[N]``). Use :func:`next_token_score_mask` and
:func:`gather_scored_tokens` to build those from model inputs.

Loss temperature is 1.0: no temperature scaling is applied here. The caller
may scale logits before gathering if a temperature variant is studied.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

DEFAULT_CHUNK_SIZE = 256


def next_token_score_mask(
    prompt_lens: Tensor,
    response_lens: Tensor,
    seq_len: int,
    device: torch.device | None = None,
) -> Tensor:
    """Boolean ``[B, T]`` mask over predictor positions.

    Position ``t`` is scored iff it predicts a response token, i.e.
    ``prompt_len - 1 <= t <= prompt_len + response_len - 2``. Position
    ``T - 1`` is never scored because it has no target. The first response
    token is predicted by the last prompt position (the classic off-by-one).
    """
    if prompt_lens.shape != response_lens.shape:
        raise ValueError("prompt_lens and response_lens must have the same shape")
    if prompt_lens.dim() != 1:
        raise ValueError("prompt_lens and response_lens must be 1-D")
    device = device or prompt_lens.device
    t = torch.arange(seq_len, device=device).unsqueeze(0)
    start = (prompt_lens - 1).unsqueeze(1).to(device)
    end = (prompt_lens + response_lens - 2).unsqueeze(1).to(device)
    return (t >= start) & (t <= end) & (t <= seq_len - 2)


def gather_scored_tokens(hidden: Tensor, input_ids: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Select ``[N, H]`` hidden states and ``[N]`` next-token targets.

    ``hidden`` is ``[B, T, H]`` and ``mask`` is the ``[B, T]`` output of
    :func:`next_token_score_mask`. Targets are ``input_ids[:, 1:][mask[:, :-1]]``.
    """
    if hidden.dim() != 3 or input_ids.dim() != 2 or mask.dim() != 2:
        raise ValueError("expected hidden [B,T,H], input_ids [B,T], mask [B,T]")
    if hidden.shape[:2] != input_ids.shape or hidden.shape[:2] != mask.shape:
        raise ValueError("inconsistent batch/sequence dimensions")
    target_mask = mask[:, :-1]
    targets = input_ids[:, 1:][target_mask]
    return hidden[mask], targets


def _maybe_checkpoint(fn, *args):
    if torch.is_grad_enabled() and any(isinstance(a, Tensor) and a.requires_grad for a in args):
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def _nll_chunk(hidden: Tensor, targets: Tensor, lm_weight: Tensor, bias: Tensor | None) -> Tensor:
    logits = F.linear(hidden, lm_weight, bias).float()
    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -token_logp.mean()


def chunked_nll(
    hidden: Tensor,
    targets: Tensor,
    lm_weight: Tensor,
    bias: Tensor | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Tensor:
    """Mean next-token NLL over scored positions."""
    if hidden.shape[0] == 0:
        return (hidden.sum() * 0.0).float()
    total = hidden.new_zeros((), dtype=torch.float32)
    count = 0
    for start in range(0, hidden.shape[0], chunk_size):
        h = hidden[start : start + chunk_size]
        t = targets[start : start + chunk_size]
        loss = _maybe_checkpoint(_nll_chunk, h, t, lm_weight, bias)
        total = total + loss.float() * h.shape[0]
        count += h.shape[0]
    return total / count


def _forward_kl_chunk(
    h_student: Tensor,
    h_teacher: Tensor,
    w_student: Tensor,
    w_teacher: Tensor,
    bias: Tensor | None,
) -> Tensor:
    logits_t = F.linear(h_teacher, w_teacher, bias).float()
    logits_s = F.linear(h_student, w_student, bias).float()
    logp_t = F.log_softmax(logits_t, dim=-1)
    logp_s = F.log_softmax(logits_s, dim=-1)
    p_t = logp_t.exp()
    return (p_t * (logp_t - logp_s)).sum(dim=-1).mean()


def chunked_forward_kl(
    h_student: Tensor,
    h_teacher: Tensor,
    lm_weight: Tensor,
    lm_weight_teacher: Tensor | None = None,
    bias: Tensor | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Tensor:
    """Mean forward KL ``KL(p_teacher || p_student)`` over scored positions."""
    if h_student.shape[0] != h_teacher.shape[0]:
        raise ValueError("student and teacher hidden states must have the same length")
    if h_student.shape[0] == 0:
        return (h_student.sum() * 0.0).float()
    w_teacher = lm_weight if lm_weight_teacher is None else lm_weight_teacher
    total = h_student.new_zeros((), dtype=torch.float32)
    count = 0
    for start in range(0, h_student.shape[0], chunk_size):
        hs = h_student[start : start + chunk_size]
        ht = h_teacher[start : start + chunk_size]
        loss = _maybe_checkpoint(_forward_kl_chunk, hs, ht, lm_weight, w_teacher, bias)
        total = total + loss.float() * hs.shape[0]
        count += hs.shape[0]
    return total / count


def _sampled_rkl_chunk(
    h_student: Tensor,
    h_teacher: Tensor,
    targets: Tensor,
    w_student: Tensor,
    w_teacher: Tensor,
    bias: Tensor | None,
) -> tuple[Tensor, Tensor]:
    logp_s = F.log_softmax(F.linear(h_student, w_student, bias).float(), dim=-1)
    logp_t = F.log_softmax(F.linear(h_teacher, w_teacher, bias).float(), dim=-1)
    idx = targets.unsqueeze(-1)
    lp_s = logp_s.gather(-1, idx).squeeze(-1)
    lp_t = logp_t.gather(-1, idx).squeeze(-1)
    advantage = (lp_s - lp_t).detach()
    surrogate = (advantage * lp_s).mean()
    return surrogate, advantage.sum()


def chunked_sampled_reverse_kl(
    h_student: Tensor,
    h_teacher: Tensor,
    targets: Tensor,
    lm_weight: Tensor,
    lm_weight_teacher: Tensor | None = None,
    bias: Tensor | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[Tensor, Tensor, int]:
    """Single-sample reverse-KL policy-gradient objective.

    Returns ``(surrogate_loss, kl_estimate_sum, n_tokens)``. The surrogate is
    minimized; it is **not** a KL value. ``kl_estimate_sum / n_tokens`` is the
    unbiased single-sample estimate of ``KL(p_student || p_teacher)`` and must
    be logged separately.
    """
    if h_student.shape[0] != h_teacher.shape[0] or h_student.shape[0] != targets.shape[0]:
        raise ValueError("student, teacher and targets must have the same length")
    if h_student.shape[0] == 0:
        zero = (h_student.sum() * 0.0).float()
        return zero, zero, 0
    w_teacher = lm_weight if lm_weight_teacher is None else lm_weight_teacher
    surrogate_total = h_student.new_zeros((), dtype=torch.float32)
    kl_total = h_student.new_zeros((), dtype=torch.float32)
    n = 0
    for start in range(0, h_student.shape[0], chunk_size):
        hs = h_student[start : start + chunk_size]
        ht = h_teacher[start : start + chunk_size]
        tg = targets[start : start + chunk_size]
        surrogate, kl_sum = _maybe_checkpoint(
            _sampled_rkl_chunk, hs, ht, tg, lm_weight, w_teacher, bias
        )
        surrogate_total = surrogate_total + surrogate.float() * hs.shape[0]
        kl_total = kl_total + kl_sum.float()
        n += hs.shape[0]
    return surrogate_total / n, kl_total, n
