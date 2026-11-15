"""Tests for chunked losses, masks and next-token alignment.

The naive reference implementations below compute the full ``[N, V]`` logits
and are used to verify that the hidden-state chunking is numerically
equivalent, including gradients. This is the Gate 3 precondition from
plan.md section 6.3.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_qad.losses import (
    chunked_forward_kl,
    chunked_nll,
    chunked_sampled_reverse_kl,
    gather_scored_tokens,
    next_token_score_mask,
)

HIDDEN = 32
VOCAB = 64


def make_batch(n: int = 17, seed: int = 0):
    torch.manual_seed(seed)
    h_student = torch.randn(n, HIDDEN, requires_grad=True)
    h_teacher = torch.randn(n, HIDDEN)
    lm_weight = torch.randn(VOCAB, HIDDEN) * 0.1
    targets = torch.randint(0, VOCAB, (n,))
    return h_student, h_teacher, lm_weight, targets


def naive_nll(h, targets, w):
    logp = F.log_softmax(F.linear(h, w).float(), dim=-1)
    return -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()


def naive_forward_kl(hs, ht, w):
    logp_s = F.log_softmax(F.linear(hs, w).float(), dim=-1)
    logp_t = F.log_softmax(F.linear(ht, w).float(), dim=-1)
    return (logp_t.exp() * (logp_t - logp_s)).sum(dim=-1).mean()


def naive_sampled_rkl(hs, ht, targets, w):
    logp_s = F.log_softmax(F.linear(hs, w).float(), dim=-1)
    logp_t = F.log_softmax(F.linear(ht, w).float(), dim=-1)
    lp_s = logp_s.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    lp_t = logp_t.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    advantage = (lp_s - lp_t).detach()
    return (advantage * lp_s).mean(), advantage.sum()


# ---------------------------------------------------------------------------
# masks and token alignment
# ---------------------------------------------------------------------------


def test_score_mask_positions_offline():
    # prompt 3 tokens, response 2 -> score positions 2 and 3, targets 3 and 4
    mask = next_token_score_mask(torch.tensor([3]), torch.tensor([2]), seq_len=8)
    assert mask.shape == (1, 8)
    assert mask[0].tolist() == [False, False, True, True, False, False, False, False]


def test_score_mask_never_scores_final_position():
    mask = next_token_score_mask(torch.tensor([2]), torch.tensor([5]), seq_len=8)
    assert not mask[0, -1]


def test_score_mask_zero_response_length():
    mask = next_token_score_mask(torch.tensor([3]), torch.tensor([0]), seq_len=8)
    assert not mask.any()


def test_score_mask_batch_rows_are_independent():
    mask = next_token_score_mask(
        torch.tensor([3, 5]), torch.tensor([2, 1]), seq_len=8
    )
    assert mask[0].tolist() == [False, False, True, True, False, False, False, False]
    assert mask[1].tolist() == [False, False, False, False, True, False, False, False]


def test_gather_scored_tokens_alignment():
    torch.manual_seed(0)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    hidden = torch.randn(1, 6, HIDDEN)
    mask = next_token_score_mask(torch.tensor([3]), torch.tensor([2]), seq_len=6)

    h, targets = gather_scored_tokens(hidden, ids, mask)

    assert h.shape == (2, HIDDEN)
    assert targets.tolist() == [3, 4]
    assert torch.equal(h[0], hidden[0, 2])
    assert torch.equal(h[1], hidden[0, 3])


# ---------------------------------------------------------------------------
# chunked vs naive: loss and gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 5, 16, 64])
def test_chunked_nll_matches_naive(chunk_size):
    h, _, w, targets = make_batch()
    h_chunked = h.detach().clone().requires_grad_(True)
    h_naive = h.detach().clone().requires_grad_(True)

    chunked = chunked_nll(h_chunked, targets, w, chunk_size=chunk_size)
    naive = naive_nll(h_naive, targets, w)
    assert torch.allclose(chunked, naive, atol=1e-6)

    chunked.backward()
    naive.backward()
    assert torch.allclose(h_chunked.grad, h_naive.grad, atol=1e-6)


@pytest.mark.parametrize("chunk_size", [2, 7, 32])
def test_chunked_forward_kl_matches_naive(chunk_size):
    h, ht, w, _ = make_batch()
    h_chunked = h.detach().clone().requires_grad_(True)
    h_naive = h.detach().clone().requires_grad_(True)

    chunked = chunked_forward_kl(h_chunked, ht, w, chunk_size=chunk_size)
    naive = naive_forward_kl(h_naive, ht, w)
    assert torch.allclose(chunked, naive, atol=1e-6)

    chunked.backward()
    naive.backward()
    assert torch.allclose(h_chunked.grad, h_naive.grad, atol=1e-6)


@pytest.mark.parametrize("chunk_size", [2, 7, 32])
def test_chunked_sampled_rkl_matches_naive(chunk_size):
    h, ht, w, targets = make_batch()
    h_chunked = h.detach().clone().requires_grad_(True)
    h_naive = h.detach().clone().requires_grad_(True)

    loss_c, kl_sum_c, n_c = chunked_sampled_reverse_kl(
        h_chunked, ht, targets, w, chunk_size=chunk_size
    )
    loss_n, kl_sum_n = naive_sampled_rkl(h_naive, ht, targets, w)

    assert n_c == h.shape[0]
    assert torch.allclose(loss_c, loss_n, atol=1e-6)
    assert torch.allclose(kl_sum_c / n_c, kl_sum_n / h.shape[0], atol=1e-6)

    loss_c.backward()
    loss_n.backward()
    assert torch.allclose(h_chunked.grad, h_naive.grad, atol=1e-6)


def test_chunked_losses_handle_empty_batch_with_grad():
    h = torch.randn(0, HIDDEN, requires_grad=True)
    w = torch.randn(VOCAB, HIDDEN)
    targets = torch.empty(0, dtype=torch.long)

    nll = chunked_nll(h, targets, w)
    assert nll.item() == 0.0
    assert nll.requires_grad

    kl = chunked_forward_kl(h, h.detach(), w)
    assert kl.item() == 0.0


def test_chunked_loss_uses_checkpoint_only_when_inputs_require_grad(monkeypatch):
    import qwen_qad.losses as losses_mod

    calls = {"n": 0}
    original = losses_mod.checkpoint

    def counting_checkpoint(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(losses_mod, "checkpoint", counting_checkpoint)
    h, _, w, targets = make_batch()

    with torch.no_grad():
        chunked_nll(h.detach(), targets, w, chunk_size=8)
    assert calls["n"] == 0

    chunked_nll(h.detach().clone().requires_grad_(True), targets, w, chunk_size=8)
    assert calls["n"] > 0


# ---------------------------------------------------------------------------
# integration with a tiny Qwen3 model
# ---------------------------------------------------------------------------


def tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(config).eval()
    model.lm_head.weight.requires_grad_(False)  # frozen, tied to the embedding
    return model


def test_tiny_model_chunked_nll_matches_full_logits():
    torch.manual_seed(1)
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    prompt_len, response_len = 3, 4

    with torch.no_grad():
        out = model(ids)
        hidden = model.model(ids).last_hidden_state

    mask = next_token_score_mask(
        torch.tensor([prompt_len]), torch.tensor([response_len]), seq_len=ids.shape[1]
    )
    h, targets = gather_scored_tokens(hidden, ids, mask)

    chunked = chunked_nll(h, targets, model.lm_head.weight, chunk_size=2)
    logits = out.logits[:, :-1, :]
    naive = -F.log_softmax(logits.float(), dim=-1)[0, 2:6].gather(
        -1, ids[0, 3:7].unsqueeze(-1)
    ).mean()

    assert targets.tolist() == [4, 5, 6, 7]
    assert torch.allclose(chunked, naive, atol=1e-6)


def test_tiny_model_gradients_reach_transformer_through_chunked_loss():
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    mask = next_token_score_mask(torch.tensor([2]), torch.tensor([4]), seq_len=ids.shape[1])

    hidden = model.model(ids).last_hidden_state
    assert hidden.requires_grad
    h, targets = gather_scored_tokens(hidden, ids, mask)
    loss = chunked_nll(h, targets, model.lm_head.weight, chunk_size=2)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)


def test_self_kl_is_numerically_computed_not_assigned():
    # Same tensors -> exactly zero; perturbed teacher -> positive. Guards against
    # replacing a computed teacher self-KL with a literal zero.
    h, _, w, _ = make_batch()
    with torch.no_grad():
        identical = chunked_forward_kl(h.detach(), h.detach(), w)
        perturbed = h.detach() + 0.05 * torch.randn_like(h)
        different = chunked_forward_kl(perturbed, h.detach(), w)
    assert identical.item() == pytest.approx(0.0, abs=1e-9)
    assert different.item() > 0.0
