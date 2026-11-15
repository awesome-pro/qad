# Qwen3 low-bit quantization + online QAD (research pilot)

Research question: **can quantization-aware distillation (QAD) recover a
weight-only low-bit Qwen3 student, and does collecting training data
on-policy — on the student's own rollouts — help compared with fixed-prefix
offline distillation?**

This repository follows `QWEN_LOW_BIT_QAD_RESEARCH_BRIEF.md`. Where the brief
and the published *OnlineQAT* paper (arXiv:2610.09346) differ, **the brief
wins**; the paper is reference material only, not a reproduction target.

## Scope (locked)

| Decision | Choice |
|---|---|
| Models | `Qwen/Qwen3-0.6B` for pipeline validation, then `Qwen/Qwen3-1.7B` for the headline run (separate decision) |
| Precision | W3A16 weight-only fake quantization (BF16 activations) |
| Scheme | signed symmetric, `q in [-3, 3]` (one code unused), group size 128 along `in_features`, per output row, no zero-point; asymmetric affine available behind a flag |
| Student init | copy pretrained weights, then fake-quantize (round-to-nearest Q0). **No block-wise QAT initialization** |
| Stage 2 | primary: offline forward KL on fixed prefixes (Q1) vs on-policy forward KL (Q2). Optional third arm: on-policy sampled reverse KL (Q3), decided after Q1/Q2 |
| Trainable | only the wrapped linear masters; embeddings, LM head and norms frozen |
| Evaluation | two tiers: NLL/teacher-KL on a fixed held-out set for every run, plus compact benchmark subsets for the headline variants |

Unquantized: embeddings, LM head (tied in Qwen3, so one tensor), all
normalization layers, attention biases (none in Qwen3).

## Status

| Increment | State |
|---|---|
| 0. Reference quantizer, losses, wrapping + tests | **done locally** (89 collected, 88 pass + 1 CUDA-only skip) |
| 1. Checkpoint inventory, fixed eval data, T0/Q0 smoke | **done locally** (not reportable) |
| 1.5. W3A16 degradation diagnosis (BF16/W4/W3-G32/G64/G128, attn vs MLP) | **done locally** — see [diagnosis](experiments/gate15/DIAGNOSIS.md) |
| 2. Official T0 + Q0 on the held-out set, memory accounting | **next** (first GPU session) |
| 3. Offline QAD, checkpoint round-trip | needs GPU |
| 4. Rollouts + on-policy Q2 | needs GPU |
| 5. Benchmark subsets, report | needs GPU |

Local Gate 1.5 establishes that the W3A16-G128 degradation is intrinsic to
pure RTN, not a wrapper bug: BF16 ppl 22.4, W4A16 ppl 30.7, W3A16-G32 96.5,
G64 223.2, G128 754.4 on 65,472 held-out tokens. Recommendation: keep W3 as the
primary arm and add W4A16-G128 as a separate variant. Details in
`experiments/gate15/DIAGNOSIS.md`; no model-quality number from the Mac is
reportable.

## Environment

Local development uses `uv` (Python 3.12, CPU/MPS); CUDA runs use the same
`pyproject.toml` on a rented GPU. Note that `uv` caches must stay inside the
workspace when the file sandbox is active:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite checks hand-computed quantization values, group shapes and tail
groups, no cross-row leakage, all-zero and constant groups, code ranges,
reconstruction formulas, identity-STE gradients, fake-forward equality with
explicit quantize-dequantize, and that the fp32 master weight is never
overwritten. It also covers exact tail-group bit accounting, the distinction
between endpoint occupancy and clipping, device-preserving conversion,
dequantized-weight cache invalidation, checkpoint-derived inventories,
quantization wrapping, causal token alignment, and chunked-loss equivalence
(losses and gradients) against a naive full-logits reference, plus the
EOS-after-truncation policy, sampled memory measurement, quantization-disabled
equivalence, and the Gate 1.5 diagnostic helpers.

## References

- Research brief: `QWEN_LOW_BIT_QAD_RESEARCH_BRIEF.md` (authoritative)
- Reference paper (not a target): <https://arxiv.org/abs/2610.09346>
