# Qwen3 low-bit quantization + online QAD (research pilot)

Research question: **can quantization-aware distillation (QAD) recover a
weight-only low-bit Qwen3 student, and does collecting training data
on-policy — on the student's own rollouts — help compared with fixed-prefix
offline distillation?**

> **Audit in progress (2026-11-15).** The GSM8K numbers in the historical report
> used a permissive parser and a 512-token cap; they are being replaced by the
> strict evaluator in `scripts/eval_gsm8k_strict.py`, and Q1/Q2 are being rerun
> with matched schedules (`scripts/train_matched.py`). See `review.md`,
> `GPU_RUNBOOK_AUDIT.md`, and `plan.md` §15; first corrected results are in
> `reports/checkpoint_a.md`. Old artifacts are preserved as historical.

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
| 0. Reference quantizer, losses, wrapping + tests | **done locally** (142 collected, 141 pass + 1 CUDA-only skip) |
| 1. Checkpoint inventory, fixed eval data, T0/Q0 smoke | **done locally** (not reportable) |
| 1.5. W3A16 degradation diagnosis (BF16/W4/W3-G32/G64/G128, attn vs MLP) | **done locally** — see [diagnosis](experiments/gate15/DIAGNOSIS.md) |
| 2. Official baseline (4 configs), generation diagnostic, training probe | **done on CUDA** (RTX A5000 24 GB) |
| 3. Offline QAD on W4A16-G128, checkpoint round-trip | **done** (Q1-200: ppl 26.28, GSM8K 28%) |
| 4. Rollouts + on-policy Q2 | **done** (200 steps: ppl 26.25, GSM8K 14%; matched 25-step arm ppl 27.17, GSM8K 16%) |
| 5. Benchmark subset + report | **done** (100 GSM8K, `reports/findings.md`) |
| 5b. Second seed + ablation + 1.7B + W3 probe | **done** (seed-43 replicates; 256-token ablation; 1.7B Q1-200 17.84 ppl / 27% GSM; W3 QAD 59.35 ppl but 0% GSM) |

Gate 2 confirms the order on CUDA (A5000 24 GB, full 256-block held-out set):
BF16 ppl 22.26, W4A16-G128 30.59, W3A16-G32 94.44, W3A16-G128 723.98.
W3-G128 generation is degenerate repetition (4-gram 0.965 vs 0.047 BF16); W4
stays readable (0.117). The 3-step probe passes every assertion at 9.32 GiB
peak on both W4 and W3. W3-G128 is therefore the **aggressive RTN baseline**,
and **W4A16-G128 is the pipeline-verification arm**; W3-G32 and a stronger W3
initialization remain separate options. Details:
`experiments/gate15/DIAGNOSIS.md` and `experiments/gate2/README.md`. No QAD
training has run.

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
