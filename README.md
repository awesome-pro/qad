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
| Models | `Qwen/Qwen3-0.6B` for pipeline validation, then `Qwen/Qwen3-1.7B` for the headline run |
| Precision | W3A16 weight-only fake quantization (BF16 activations) |
| Scheme | signed symmetric, `q in [-3, 3]` (one code unused), group size 128 along `in_features`, per output row, no zero-point; asymmetric affine available behind a flag |
| Student init | copy pretrained weights, then fake-quantize (round-to-nearest Q0). **No block-wise QAT initialization** |
| Stage 2 | offline forward KL on fixed prefixes (Q1) vs on-policy forward KL (Q2 control) vs on-policy sampled reverse KL (Q3 variant) |
| Evaluation | two tiers: NLL/teacher-KL on a fixed held-out set for every run, plus compact benchmark subsets for the headline variants |

Unquantized: embeddings, LM head (tied in Qwen3, so one tensor), all
normalization layers, attention biases (none in Qwen3).

## Status

| Increment | State |
|---|---|
| 0. Reference quantizer + STE wrapper + unit tests | **done** (28 tests, CPU) |
| 1. Model loading, module inventory, fixed eval sets, T0 baseline | next |
| 2. Fake-quant wrapping, Q0 evaluation, bit accounting | needs GPU |
| 3. Offline QAD, checkpoint round-trip | needs GPU |
| 4. Rollouts + on-policy objectives | needs GPU |
| 5. Benchmark subsets, report | needs GPU |

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
overwritten.

## References

- Research brief: `QWEN_LOW_BIT_QAD_RESEARCH_BRIEF.md` (authoritative)
- Reference paper (not a target): <https://arxiv.org/abs/2610.09346>
