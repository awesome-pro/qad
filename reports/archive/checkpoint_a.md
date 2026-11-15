# Checkpoint A - strict grader and corrected evaluation

> **Superseded metrics:** the numbers below use parser v1. See
> `reports/checkpoint_a_patch.md` for parser v2 and the regraded results.

**Purpose:** let the parent reviewer verify the GSM8K grader, Qwen3 thinking-mode
handling, and evaluation artifacts before any further 200-step training.

**Scope of this checkpoint:** code, tests, disjoint data splits, one matched
Q1/Q2 seed-42 pair, and strict non-thinking/thinking evaluations. Seed 43, W3,
and corrected 1.7B QAD were **not** run (per reviewer sequencing).

## 1. What changed

- `src/qwen_qad/answer_parsing.py`: strict parser. Only explicit `####`,
  `\boxed{}`, or line-start `Answer:` markers count. No last-number fallback.
  Thinking mode requires a closing `</think>`; unfinished reasoning is
  `incomplete_thinking` and is never graded.
- `scripts/eval_gsm8k_strict.py`: non-thinking and thinking evaluators. Reports
  accuracy with bootstrap 95% CI, completion rate, truncation rate,
  thinking-closed rate and parse-status counts separately. Saves full prompt
  and response token IDs, full decoded text, and parse text per example.
- `scripts/train_matched.py`: matched Q1/Q2 protocol. Both arms use the same
  prompt schedule, prompt IDs per step, 32 prompts/step, 128-token
  continuation cap, 200 steps, LR schedule, teacher, quantizer and seed.
  Saves best (validation NLL) and final checkpoints, hashes, and portable
  safetensors inference weights.
- `scripts/check_matched.py`: verifies prompt hashes per step, step count,
  config, prompt-instance counts and <5% response-token mismatch.
- `scripts/compare_gsm8k.py`: paired bootstrap difference and exact McNemar.
- `scripts/prepare_splits.py` + `data/manifests/gate8_splits_manifest.json`:
  disjoint validation and final-test blocks.

## 2. Qwen3 tokenizer / thinking-mode verification

Verified against the pinned tokenizer
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`:

- `<think>` is added token **151667**.
- `</think>` is added token **151668**.
- EOS is `<|im_end|>` (**151645**).
- `apply_chat_template(..., enable_thinking=False)` pre-fills
  `<|im_start|>assistant\n<think>\n\n</think>\n\n`; the completion is answer-only.
- `apply_chat_template(..., enable_thinking=True)` ends at
  `<|im_start|>assistant\n`; the model must generate `<think>...</think>`
  before any answer region exists.

The test suite asserts the ASCII markers and, when the pinned tokenizer is in
the local cache, asserts IDs 151667/151668.

## 3. Data splits

| Split | Source | Blocks | Tokens | Notes |
|---|---|---:|---:|---|
| Validation | original FineWeb-Edu held-out file, first 128 blocks | 128 | 131,072 | checkpoint selection only |
| Final test | WikiText-103 test, pinned revision | 128 | 131,072 | never used for selection |

Block-hash intersection is **0**. Manifest includes dataset revision, license
(`cc-by-sa-3.0 / gfdl` for WikiText), array hashes and tokenizer revision.
WikiText is a public benchmark and pretraining contamination is possible; this
is stated as a limitation.

## 4. Matched run check (seed 42)

Both runs: 200 steps × 32 prompts/step = 6,400 prompt instances; 3,000 unique
prompts; continuation cap 128; W4A16-G128; same teacher and LR schedule.

| Check | Result |
|---|---|
| Prompt hashes identical on all 200 steps | **yes** (0 mismatched steps) |
| Prompt instances | 6,400 / 6,400 |
| Response tokens | Q1 819,200; Q2 818,232; ratio 1.0012 |
| Quantization / model / teacher / LR / steps / val hash | identical |
| `check_matched.py` | `matched: true`, no mismatches |

## 5. Results

### Strict GSM8K, non-thinking, 100 test examples

Same 100 questions, same instruction, 256-token cap, temperature 0.6/top-p 0.95/
top-k 20, one sample per question.

| Variant | Accuracy | 95% CI | Complete | Truncated | Parse status |
|---|---:|---:|---:|---:|---|
| T0 BF16 | **38%** | 29–48% | 76% | 24% | ok 59, missing 36, malformed 5 |
| Q0 W4-G128 RTN | 13% | 7–20% | 50% | 50% | ok 43, missing 51, malformed 6 |
| Q1-200 matched (best step 150) | 34% | 25–44% | 64% | 36% | ok 63, missing 34, malformed 3 |
| Q2-200 matched (best step 150) | 36% | 27–46% | 72% | 28% | ok 70, missing 27, malformed 3 |

Paired Q1 vs Q2 on the same questions: **Q2 − Q1 = +2 points**, 95% CI
[−4, +9], exact McNemar p = 0.77. This is not evidence of a difference.

### Held-out language modelling (WikiText-103 final test, lower is better)

| Variant | NLL | PPL | KL(T‖S) |
|---|---:|---:|---:|
| T0 BF16 | 3.2395 | 25.52 | 0.0000 |
| Q0 W4-G128 | 3.6052 | 36.79 | 0.4520 |
| Q1-200 matched | 3.4222 | 30.64 | 0.3070 |
| Q2-200 matched | **3.3986** | **29.92** | 0.3124 |

On the disjoint test set, Q2 is slightly better than Q1 on NLL/PPL; the
difference is small relative to domain shift (validation FineWeb PPL ~26.6 for
both, test WikiText PPL ~30).

### Thinking-mode subset (pilot, n = 10)

Cap 1,536 tokens; this is a **completion/parser sanity subset, not an accuracy
benchmark**.

| Variant | Accuracy | Complete | Truncated | Thinking closed | Parse status |
|---|---:|---:|---:|---:|---|
| T0 | 10% | 60% | 40% | 60% | ok 3, missing 2, incomplete 4, malformed 1 |
| Q0 | 0% | 0% | 100% | 0% | incomplete 10 |
| Q1 | 10% | 30% | 70% | 40% | ok 2, missing 1, incomplete 6, malformed 1 |
| Q2 | 0% | 10% | 90% | 10% | ok 1, incomplete 9 |

The old 512-token thinking numbers were therefore not valid reasoning accuracy.
Even 1,536 tokens is insufficient for most QAD students; a proper thinking
benchmark needs a larger cap and a larger subset, and must report completion
separately rather than grading incomplete traces.

## 6. What this changes relative to the historical report

- Old GSM8K Q1 28% vs Q2 14% is **not reproduced** under strict non-thinking
  grading: Q1 34% vs Q2 36% (paired p = 0.77). The old gap was substantially a
  parser/cap and unmatched-protocol artifact.
- Q0 is not at 1%: with direct non-thinking answers it scores 13%; its failure
  is concentrated in long reasoning/truncation.
- The old claim that offline QAD recovers much more generative quality than
  on-policy is not supported by this corrected pair. Both recover most of the
  T0–Q0 non-thinking gap, and their difference is within noise.
- On disjoint WikiText test NLL, Q2 is slightly better than Q1; this also does
  not support a simple “offline is better” conclusion.

## 7. Limitations

- One seed (42) for the corrected matched pair; 100 non-thinking examples.
- Thinking subset is 10 examples and clearly underpowered; it is for parser and
  completion verification only.
- Best checkpoints are selected on FineWeb validation NLL; WikiText test is
  disjoint but a different domain.
- WikiText pretraining contamination is possible.
- No seed 43, W3, or corrected 1.7B run in this checkpoint.
- Q1/Q2 optimizers are not retained in the portable safetensors; `.pt` best/
  final checkpoints exist on the pod and are copied separately when needed.

## 8. Files to verify

```text
src/qwen_qad/answer_parsing.py
scripts/eval_gsm8k_strict.py
scripts/train_matched.py
scripts/check_matched.py
scripts/compare_gsm8k.py
scripts/prepare_splits.py
data/manifests/gate8_splits_manifest.json
experiments/gate8/gsm8k_*_nonthink.{json,generations.jsonl}
experiments/gate8/gsm8k_*_think10.{json,generations.jsonl}
experiments/gate8/m_q1_w4_s42.json
experiments/gate8/m_q2_w4_s42.json
experiments/gate8/match_check_s42.json
experiments/gate8/gsm8k_m_q1_vs_q2_s42_nonthink.json
```

Tests: `pytest -q` → **143 collected, 141 pass, 2 environment skips**
(CUDA-only quantizer test; pinned-tokenizer test when the tokenizer is not in
the local cache).

## 9. Decision requested

Please verify the grader and these outputs. If approved, the next batch is:
seed 43 matched pair, W3 matched pair, corrected 1.7B baseline/Q1, then the
final report rewrite. No further 200-step jobs have been started.
