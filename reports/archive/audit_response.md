# Response to the external audit (`review.md`)

Status: Mac-side corrections landed; GPU reruns pending. Old results are
preserved but labelled historical where the review invalidates them.

## Agreement summary

The review is correct on every material methodological point: the old GSM8K
metric was not strict, Q1/Q2 were not a matched comparison, validation and test
overlapped, checkpoint selection was not implemented, checkpoints were not
durable, several API correctness properties lacked direct tests, and the report
overstated what 100 GSM8K examples and one seed can establish.

## Point-by-point

| Priority | Verdict | Action taken / blocker |
|---|---|---|
| P0 strict GSM8K | **Agree** | New `answer_parsing.py` accepts only explicit `####`, `\boxed{}`, or line-start `Answer:` markers; thinking mode requires a closed `</think>`. Qwen3 markers verified against the pinned tokenizer (`<think>`=151667, `</think>`=151668), not assumed. New `eval_gsm8k_strict.py` supports non-thinking and thinking modes, stores prompt/response token ids + full text, and reports completion/truncation/thinking-closed/parse status separately. |
| P0E regrade old checkpoints | **Blocked** | The old checkpoints were deleted after each evaluation and the pods are gone. The review assumed they were available; they are not. Corrected evaluation requires retraining the matched arms (cheap: ~5 min Q1 + ~40 min Q2 per seed). This is documented in `GPU_RUNBOOK_AUDIT.md`. |
| P1 matched Q1/Q2 | **Agree** | New `schedule.py` + `train_matched.py`: identical prompt schedule, prompt IDs per step, 32 prompts/step, 128-token continuation cap, steps, LR, teacher, quantizer, seed. `check_matched.py` verifies hashes and token ratio (<5%). Old unmatched results remain historical. |
| P2 disjoint validation/test | **Agree** | Validation = first 128 blocks of the original FineWeb-Edu file; final test = newly prepared WikiText-103 test (128 blocks, pinned revision/license). Block-hash intersection is 0. Manifest: `data/manifests/gate8_splits_manifest.json`. |
| P3 best checkpoint | **Agree** | `train_matched.py` selects best by validation NLL at eval intervals, saves best and final separately, records step/metric/sha256, and reloads. Historical Q1-200 seed42 best was step 100 (26.06) vs final 26.26; seed43 26.03 vs 26.09; Q2 26.00 vs 26.07. |
| P4 durable checkpoints | **Agree, with a factual correction** | Old checkpoints are gone, not merely undiscoverable. New runs save best/final `.pt` plus portable inference `.safetensors` (optimizer omitted) and hashes. |
| P5 correctness tests | **Agree** | Added tests for strict parsing, Qwen3 markers, schedule identity, matched-run comparison, portable weight roundtrip, and rollouts using updated master weights. Some properties were already covered (teacher freeze, cache invalidation, loss alignment, left-padding extraction). |
| P6 memory accounting | **Agree** | Added theoretical packed-size table (`reports/memory_accounting.md`) with payload/metadata breakdown. It explicitly excludes packing overhead and makes no serving-speedup claim. |
| P7 statistics | **Agree** | Evaluator now reports bootstrap 95% CI; `compare_gsm8k.py` reports paired bootstrap difference and exact McNemar counts. Old single-seed/100-example claims are labelled exploratory; 1.7B is exploratory. |
| P8 report | **Pending GPU** | Historical report carries an audit notice; final tables await the corrected runs. |

## One nuance / partial disagreement

P0 says "nearly all generations hit the token limit". This is correct for the
old run (mean 508–512 of 512), but the fallback parser also produced a 96–100%
"parse rate"; the real damage is that a truncated trace could be scored from an
intermediate number. The strict evaluator therefore reports truncation and
completion separately rather than only accuracy.

P6 says the report must not present whole-model size reductions as deployment
memory savings. No such savings were claimed, but the new table makes the
distinction explicit; no packed artifact is produced.

## Checkpoint A outcome (seed 42, 0.6B)

The first corrected matched pair is complete and written up in
`reports/checkpoint_a.md`.

- Strict non-thinking GSM8K (100 questions, parser v2 regrade): T0 37%, Q0 13%,
  Q1 33%, Q2 31%. Paired Q2−Q1 = −2 points, 95% CI [−9,+5], McNemar p = 0.77.
- The old Q1 28%/Q2 14% gap is not reproduced; parser, decoding, checkpoint
  selection and matching all changed, so their independent contributions are
  not isolated.
- Disjoint WikiText test PPL: Q1 30.64, Q2 29.92; matched prompt hashes and
  token budget check clean (ratio 1.0012).
- Thinking subset (10 questions, 1,536 cap): mostly truncated/incomplete;
  reported for completion/parser verification, not accuracy.

Per the parent reviewer's sequencing, seed 43, W3, corrected 1.7B QAD and the
final report rewrite are **paused** until Checkpoint A is verified.

## What remains

- Reviewer verification of Checkpoint A / parser v2 patch (`reports/checkpoint_a_patch.md`).
- Then seed 43 matched pair, W3 matched pair, corrected 1.7B baseline/Q1,
  a larger thinking benchmark, and the final report rewrite.
