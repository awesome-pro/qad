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
memory savings. We never claimed such savings, but the new table makes the
distinction explicit; no packed artifact is produced.

## What remains

- Corrected GPU runs: strict non-thinking GSM8K, matched Q1/Q2 (seeds 42/43),
  best/final evaluation on the disjoint test set, thinking subset, W3 probe,
  exploratory 1.7B. Estimated 2–3 GPU-hours total, two sessions.
- Final report rewrite with corrected tables and softened conclusions.
