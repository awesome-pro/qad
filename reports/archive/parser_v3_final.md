# Parser v3.0.0 - frozen GSM8K evaluation and final seed-42 regrade

Status: **frozen**. Spec: `docs/parser_spec_v3.md`. Seed-42 models and
generations were **not** regenerated or retrained.

## 1. Final parser changes (v2 → v3)

- Nested markers after `####` are one answer:
  `#### $\boxed{8}$` → 8 (Q2 examples 23, 59, 80, 99).
- Nested markers after `Answer:` are one answer:
  `**Answer:** #### 225` → 225; `**Answer:** $\boxed{10}$ games.` → 10.
- Markdown answers supported: `**Answer: $694**` → 694,
  `**Answer: $233.00**` → 233.
- One leading number followed only by non-numeric units is accepted
  (`14 bags.` → 14, `120 minutes` → 120); extra digits are rejected
  (`8 and 9`, `12 + 3`).
- The `Answer:` regex uses `[^\n]`/`[ \t]`, so it cannot consume a following
  line.
- Contradictory answers, malformed+valid mixes, incomplete thinking, and
  intermediate boxed arithmetic remain rejected.

## 2. Tests

**172 collected, 170 passed, 2 environment skips** (CUDA-only quantizer test;
pinned-tokenizer test when not in the local cache). New regression tests cover
all of the above.

## 3. Final seed-42 regrade (parser v3.0.0)

| Variant | Accuracy (95% CI) | Marker present | Terminal marker | Parsed | Complete | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| T0 BF16 | 39% (29–49) | 66% | 62% | 61% | 76% | 24% |
| Q0 W4-G128 | 17% (10–25) | 54% | 49% | 46% | 50% | 50% |
| Q1-200 matched | 33% (24–42) | 66% | 63% | 62% | 64% | 36% |
| Q2-200 matched | 35% (26–45) | 73% | 72% | 70% | 72% | 28% |

Paired Q1 vs Q2 (same 100 questions): **+2 points**, 95% CI [−4, +9], exact
McNemar p = 0.77 (q1-only 5, q2-only 7, both 28, neither 60). No evidence of a
difference.

### Change from v2

| Variant | v2 accuracy | v3 accuracy | Net |
|---|---:|---:|---:|
| T0 | 37% | 39% | +2 |
| Q0 | 13% | 17% | +4 |
| Q1 | 33% | 33% | 0 |
| Q2 | 31% | 35% | +4 |

Q2 changed 7 examples: 4 correct newly parsed (694, 8, 187, 58), 0 correct lost,
and 3 wrong values newly parsed (233/243, 700/100, 14/10). Q1 changed 0
examples relative to v2. T0/Q0 changed 4/8 examples, mostly Markdown/units.

This is not a rule designed to maximize accuracy: three of the seven newly
parsed Q2 answers are wrong, and contradictory/intermediate answers remain
rejected.

## 4. Per-example diffs (preserved)

- v2 artifacts: `experiments/gate8/regraded/`
- v3 artifacts: `experiments/gate8/regraded_v3/`
- Q2 v2→v3 diff: `experiments/gate8/regraded_v3/diff_gsm8k_m_q2_w4_s42_nonthink.json`
- Q1 v2→v3 diff: `experiments/gate8/regraded_v3/diff_gsm8k_m_q1_w4_s42_nonthink.json`
- T0/Q0 diffs: `regraded_v3/diff_gsm8k_{t0,q0_w4}_nonthink.json`
- Previous v1/v2 summaries and generation JSONLs are untouched.

## 5. Consistency audit and provenance regeneration

The first v3 provenance files were generated from an **intermediate** v3 pass,
before the nested-`Answer:` and units fixes. The canonical v3 summaries were
then regenerated, but the provenance files were not, producing the reviewer's
mismatch (Q1 33/62 vs 32/61; Q2 35/70 vs 35/69).

Actions:

- Archived the stale files under
  `experiments/gate8/regraded_v3/archive_stale_provenance/`.
- Regenerated all four Q1/Q2 provenance summaries from the canonical v3
  generation records; Q1 is now 33 correct/62 parsed, Q2 35 correct/70 parsed.
- Added `src/qwen_qad/eval_integrity.py` and
  `scripts/check_eval_integrity.py`. The check recomputes metrics from the
  per-example records and verifies:
  - summary metrics, status counts and correct flags against records;
  - parser version 3.0.0 and source hash against `regraded_from`;
  - provenance metrics/parser/source/checkpoint SHA against canonical + run;
  - paired accuracy, counts, McNemar p, parser versions and source hashes.
- Manifest: `experiments/gate8/regraded_v3/integrity_manifest.json`.
- Report: `experiments/gate8/regraded_v3/integrity_report.json` - **passed**
  for all six seed-42 summaries/provenance files and the paired comparison.
  The regenerated paired output is `paired_integrity.json`; the old
  `gsm8k_m_q1_vs_q2_s42_nonthink.json` is preserved.

No seed-42 retraining and no parser-semantic changes were made for this fix.

## 6. Checkpoint provenance

The regraded Q1/Q2 summaries have checkpoint provenance attached:
Q1 `.pt` `3fc424d2…eaf1930`, Q2 `.pt` `23c15270…7fd49a6`; best safetensors are
in `checkpoints/gate8/` and verified byte-for-byte. Clean-process tensor
restoration max diff 0.0; on-pod logits/held-out metrics identical. See
`experiments/gate8/BACKUP_VERIFICATION.md`.

## 7. Seed 43

Seed-43 matched training is proceeding independently with the approved seed-42
protocol (same 32 prompts/step, 128-token cap, 200 steps, schedule seed 43,
teacher/quantizer). Evaluations will use frozen parser v3.0.0 and the disjoint
validation/test sets; configuration equality will be checked with
`check_matched.py`.

## 8. Remaining limitations

- One corrected seed per arm until seed 43 completes; 100 non-thinking
  questions; bootstrap/McNemar uncertainty remains wide.
- Thinking-mode evaluation is still only a 10-question completion/parser sanity
  subset, not an accuracy benchmark.
- WikiText-103 test is disjoint from validation but a different domain; public
  benchmark contamination is possible.
- Old 28% vs 14% is not reproduced; parser, decoding, checkpoint selection and
  training matching changed together, so their independent contributions are
  not isolated.
