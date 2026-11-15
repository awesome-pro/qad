# Checkpoint A patch — parser v2, regrade, checkpoint verification

This patch responds to the parent review of commit
`525ea3dd8b52bc46e08a0226d9640fc9ef57c42d`. The seed-42 model pair was **not**
retrained; the existing generations and weights were reused.

## 1. Strict parser v2

`src/qwen_qad/answer_parsing.py` now:

- considers only the last content line of the answer region (ignoring trailing
  LaTeX display delimiters such as `$$`);
- rejects conflicting explicit markers, including a valid marker plus a
  malformed one, instead of accepting the first valid value;
- scans the whole answer region for conflicting `####`/`Answer:` attempts;
- uses `####[ \t]*` so the marker cannot consume a newline and accept the next
  line's number;
- accepts a boxed value only when it is the last numeric token on the final
  line; `\boxed{12} + 3 = 15` is rejected, while `$$ \boxed{125} $$`,
  `3 + 4 = \boxed{7} \text{ DVDs}`, and `**Answer:** $ \boxed{5} $` are accepted;
- ignores intermediate boxed values on earlier lines.

Regression tests cover each case (`tests/test_answer_parsing.py`).

## 2. Offline regrade (no retraining)

`scripts/regrade_gsm8k.py` re-parses each saved `parse_text`, updates the
per-example parse fields, and writes new artifacts under
`experiments/gate8/regraded/`. The original files are untouched.

### Strict non-thinking GSM8K, 100 questions

| Variant | Accuracy (95% CI) | Marker present | Parsed answer | Complete | Truncated | Parse status |
|---|---:|---:|---:|---:|---:|---|
| T0 BF16 | 37% (28–46) | 60% | 57% | 76% | 24% | ok 57, missing 40, malformed 3 |
| Q0 W4-G128 | 13% (7–20) | 44% | 39% | 50% | 50% | ok 39, missing 56, malformed 2, ambiguous 3 |
| Q1-200 matched | 33% (24–42) | 63% | 62% | 64% | 36% | ok 62, missing 37, malformed 1 |
| Q2-200 matched | 31% (22–41) | 70% | 63% | 72% | 28% | ok 63, missing 30, ambiguous 5, malformed 2 |

Paired Q1 vs Q2: **−2 points**, 95% CI [−9, +5], exact McNemar p = 0.77
(q1-only 7, q2-only 5, both 26, neither 62). No evidence of a difference.

Changes from the v1 grader:

| Variant | v1 | v2 | Changed examples |
|---|---:|---:|---:|
| T0 | 38% | 37% | 4 |
| Q0 | 13% | 13% | 8 |
| Q1 | 34% | 33% | 3 |
| Q2 | 36% | 31% | 8 |

The Q2 change is mostly conflicting-marker rejection (`ambiguous` 5) and
stricter final-line handling. This is the conservative outcome the review asked
for: ambiguous traces are not silently resolved into correct answers.

### Thinking subset (unchanged, 10 questions, 1,536 cap)

The deterministic generations were unchanged: T0 10%, Q0 0%, Q1 10%, Q2 0%;
completion is 60%/0%/30%/10% and thinking-closed 60%/0%/40%/10%. This remains a
parser/completion sanity subset, **not** an accuracy benchmark.

## 3. Metric terminology

The evaluator and regrade artifacts now report separately:

- **marker-present rate** — any explicit `####`/`\boxed{}`/`Answer:` marker was
  found, whether parseable or conflicting;
- **parsed-answer rate** — marker parsed to a single numeric value (`ok`);
- **completion rate** — model emitted EOS before the cap;
- **truncation rate** — generation hit the cap;
- **accuracy** — parsed value equals gold.

`explicit_answer_rate` (which conflated ambiguous/malformed with parsed) is
removed from new artifacts.

## 4. Causal wording

The historical 28% vs 14% result is **not reproduced** under the corrected
protocol. However, the causes were not isolated: parser strictness, decoding
mode/cap, checkpoint selection (best vs final), and training matching all
changed. This patch does not claim that any single factor explains the old gap.

## 5. Checkpoint verification

`scripts/verify_checkpoints.py` loaded each `.pt` through
`load_checkpoint_verified` (model revision, bits, group size, scheme), loaded
the portable `.safetensors` into a fresh model, compared logits and held-out
metrics:

| Arm | Checkpoint SHA-256 | Safetensors SHA-256 | Max logit diff | Argmax match | NLL .pt / safetensors |
|---|---|---|---|---:|---:|
| Q1 best | `3fc424d2…eaf1930` | `a907fb93…24d5f65` | 0.0 | 1.0 | 3.295691 / 3.295691 |
| Q2 best | `23c15270…7fd49a6` | `c7b6a27a…a78c3a3` | 0.0 | 1.0 | 3.294007 / 3.294007 |

KL differences are also 0.0. Both verification JSONs are
`experiments/gate8/verify_q{1,2}_s42.json` with `passed: true`.

Full-test evaluation was re-run through the verified loader and records the
checkpoint SHA-256 per artifact:
`experiments/gate8/test_m_q1_w4_s42_verified.json` (30.637 PPL, SHA `3fc424d2…`)
and `test_m_q2_w4_s42_verified.json` (29.923 PPL, SHA `23c15270…`).

## 6. Backup status

Remote SHA-256 manifest: `checkpoints/gate8/SHA256SUMS.remote.txt`.

Selected best optimizer checkpoints and best/final inference safetensors are
being copied to `checkpoints/gate8/` (gitignored, outside ephemeral pod storage).
The copy is bandwidth-limited (~7 MB/s). Final local hashes and clean-process
restoration are recorded in `checkpoints/gate8/BACKUP_VERIFICATION.md` once the
copy completes; no pod files are deleted before that verification.

## 7. Runbook and tests

- `GPU_RUNBOOK_AUDIT.md` no longer says “no pod is running” or that all
  checkpoints are missing; it records Checkpoint A status and uses valid CLI
  flags (`--eval`, not `--evals`).
- All tests: **160 collected, 158 pass, 2 environment skips** (CUDA-only and
  tokenizer-cache-only).
