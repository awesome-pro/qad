# Checkpoint B — matched seed 43

Protocol identical to seed 42: 200 steps, 32 prompts/step, 128-token continuation
cap, same schedule seed/teacher/quantizer, validation-selected best checkpoint,
disjoint WikiText test, frozen parser v3.0.0.

## Matched check

`check_matched_s43.json`: **matched: true**, zero prompt-hash mismatches,
6,400 prompt instances, 3,000 unique prompts, token ratio 1.00138.

## Results

| Arm | Best step (val PPL) | WikiText test PPL | Strict GSM8K | 95% CI |
|---|---:|---:|---:|---:|
| Q1-200 | 150 (26.56) | **30.68** | 35% | 25–44% |
| Q2-200 | 200 (26.64) | 30.72 | **43%** | 33–52% |

Paired Q1 vs Q2: **+8 points**, 95% CI [−2, +19], exact McNemar p = 0.169
(q1-only 9, q2-only 17, both 26, neither 48). This is a directional on-policy
advantage but not statistically significant.

Seed 42 was +2 points (Q1 33, Q2 35; p = 0.77). Across the two seeds, Q2 is
+2 and +8 points on GSM8K, while test NLL is essentially tied (seed 42 Q2
slightly better; seed 43 Q1 slightly better). The evidence does **not** support
a reliable on-policy advantage; it does suggest the effect is seed-dependent.

## Integrity

`regraded_v3` Q1/Q2 summaries and provenance:
`integrity_report_s43.json` → **passed** for both summaries and the paired
comparison (parser 3.0.0, source hashes, records, provenance and checkpoint
SHA all consistent).

Raw evaluation summaries are preserved; canonical analysis uses the v3
regrades (which had zero changed examples versus the evaluator output).

## Artifacts

- `experiments/gate8/m_q1_w4_s43.json`, `m_q2_w4_s43.json`
- `experiments/gate8/match_check_s43.json`
- `experiments/gate8/regraded_v3/gsm8k_m_q*_w4_s43_nonthink.*`
- `experiments/gate8/regraded_v3/paired_integrity_s43.json`
- `experiments/gate8/regraded_v3/integrity_report_s43.json`
