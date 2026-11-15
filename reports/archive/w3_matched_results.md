# W3A16-G128 matched QAD - 0.6B

Protocol: matched Q1, 200 steps, 32 prompts/step, 128-token continuation cap,
same schedule/teacher/seed 42/validation split/checkpoint selection as W4.
First attempt hit the pod's 50 GB disk limit during checkpoint save; after
freeing verified seed-42 pod copies, the retry completed. Only the completed
retry is reported.

## Results

| Metric | W3 Q0 (historical) | W3 matched Q1-200 | W4 Q1-200 (for scale) |
|---|---:|---:|---:|
| Validation PPL (best) | 723.98 | **62.42** (step 150) | 26.56 |
| WikiText test PPL | - | 88.24 | 30.68 |
| Test KL(T‖S) | - | 1.443 | 0.307 |
| Strict GSM8K (non-thinking) | ~0–1% | **0%** (0/100) | 35% |
| Completion / truncation | - | 1% / 99% | 64% / 36% |
| Marker present / parsed | - | 4% / 0% | 64% / 61% |

- Best validation checkpoint: step 150, NLL 4.1338; final step 200 was worse.
- The student emits almost no final-answer markers; 99% hit the 256-token cap
  without EOS. The likelihood recovery did not translate into usable generation.

## Interpretation

W3-G128 QAD recovers held-out likelihood by an order of magnitude but remains
far above W4 in both validation (62.4 vs 26.6 PPL) and test (88.2 vs 30.7 PPL),
and its non-thinking GSM8K generation is unusable at this recipe. This is
consistent with the earlier W3 probe (59.4 PPL, 0% GSM) and strengthens the
conclusion: **W3 is not a viable generative arm without a stronger
initialization**, despite likelihood recovery.

## Artifacts

- `experiments/gate8/m_q1_w3_s42.json`, `_steps.jsonl`
- `experiments/gate8/test_m_q1_w3_s42_verified.json`
- `experiments/gate8/regraded_v3/gsm8k_m_q1_w3_s42_nonthink.*`
- `experiments/gate8/regraded_v3/integrity_report_w3.json` - **passed**
- Best checkpoint SHA-256 `b9c3827a9533…`; safetensors backup in
  `checkpoints/gate8_w3/` (gitignored)
