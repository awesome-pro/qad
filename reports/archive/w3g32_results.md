# W3A16-G32 matched QAD — stronger granularity check

Question: does reducing W3 group size from 128 to 32 close the W3 generation
gap? Same matched protocol as W4/W3-G128: 200 steps, 32 prompts/step,
128-token cap, seed 42, validation-selected best checkpoint, WikiText test,
frozen parser v3.

## Results

| Run | Val PPL (best) | Test PPL | Test KL | Strict GSM8K |
|---|---:|---:|---:|---:|
| W3-G32 Q0 RTN | 97.73 | 112.49 | 1.762 | — |
| W3-G32 Q1-200 | **40.78** (step 150) | **51.43** | 0.904 | **0%** (0/100) |
| W3-G128 Q1-200 | 62.42 | 88.24 | 1.443 | 0% (0/100) |
| W4 Q1-200 | 26.56 | 30.68 | 0.307 | 35% |

W3-G32 changes:

- Validation PPL 62.4 → 40.8; test PPL 88.2 → 51.4. Granularity is a real
  lever, not a no-op.
- Generation still fails: 99% missing markers, 31% completion, 69% truncation,
  0% parsed/correct. Only 1% of outputs contain any explicit marker.
- It remains far from W4 (test 51 vs 31 PPL; GSM 0% vs 35%).

Conclusion: **better granularity substantially improves likelihood but does not
make W3 generatively usable at this recipe.** The next test is a stronger
data-aware initialization (GPTQ/AWQ) or block-wise QAT, not more QAD steps.

## Artifacts

- `experiments/gate9/m_q1_w3g32_s42.json`
- `experiments/gate9/test_m_q1_w3g32_s42_verified.json`
- `experiments/gate9/regraded_v3/gsm8k_m_q1_w3g32_s42_nonthink.*`
- `experiments/gate9/regraded_v3/integrity_report_w3g32.json` — passed
- Best checkpoint SHA-256 `81505d2adf25616b…`; durable backup verified: local=remote hashes, clean-process 310-tensor restoration max diff 0.0 (`experiments/gate9/BACKUP_VERIFICATION_w3g32.md`).
