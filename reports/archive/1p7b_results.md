# 1.7B controlled scaling check

Preconditions before training (A6000 48 GB):

- Tokenizer compatibility with 0.6B: **passed** (vocab, chat template, special
  IDs `<think>`/`</think>`/`<|im_start|>`/`<|im_end|>`, sample encodings).
- 3-step W4 memory/gradient probe: **passed**, finite loss/grads, teacher
  gradients absent, quantized forward active, master weights changed.
- Probe peak CUDA memory **28.6 GB**; full Q1 training fit comfortably.
- No 1.7B online Q2 was run.

## Results

| Variant | Val PPL (best) | WikiText test PPL | Strict GSM8K |
|---|---:|---:|---:|
| T0 BF16 | 17.47 | 19.53 | **58%** |
| Q0 W4-G128 RTN | 20.46 | 25.41 | 44% |
| Q1-200 offline (best step 100) | **18.18** | **19.47** | 42% |

Q1 test PPL is essentially identical to BF16 (19.47 vs 19.53) and far below
the W4 RTN student (25.41). However, GSM8K does **not** improve: Q1 42% vs Q0
44% vs T0 58%; the Q1−Q0 difference is within the 100-example noise (±5 pts).

## Interpretation

- At 1.7B, W4 damage is much smaller than at 0.6B on non-thinking GSM
  (Q0 44% vs T0 58% here; 0.6B Q0 17% vs T0 39%).
- Offline QAD recovers held-out likelihood almost completely, but this does not
  translate into task-accuracy recovery. This is the same likelihood/task
  dissociation seen at 0.6B and W3.
- A controlled scaling claim is limited: one seed, 100 GSM questions,
  1,536-token max sequence, 200 steps/819K response tokens, and a different
  GPU from the 0.6B runs. The result is useful as a scaling check, not a
  scaling law.

## Artifacts

- `experiments/gate8/tokenizer_compat_1p7b_pod.json`
- `experiments/gate8/probe_1p7b_w4.json`
- `experiments/gate8/{val,test}_1p7b_{t0,q0_w4}.json`
- `experiments/gate8/m_q1_w4_1p7b.json`
- `experiments/gate8/test_m_q1_w4_1p7b_verified.json`
- `experiments/gate8/regraded_v3/gsm8k_*1p7b*` and
  `integrity_report_1p7b.json` — **passed**
- Best checkpoint hashes in `checkpoints/gate8_1p7b/` (backup progress/status
  must be verified before any claim of durability).
