# Findings — Qwen3-0.6B W4A16-G128 QAD pilot

**Status:** first pilot result, single seed, 2026-11-15. Not a publication-ready
claim. W4A16-G128 only; W3A16-G128 remains un-recovered and is the aggressive
baseline.

## 1. Question

Can teacher distillation recover a Qwen3-0.6B student whose linear weights are
fake-quantized to W4A16-G128 by pure RTN, and does collecting the distillation
data **on-policy** (student-generated rollouts) beat **offline** distillation on
gold completions at a matched response-token budget?

## 2. Setup

| Item | Value |
|---|---|
| Model / tokenizer | `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca` |
| Quantizer | symmetric signed W4, group 128, per output row, fp32 masters + STE |
| Trainable | 440,401,920 wrapped linear masters; embeddings/norms/LM head frozen |
| Teacher | same checkpoint, BF16, frozen, two-pass self-KL 0.0 |
| Held-out eval | FineWeb-Edu 256 × 1,024 tokens (261,888 scored positions), hash `8676a899…` |
| Offline data | OpenThoughts prompts + gold completions (2,979/3,000 truncated at 1,024 tokens; EOS only on complete ones) |
| Hardware | RunPod RTX A5000 24 GB, driver 595.91.07, torch 2.13.0+cu129, transformers 5.23.0 |
| Optimizer | AdamW, lr 1e-5, 20-step warmup (5 for 25-step arms), cosine to 10%, grad clip 1.0 |
| Loss | forward KL(teacher‖student) on response tokens only, no CE term, τ=1.0 |

Training arms:

| Arm | Steps | Data | Response tokens | Seq/rollout length |
|---|---:|---|---:|---|
| Q1-200 | 200 | gold completions | 818,760 | 1,024 |
| Q1-128-m25 | 25 | gold completions (capped) | 102,400 | 128 |
| Q2-m25 | 25 | student rollouts | 101,499 | 128 |
| Q2-200 | 200 | student rollouts | *in progress* | 128 |

Q2 generation: temperature 0.6, top_p 0.95, top_k 20, 32 prompts/step, max 128
new tokens, fresh rollouts every step. Q1/Q2 25-step arms are matched at
~102K response tokens and 128-token responses; this is the clean comparison.

## 3. Held-out results (full 256-block set)

| Variant | NLL ↓ | PPL ↓ | KL(teacher‖student) ↓ |
|---|---:|---:|---:|
| T0 BF16 | 3.1027 | 22.26 | 0.0000 |
| Q0 W4A16-G128 (RTN) | 3.4207 | 30.59 | 0.4057 |
| Q1-128-m25 (offline) | 3.2942 | 26.96 | 0.3009 |
| Q2-m25 (on-policy) | 3.3021 | 27.17 | 0.3124 |
| Q1-200 (offline) | **3.2688** | **26.28** | **0.2532** |

Q1-200 closes about half of the RTN NLL gap (0.318 → 0.152 recovered). At the
matched 25-step budget, offline is slightly better than on-policy on both NLL
(3.2942 vs 3.3021) and KL (0.3009 vs 0.3124); the difference is small and
single-seed.

## 4. Generative quality

### GSM8K, 100 test examples, thinking mode, 512-token cap

| Variant | Accuracy | Parsed | Mean response tokens |
|---|---:|---:|---:|
| T0 BF16 | **33%** | 98% | 509 |
| Q0 W4A16-G128 | 1% | 96% | 512 |
| Q1-128-m25 | 13% | 99% | 512 |
| Q2-m25 | **16%** | 100% | 512 |
| Q1-200 | **28%** | 98% | 506 |

QAD recovers most of the generative damage (Q0 1% → Q1-200 28%, T0 33%). At the
matched 25-step budget, on-policy is 3 points better than offline (16% vs 13%),
the opposite ordering from NLL/KL. With 100 examples the standard error is about
±4–5 points, so this is suggestive, not conclusive.

### Fixed-prompt repetition, 8 prompts, 256 tokens

| Variant | 4-gram repetition ↓ | distinct-2 ↑ | Hit cap |
|---|---:|---:|---:|
| T0 BF16 | 0.047 | 0.801 | 8/8 |
| Q0 W4A16-G128 | 0.117 | 0.697 | 8/8 |
| Q1-200 | 0.067 | 0.780 | 8/8 |
| Q2-m25 | 0.053 | 0.778 | 8/8 |

Both QAD arms move generation back toward BF16. No config emitted EOS within
256 tokens, so premature-EOS is uninformative at this cap.

## 5. Cost, memory, artifacts

| Run | GPU time | Response tokens | Peak GPU | Checkpoint |
|---|---:|---:|---:|---:|
| Q1-128-m25 | 164 s | 102,400 | 11.3 GiB | 5.6 GB |
| Q2-m25 | 308 s | 101,499 | 14.0 GiB | 5.6 GB |
| Q1-200 | 313 s | 818,760 | 14.6 GiB | 5.3 GB |
| Q2-200 | *in progress* | ~819K | — | — |

Checkpoints stay on the pod and are not committed. The 24 GB card is sufficient
for 0.6B QAD at 2,048-token sequences with microbatches of one.

## 6. What is established vs hypothesis

**Established (measured):**
- W4A16-G128 RTN costs 8.3 PPL (22.26 → 30.59) on the held-out set.
- Forward-KL QAD recovers a large part of it: 26.28 PPL and 28% GSM8K after 200
  offline steps, against 33% for BF16 and 1% for the RTN student.
- The training plumbing is correct: teacher gradients absent, only masters
  trainable, gradients finite, quantization active, checkpoints reload, rollouts
  refresh every step.
- At a matched ~102K-token, 128-token-response budget, offline beats on-policy
  on NLL/KL but loses 3 GSM8K points; neither difference is decisive.

**Hypothesis (not established):**
- On-policy data may help generative behavior more than likelihood at this
  scale, but this needs multiple seeds and more examples.
- W3A16-G128 RTN is not recoverable by the same recipe; no QAD run was made at
  W3 in this pilot.
- A stronger initialization (GPTQ/AWQ, block-wise QAT) would close more of the
  gap; outside this pilot's scope.

## 7. Limitations

- Single seed, one model size (0.6B), one quantization scheme (W4-G128).
- GSM8K subset is 100 examples with a 512-token thinking cap; ±4–5 point noise.
- Q2 rollouts are 128 tokens because HF generation throughput on the A5000 was
  the binding constraint; Q1-200 used 1,024-token gold completions, so the
  200-step arms are token-matched but not length-matched.
- Gold completions were truncated at 1,024 tokens (2,979/3,000); Q1 learns from
  prefixes, not full traces.
- No MMLU/MATH/IFEval, no 1.7B run, no packed INT4 export, no vLLM/real speed
  measurement. Fake quantization gives no serving speedup and must not be
  presented as such.
- Checkpoints are not retained in the repository.

## 8. Reproduction

```bash
# baseline (four configs)
python scripts/baseline.py --variant T0 ... --out experiments/gate2/t0_bf16.json
python scripts/baseline.py --variant Q0 --bits 4 --group-size 128 ... \
  --out experiments/gate2/q0_w4a16_g128.json

# offline Q1
python scripts/train_offline.py --device cuda --bits 4 --group-size 128 \
  --steps 200 --response-tokens-per-step 4096 --max-seq-len 2048 \
  --run-name q1_w4a16_g128_200 --out-dir experiments/gate3

# on-policy Q2
python scripts/train_online.py --device cuda --bits 4 --group-size 128 \
  --steps 200 --prompts-per-step 32 --max-new-tokens 128 \
  --generation-batch-size 32 --run-name q2_w4a16_g128_200 \
  --out-dir experiments/gate4

# checkpoint evaluation and GSM8K
python scripts/baseline.py --variant Q0 --bits 4 --group-size 128 \
  --checkpoint checkpoints/q1_w4a16_g128_200_step200.pt \
  --out experiments/gate5/eval_q1_200.json
python scripts/eval_gsm8k.py --config-name Q1_200 --bits 4 \
  --checkpoint checkpoints/q1_w4a16_g128_200_step200.pt \
  --out experiments/gate5/gsm8k_q1_200.json
```

Artifacts: `experiments/gate2` (baseline), `gate3` (offline), `gate4`
(on-policy), `gate5` (checkpoint eval + GSM8K + generation).

## 9. Follow-ups

1. Finish Q2-200; if it does not beat Q1-200, report on-policy as neutral to
   negative at this scale.
2. Add a second seed for Q1 and Q2 at the 25-step matched budget to size the
   GSM8K difference.
3. If more budget: 1.7B W4, W3 with a stronger initialization, and a real packed
   INT4 export.
