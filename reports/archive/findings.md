# Findings - Qwen3-0.6B W4A16-G128 QAD pilot

**Status:** historical pilot report, 2026-11-15. Superseded in part by the audit
corrections in `review.md` / `plan.md` §15. Not publication-ready.

> **Audit notice.** All GSM8K numbers in this document were produced with the
> legacy permissive parser and a 512-token thinking cap. They are **historical
> and exploratory**, not strict reasoning accuracy: an unfinished reasoning
> trace could be graded from its last number. The corrected protocol uses
> `scripts/eval_gsm8k_strict.py` (only explicit final answers; non-thinking
> first, thinking subset with completion/truncation tracked) and
> `scripts/train_matched.py` for matched Q1/Q2 schedules. See
> `GPU_RUNBOOK_AUDIT.md` for the rerun plan. The first corrected results are in
> `reports/checkpoint_a.md` and `reports/checkpoint_a_patch.md` (parser v2 strict
> non-thinking: T0 37%, Q0 13%, matched Q1 33%, matched Q2 31%; paired difference
> not significant).

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
| Q2-200 | 200 | student rollouts | 818,374 | 128 |

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
| Q1-200 (offline) | 3.2688 | 26.28 | **0.2532** |
| Q2-200 (on-policy) | **3.2676** | **26.25** | 0.2646 |

Q1-200 closes about half of the RTN NLL gap (0.318 → 0.152 recovered). At the
matched 25-step budget, offline is slightly better than on-policy on both NLL
(3.2942 vs 3.3021) and KL (0.3009 vs 0.3124). At 200 steps and ~819K response
tokens, the two are effectively tied on NLL/PPL (difference 0.001–0.003), with
offline slightly better on teacher KL.

**Second-seed replication (seed 43, 25-step matched arms):**

| Arm | NLL seed 42 | NLL seed 43 | PPL 42/43 | KL 42/43 | GSM8K 42/43 |
|---|---:|---:|---:|---:|---:|
| Q1-128-m25 | 3.2942 | 3.2938 | 26.96 / 26.95 | 0.3009 / 0.3015 | 13% / 16% |
| Q2-m25 | 3.3021 | 3.2989 | 27.17 / 27.08 | 0.3124 / 0.3083 | 16% / 18% |

The NLL/KL ordering replicates (offline slightly better); the GSM8K ordering
also replicates (on-policy +2–3 points). Both differences remain small and
single-seed-per-arm, but they are consistent across seeds.

**Second-seed replication (200-step arms, ~819K response tokens):**

| Arm | NLL 42 / 43 | PPL 42 / 43 | KL 42 / 43 | GSM8K 42 / 43 |
|---|---:|---:|---:|---:|
| Q1-200 | 3.2688 / 3.2634 | 26.28 / 26.14 | 0.2532 / 0.2534 | 28% / 28% |
| Q2-200 | 3.2676 / 3.2645 | 26.25 / 26.17 | 0.2646 / 0.2640 | 14% / 22% |

At 200 steps Q1 and Q2 are tied on NLL/PPL in both seeds; Q1 is better on KL in
both. Q1's GSM8K is 28% in both seeds; Q2 is unstable (14% / 22%) and below Q1
in both. There is no evidence that on-policy helps here.

## 4. Generative quality

### GSM8K (legacy parser; exploratory only, do not cite as strict accuracy)

100 test examples, thinking mode, 512-token cap, last-number fallback allowed.

| Variant | Accuracy | Parsed | Mean response tokens |
|---|---:|---:|---:|
| T0 BF16 | **33%** | 98% | 509 |
| Q0 W4A16-G128 | 1% | 96% | 512 |
| Q1-128-m25 | 13% | 99% | 512 |
| Q2-m25 | **16%** | 100% | 512 |
| Q1-200 | **28%** | 98% | 506 |
| Q2-200 | 14% | 97% | 506 |

QAD recovers most of the generative damage for offline Q1 (Q0 1% → Q1-200 28%,
T0 33%). At the matched 25-step budget, on-policy is 3 points better than
offline (16% vs 13%), the opposite ordering from NLL/KL. But at 200 steps this
reverses sharply: Q1-200 28% vs Q2-200 14%, despite nearly identical held-out
NLL/PPL. With 100 examples the standard error is about ±4–5 points, so the
25-step difference is suggestive; the 200-step gap is large enough to matter.
Q2's 200-step result is *worse* than its 25-step result (16% → 14%).

### Fixed-prompt repetition, 8 prompts, 256 tokens

| Variant | 4-gram repetition ↓ | distinct-2 ↑ | Hit cap |
|---|---:|---:|---:|
| T0 BF16 | 0.047 | 0.801 | 8/8 |
| Q0 W4A16-G128 | 0.117 | 0.697 | 8/8 |
| Q1-200 | 0.067 | 0.780 | 8/8 |
| Q2-m25 | 0.053 | 0.778 | 8/8 |
| Q2-200 | **0.050** | **0.789** | 8/8 |

Both QAD arms move generation back toward BF16; Q2-200 is the least repetitive
of the quantized students even though it is the worst on GSM8K. Repetition
metrics and task accuracy dissociate here. No config emitted EOS within
256 tokens, so premature-EOS is uninformative at this cap.

### Rollout-length ablation (25 steps, ~102,400 response tokens)

| Arm | 128-token rollouts (PPL / KL / GSM8K) | 256-token rollouts (PPL / KL / GSM8K) |
|---|---|---|
| Q1 (gold prefixes) | 26.96 / 0.3009 / 13% | 26.81 / 0.2895 / 15% |
| Q2 (student rollouts) | 27.17 / 0.3124 / 16% | 27.11 / 0.3032 / 18% |

Doubling the rollout length improves both arms slightly and does not change the
ordering: Q1 still wins NLL/KL, Q2 still wins GSM8K by 3 points. The 200-step
Q2 GSM8K collapse (14%) is therefore not explained by the 128-token cap alone;
training duration on short self-generated rollouts matters.

### Exploratory 1.7B (A6000 48 GB, single seed)

Same data and W4A16-G128 recipe; Q1-200 uses `max_seq_len=1536` (responses
truncated to fit), Q2 is a 25-step/102,400-token pipeline check.

| Variant | NLL | PPL | KL | GSM8K |
|---|---:|---:|---:|---:|
| T0 BF16 | 2.8327 | 16.99 | 0.0000 | 36% |
| Q0 W4-G128 | 2.9935 | 19.96 | 0.2516 | 16% |
| Q1-200 | 2.8812 | 17.84 | 0.1839 | 27% |
| Q2-25 | 2.8947 | 18.08 | 0.2091 | 19% |

- W4 RTN costs 2.97 PPL (17.5%); Q1-200 recovers 71% of that gap and 55% of the
  GSM8K gap (16% → 27%, T0 36%).
- Q2-25 (14% of Q1's tokens) improves over Q0 but trails Q1, consistent with
  the 0.6B ordering: longer offline recovery is stronger; short on-policy does
  not beat it.
- Training: Q1 572 s, 725,916 response tokens, 37.7 GiB peak, 17.5 GB
  checkpoint; Q2-25 102,400 tokens, 36.2 GiB. All assertions passed.

### W3A16-G128 QAD probe (0.6B, offline)

The earlier diagnosis treated W3-G128 RTN as unrecoverable. A short QAD probe
shows that likelihood recovers strongly, but generation does not:

| Variant | NLL | PPL | KL | GSM8K |
|---|---:|---:|---:|---:|
| W3 Q0 RTN | 6.5848 | 723.98 | 3.8674 | 1% |
| W3 Q1-25 (128-token gold) | 4.4256 | 83.56 | 1.5954 | 3% (parse 83%) |
| W3 Q1-200 (1,024-token gold) | 4.0834 | 59.35 | 1.2008 | 0% (parse 25%) |

- QAD cuts W3 PPL by an order of magnitude (724 → 59) but plateaus far above
  W4 Q0 (30.6), and the 200-step student cannot produce parseable GSM8K answers
  (25% parse, 0% correct).
- W3-G128 is therefore **partially recoverable in likelihood but not a viable
  generative initialization** with this recipe. W4 remains the tractable arm;
  a stronger W3 init would need a separate decision.

## 5. Cost, memory, artifacts

| Run | GPU time | Response tokens | Peak GPU | Checkpoint |
|---|---:|---:|---:|---:|
| Q1-128-m25 | 164 s | 102,400 | 11.3 GiB | 5.6 GB |
| Q2-m25 | 308 s | 101,499 | 14.0 GiB | 5.6 GB |
| Q1-200 | 313 s | 818,760 | 14.6 GiB | 5.3 GB |
| Q2-200 | 2,382 s | 818,374 | 14.0 GiB | 5.6 GB |
| Q1-256-m25 | 95 s | 102,400 | 11.9 GiB | 5.6 GB |
| Q2-len256-m25 | 328 s | 102,384 | 13.6 GiB | 5.6 GB |
| W3 Q1-200 | 312 s | 818,760 | 14.6 GiB | 5.6 GB |

Checkpoints stayed on the pod and were deleted after evaluation; they are not
durable. The 24 GB card is sufficient for 0.6B QAD at 2,048-token sequences with
microbatches of one.

Theoretical whole-model storage (not measured packed artifacts): see
`reports/memory_accounting.md`. W4A16-G128 is ~0.50 GiB for 0.6B and ~1.26 GiB
for 1.7B versus ~1.11/3.20 GiB BF16; these numbers exclude packing overhead and
imply no serving speedup.

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
- At the 200-step, ~819K-token budget, offline and on-policy are tied on
  held-out NLL/PPL in both seeds; offline is better on KL in both. GSM8K:
  Q1 28%/28%, Q2 14%/22% - on-policy is below offline in both seeds and
  unstable. There is no evidence that on-policy helps at this scale.
- Q2-200 is the cleanest generation of the quantized arms (4-gram repetition
  0.050 vs 0.067 Q1, 0.117 Q0) while being the worst on GSM8K.
- The 25-step ordering replicates across seeds: Q1 better on NLL/KL, Q2 better
  on GSM8K by 2–3 points.
- Doubling rollouts to 256 tokens at 25 steps raises Q1 to 26.81 PPL / 15% GSM
  and Q2 to 27.11 / 18%; the ordering is unchanged, so the 128-token cap is not
  the whole explanation for Q2's 200-step GSM8K collapse.
- W3A16-G128 QAD recovers likelihood by an order of magnitude (PPL 724 → 59
  after 200 steps) but remains unusable generatively (0% GSM8K, 25% parse).
  W4 is still the only viable generative initialization in this pilot.
- At 1.7B, W4 RTN is less damaging (17.5% PPL vs 37% at 0.6B), Q1-200 recovers
  most of it, and short Q2 again trails Q1.

**Hypothesis (not established):**
- The 25-step on-policy GSM8K edge did not persist at 200 steps: short
  self-generated rollouts may train behaviour that helps repetition and
  plain-text likelihood but hurts a 512-token thinking benchmark. The 256-token
  ablation helps slightly but does not reverse the ordering; longer rollouts
  combined with longer training remain untested.
- W3-G128 likelihood recovery plateaus around PPL 59 while generation collapses;
  the bottleneck may be representational capacity, not the recovery objective.
- A stronger initialization (GPTQ/AWQ, block-wise QAT) could close more of the
  W3 gap; outside this pilot's scope.

## 7. Limitations

- One quantization scheme (W4-G128) for the main study; two seeds for the
  0.6B arms, single seed for 1.7B and W3.
- The historical GSM8K numbers used a permissive parser and 512-token cap; they
  are not strict reasoning accuracy and are being replaced (review.md P0).
- Old checkpoints were deleted after evaluation; corrected re-evaluation
  requires retraining (documented in GPU_RUNBOOK_AUDIT.md).
- Q2 rollouts are 128 tokens for the 200-step runs (256 only in the 25-step
  ablation); Q1-200 used 1,024-token gold completions, so the 200-step arms are
  token-matched but not length-matched.
- 1.7B Q1 used `max_seq_len=1536` (725,916 response tokens) and Q2 only 25
  steps/102,400 tokens; it is an exploratory arm, not matched to 0.6B.
- Gold completions were truncated at 1,024 tokens (2,979/3,000); Q1 learns from
  prefixes, not full traces.
- No MMLU/MATH/IFEval, no packed INT4 export, no vLLM/real speed measurement.
  Fake quantization gives no serving speedup and must not be presented as such.
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

1. Run Q2-200 with 256–512-token rollouts (longer training, not just the
   25-step ablation) to test whether the GSM8K collapse is caused by rollout
   length or by prolonged training on self-generated data.
2. Decide whether a stronger W3 initialization (GPTQ/AWQ, block-wise QAT) is
   worth pursuing; W3 QAD recovers likelihood but not generation, so the
   deployment case for W3-G128 remains weak.
3. If more budget: 1.7B Q2 matched budget, then a real packed INT4 export and
   one objective benchmark beyond GSM8K (MMLU-Redux or MATH).
