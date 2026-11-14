# PLAN — Qwen3 low-bit weight quantization + online QAD

**Status:** draft for external review (parent researcher agent)
**Author:** implementation agent
**Date:** 2026-11-15
**Authoritative source:** `QWEN_LOW_BIT_QAD_RESEARCH_BRIEF.md`. Where this plan and the brief disagree, the brief wins. The published *OnlineQAT* paper (arXiv:2610.09346) is reference material only — this is not a reproduction of it.

Everything below marked **ESTIMATE** is a planning number. Nothing marked *measured* exists yet except the quantizer test results in §6.4.

---

## Summary for the reviewer

- **Question:** does on-policy QAD recover a pure-RTN W3A16 Qwen3 student better than offline QAD, and which factor (prefix source vs loss estimator) matters?
- **Design:** T0 (BF16) / Q0 (RTN) / Q1 (offline forward KL on gold completions) / Q2 (on-policy forward KL) / Q3 (on-policy sampled reverse KL). Q1 vs Q2 isolates the prefix source at matched loss, prompts and budget; Q2 vs Q3 isolates the estimator.
- **Scope:** W3A16 only, no Stage-1 initialization, Qwen3-0.6B first, 1.7B conditional on a pre-registered threshold. Estimated 7–10 GPU-hours for the complete 0.6B answer; the first paid session is ~1 h on a 24 GB card.
- **Status:** quantizer + 29 tests done; everything else is proposed. Nothing is claimed as validated beyond §6.4.
- **Decisions I most want checked:** §13 items 1 (Q1 definition), 2 (loss temperature), 6 (budget matching), 7 (promotion criterion), 10 (whether the resulting claim is strong enough for the trial).
- **Deliberate deviations / judgment calls:** Q1 uses the same prompts with gold completions rather than arbitrary fixed text; greedy decoding for the generative table; sampled RKL included as a third arm even though the brief treats it as optional.

---

## 1. Research question

> Can quantization-aware distillation (QAD) recover a Qwen3 model whose linear
> weights are fake-quantized to W3A16 by pure round-to-nearest (RTN), and does
> collecting the distillation data **on-policy** — on prefixes the current
> quantized student actually visits — beat **offline** distillation on fixed
> prefixes at a matched token/step budget?

Sub-questions:

1. How much quality does W3A16 RTN destroy relative to the BF16 teacher (Q0 vs T0)?
2. How much does offline QAD recover on a fixed held-out set (Q1 vs Q0)?
3. Does switching only the **prefix source** (fixed gold completions → student
   rollouts) while holding loss, prompts, teacher, optimizer and budget fixed
   change the recovery (Q2 vs Q1)?
4. Does switching only the **loss direction/estimator** (forward KL →
   sampled reverse KL) on the same student-visited prefixes help (Q3 vs Q2)?

Question 3 is the scientifically clean version of the brief's core comparison.
The brief notes that published methods change both factors at once; this plan
separates them.

---

## 2. Scope

### 2.1 Locked decisions (from the brief + user)

| Dimension | Decision |
|---|---|
| Pilot model | `Qwen/Qwen3-0.6B`, pinned revision (real causal LM, cheap, correct pipeline) |
| Headline model | `Qwen/Qwen3-1.7B`, only after the 0.6B pipeline is validated |
| Precision | W3A16 weight-only; BF16/FP16 activations; **no** KV-cache quantization |
| Weight format | signed symmetric, codes `[-3, 3]`, one signed code unused |
| Grouping | group size 128 along `in_features`, independent per output row |
| Scales | `max(abs(W_group)) / qmax`, detached, safe floor `1e-8`, no zero-point, not learned |
| Quantized modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` only |
| Not quantized | embeddings, LM head (tied to embeddings in both models), all RMSNorms (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`) |
| Student init | copy pretrained weights, then enable fake quantization (Q0 = RTN). **No block-wise QAT initialization** |
| Teacher | same checkpoint, BF16, `eval()`, all parameters `requires_grad_(False)` |
| Offline loss | forward KL `KL(p_T || p_θ)` at response-token positions |
| On-policy loss | forward KL on student-visited prefixes (Q2) and sampled reverse KL (Q3) |
| Optimizer | AdamW on fp32 master weights only; teacher has no optimizer state |
| Data | OpenThoughts prompts (+ gold completions for Q1) and a FineWeb-Edu sample, pinned |
| Hardware | rented RunPod CUDA GPU, chosen per session; 24 GB for 0.6B, 48 GB if we promote to 1.7B |
| Alternative scheme | asymmetric affine `[0, 2^b-1]` + zero-point is implemented behind an explicit flag, **never selected implicitly**; no paper-comparability claims |

### 2.2 Explicit non-goals

- Reproducing OnlineQAT numbers (different student initialization, single seed, different data/training exactness).
- W2A16 (from RTN it is expected to be unrecoverable; revisit only if W3 shows large headroom).
- Stage-1 block-wise QAT (explicitly excluded by the user; the brief's Q0 is RTN).
- Packed INT3 export, custom kernels, vLLM/SGLang integration, inference speedups.
- Claiming fake-quantized runs save VRAM or latency.
- Public model release, benchmark leaderboard claims, statistical significance claims from one seed.

---

## 3. Scientific design

### 3.1 Variants

| ID | System | Trained? | Prefix source | Loss |
|---|---|---|---|---|
| T0 | BF16 teacher | no | fixed held-out text | — |
| Q0 | W3A16 RTN student | no | fixed held-out text | — |
| Q1 | Q0 + offline QAD | yes | fixed (prompt, gold completion) | forward KL |
| Q2 | Q0 + on-policy QAD | yes | student rollouts | forward KL |
| Q3 | Q0 + on-policy QAD | yes | student rollouts | sampled reverse KL (REINFORCE) |
| Q4 (conditional) | best Q1/Q2/Q3 + 2nd seed | yes | as chosen | as chosen |

Q1, Q2 and Q3 all start from the **same Q0 checkpoint** and use the **same
prompt pool**, teacher, optimizer, LR schedule, step count and response-token
budget. Q1 is the only variant that does not require rollout generation.

### 3.2 What each comparison isolates

- Q0 − T0: quantization damage (RTN).
- Q1 − Q0: offline recovery.
- Q2 − Q1: effect of student-visited prefixes with the same loss.
- Q3 − Q2: effect of the loss estimator/direction on identical rollouts.
- Q3 − Q1: total effect, but confounded (prefix + loss); reported as such, not
  as causal evidence.

### 3.3 Loss definitions

All losses are computed on **response tokens only**, after the prompt, with
padding and post-EOS positions masked. Loss temperature `τ_loss = 1.0`
(no temperature scaling) for every KL objective, so Q1/Q2/Q3 differ only in
prefix source and estimator. Rollout *sampling* temperature is separate and
logged (`τ_sample = 0.6`, Qwen3 recommended settings).

**Next-token NLL** (T0, Q0, and held-out evaluation):

```text
ids: [B, T]  ->  logits: [B, T, V]
logits[:, :-1, :] predicts ids[:, 1:]
nll = -mean_over_valid_positions( log_softmax(logits[:, :-1, :])[target_ids] )
```

**Forward KL** (Q1, Q2):

```text
KL_t = sum_v p_T,t(v) * ( log p_T,t(v) - log p_θ,t(v) )
loss = mean over valid response positions t
```

Computed in fp32; teacher probabilities are `softmax(teacher_logits.float())`.
Full-vocabulary KL, no top-k truncation (top-k is an open question, §13).

**Sampled reverse KL** (Q3), following the standard single-sample
policy-gradient estimator:

```text
a_t   ~ student (the sampled token)
A_t   = stopgrad( log p_θ(a_t) - log p_T(a_t) )          # scalar per position
loss  = mean_t( A_t * log p_θ(a_t) )                     # surrogate, minimized
KL_hat = mean_t( A_t )                                   # unbiased estimate, reported separately
```

The surrogate loss value and `KL_hat` are logged as different quantities. The
surrogate is not a KL and can be negative. A sign test on a 2-token toy model
is part of the test suite.

### 3.4 Logit alignment and masking (known bug magnet)

For prompt length `P` and `R` generated tokens:

- the distribution for generated token `g[0]` is at logit index `P-1`;
- for `g[j]`, it is logit index `P-1+j`;
- response positions scored are `P-1 ... P+R-2` inclusive;
- the logit at `P+R-1` (after the last generated token) is **never** scored.

The teacher and student are evaluated on **the identical token id sequence**;
rollouts are generated once, then re-scored. No teacher-generated continuations
are substituted.

Masking rules:

- prompt tokens: excluded from the loss (all variants);
- response tokens up to and including a sampled EOS: included;
- tokens after EOS, padding: excluded;
- sequences truncated at `max_new_tokens` without EOS: included but flagged;
- offline Q1: prompt and response tokenized separately, so the boundary is exact;
  the same chat template is applied for offline and on-policy.

Two unit tests use a tiny deterministic model with a hand-written vocabulary:
one for the causal shift `[B,T,V]`, one for the prompt/rollout offsets above.

---

## 4. Data

### 4.1 Sources

| Use | Source | License to verify at prep | Size used |
|---|---|---|---|
| Stage-2 prompts + gold completions | `open-r1/OpenThoughts-114k` (or the pinned successor) | verify on the dataset card; record | ~2,000 pairs, fixed list |
| Held-out NLL/KL | `HuggingFaceFW/fineweb-edu` held-out shard | ODC-By 1.0 | 256 × 1,024 tokens, fixed hash |
| Offline prompt text (if needed) | same FineWeb-Edu sample | ODC-By 1.0 | small |
| Generative eval | GSM8K (`openai/gsm8k`) test | MIT | 300 examples |
| Generative eval | MMLU-Redux test | verify | 300 examples |
| Generative eval | MATH-500 test | MIT | 250 examples |
| Generative eval (1.7B only) | GPQA-Diamond | CC BY 4.0, gated | 100 examples |

If a license cannot be verified, the dataset is replaced or dropped, not used
"because it is public".

### 4.2 Splits and leakage

- All split indices are derived from a stable hash of the example id, not from
  a random shuffle without a seed.
- Train prompts, validation prompts, and evaluation examples are disjoint by id
  and by content hash; a test asserts zero overlap by hash.
- The held-out NLL/KL set is never used for checkpoint selection; selection uses
  a separate 32-batch validation slice from FineWeb-Edu.
- Benchmark test sets are never trained on and never used for LR/temperature
  selection.

### 4.3 Formatting

- Use the model's own tokenizer and chat template from the pinned revision;
  do not hand-write the prompt format.
- Qwen3 thinking mode: `enable_thinking=True` for rollouts and generative eval;
  record the exact template string hash.
- Offline Q1: `prompt = apply_chat_template(question, add_generation_prompt=True)`;
  `response` tokenized separately and appended; EOS appended if missing.
- Rollouts: generate with `return_dict_in_generate=True`, keep the raw
  `sequences` tensor, and derive `prompt_len` from the templated prompt ids.
  Never decode-and-retokenize a rollout (that changes tokens).
- Record EOS id, pad id, and whether a BOS is added at Gate A.

---

## 5. Quantization specification

### 5.1 Scheme (pilot default)

For `W ∈ R^{out×in}`, group size `G = 128` along `in`, independently per row:

```text
qmax = 2^(b-1) - 1                     # b=3 -> 3
scale_g = max(|W_g|) / qmax, floor 1e-8
q_g = clamp(round(W_g / scale_g), -qmax, qmax)
W_hat_g = q_g * scale_g
```

Tail groups (when `G` does not divide `in`) are quantized separately with
their own scale; no padding value may enter a group statistic. For both Qwen3
checkpoints all quantized `in_features` are divisible by 128 (1024/3072 for
0.6B; 2048/6144 for 1.7B), so the tail path exists for safety and tests only.

Scales are recomputed from the current master weight on every forward and are
detached (no scale learning in this pilot). The asymmetric alternative:

```text
scale_g = (max - min) / (2^b - 1)
zero_g  = round(-min / scale_g)        # clamped to [0, 2^b-1]
q_g     = clamp(round(W_g / scale_g) + zero_g, 0, 2^b - 1)
W_hat_g = (q_g - zero_g) * scale_g
```

Groups with zero dynamic range use a centred symmetric grid, which recovers
constant groups exactly; the count of such groups is reported. This
`degenerate` policy is documented because an affine grid anchored at zero
would map a constant negative group to 0.

### 5.2 Module coverage

Qwen3 (both sizes) has 28 layers × 7 linear projections = **196 linear modules**.
Everything else stays float. `lm_head.weight` is tied to
`model.embed_tokens.weight`, so the unquantized "embeddings + LM head" is one
155.6M-parameter (0.6B) / 311.2M-parameter (1.7B) tensor and is counted once.

Compute the coverage by role and report it as a table: quantized vs unquantized
parameters, before and after.

### 5.3 Memory accounting — three distinct numbers

1. **Raw payload** = `sum(numel_quantized × bits / 8)` per module. For W3 this
   is 3/8 byte per weight; no metadata.
2. **Theoretical packed checkpoint** = payload + 16-bit scale per group +
   unquantized parameters (BF16) + the tied embedding. Packing/alignment
   overhead is **unknown** and stated as such.
3. **Measured** = `torch.cuda.max_memory_allocated()` and `reserved` during a
   real run, plus checkpoint file size on disk. Fake-quantized weights and fp32
   master weights **do not** realize (2).

Effective bits per quantized weight, `G=128`, symmetric: `3 + 16/128 = 3.125`
(asymmetric with 16-bit zero-point: `3 + 32/128 = 3.25`). Parameter-weighted
across included and excluded modules.

**ESTIMATE:** theoretical packed total ≈ 439 MiB for 0.6B and ≈ 1.09 GiB for
1.7B, versus ≈ 1.0 GiB and ≈ 3.2 GiB of unique BF16 parameters (about 2.3×
and 2.9×). These are not measured and must not be presented as deployment
savings.

---

## 6. Implementation plan

### 6.1 Repository layout

```text
qad/
├── plan.md                        # this file
├── README.md
├── QWEN_LOW_BIT_QAD_RESEARCH_BRIEF.md
├── pyproject.toml
├── configs/
│   ├── pilot_0p6b.yaml            # model, quantizer, data, eval defaults
│   ├── offline_qad.yaml           # Q1 optimizer + KL settings
│   └── online_qad.yaml            # Q2/Q3 rollout + loss settings
├── src/qwen_qad/
│   ├── quantization.py            # DONE
│   ├── models.py                  # load, inspect, wrap eligible linears
│   ├── data.py                    # sources, splits, tokenization, masks
│   ├── losses.py                  # chunked log-probs, NLL, FKL, sampled RKL
│   ├── evaluate.py                # NLL/KL/memory + benchmark suite
│   ├── train_offline.py           # Q1
│   ├── rollouts.py                # student rollouts with metadata
│   ├── train_online.py            # Q2/Q3
│   ├── checkpoint.py              # master weights + quant config + optimizer
│   ├── diagnostics.py             # scales, codes, layer outputs, gradients
│   └── runlog.py                  # run manifest, JSONL metrics, git hash
├── scripts/                       # thin argparse CLIs per gate
├── tests/
│   ├── test_quantizer.py          # DONE (29 tests)
│   ├── test_ste_gradients.py      # to be folded into test_quantizer
│   ├── test_masks_alignment.py
│   ├── test_teacher_freeze.py
│   ├── test_checkpoint_roundtrip.py
│   ├── test_online_rollout.py
│   └── test_losses.py
├── experiments/index.md           # registry: run id -> config -> metrics
└── reports/                       # findings template, figures
```

### 6.2 Increments and gates

| # | Increment | Acceptance criteria | Where | GPU-h (ESTIMATE) |
|---|---|---|---|---|
| 0 | Reference quantizer + tests | **DONE**: 29 tests pass; hand-computed values; grouping/leakage; zero/constant groups; STE identity; no master overwrite | local CPU | 0 |
| 1 | Model loading, inventory, fixed eval sets, T0 baseline | revision + license recorded; logits `[B,T,V]`; 196 linears enumerated by role; teacher self-KL < 1e-5; deterministic eval; T0 metrics persisted | local MPS (smoke); official T0 in the Gate 2 session | 0 |
| 2 | Fake-quant wrapping, Q0, diagnostics, bit accounting | same logits shape; Q0 loss finite; nonzero quant error only in selected weights; excluded modules byte-identical; Q0 evaluated on T0's exact set; reload reproduces Q0 | GPU 24 GB | 1–1.5 |
| 3 | `losses.py` + offline QAD (Q1) + checkpointing | teacher byte-identical, `grad is None`; student grads finite; quantized forward active in train and eval; Q1 beats Q0 on validation NLL; best checkpoint reloads and reproduces | GPU 24 GB | 2–3 |
| 4 | Rollouts + on-policy Q2/Q3 | index/mask tests pass; teacher and student score identical tokens; no gradient through sampling; rollouts change after an update; logs record rollout config; Q2/Q3 complete at matched budget | GPU 24 GB | 3–4 |
| 5 | Benchmark subsets + report | T0/Q0/Q1/Q2/(Q3) table on identical eval; memory method stated; limitations and failures documented; one-command repro | GPU 24 GB | 1–2 |
| 6 | 1.7B W3A16 (conditional, go/no-go after 5) | same as 2–5 at larger scale | GPU 48 GB | 4–7 |

Total for a complete 0.6B answer: **~7–10 GPU-hours (ESTIMATE)**, excluding
failed-run repeats.

### 6.3 Engineering decisions

1. **Wrapper, not weight rewrite.** `FakeQuantLinear` holds an fp32
   `nn.Parameter`; forward is `F.linear(x, fake_quantize(weight), bias)`.
2. **Chunked log-probs.** Never materialize `[B,T,V]`. For `T=2048, V=151,936`
   fp32 that is 1.2 GB per sequence; at 8k tokens it is 5 GB per sequence.
   Process token chunks of 256 positions: compute logits, log-probs, the loss
   term, and call `backward()` per chunk with gradient accumulation.
3. **Teacher pass without autograd**, logits cast to fp32 before softmax; the
   teacher is scored once per token batch and reused for Q1/Q2/Q3 variants.
4. **Inference dequant cache** for rollout generation, so each decode step does
   not re-dequantize 0.38B weights; training always recomputes.
5. **Checkpointing** stores fp32 master weights, `QuantConfig` dict, optimizer
   state, step, RNG state, and model/tokenizer revisions. Reload equality on a
   fixed batch is a hard gate.
6. **Determinism**: seed everything (`random`, `numpy`, `torch`,
   `torch.cuda.manual_seed_all`); fixed eval batches; greedy decode for
   deterministic quality checks; sampled rollouts recorded verbatim. CUDA
   kernels are not bitwise deterministic — this is documented, not hidden.
7. **No automatic retry after OOM.** Reduce microbatch, re-measure, or stop.
8. **Run manifest** per brief §8: UTC timestamp, git commit, config hash,
   torch/transformers/CUDA versions, GPU model/VRAM, revisions, token counts,
   seed, exact command, artifact paths, warnings, GPU-seconds.

### 6.4 Already measured

- 29 unit tests pass on macOS/Python 3.12, torch 2.15.0 (CPU).
- One 3072×1024 W3 quantize-dequantize takes 1.7 ms; MSE 3.0e-5 on
  `N(0, 0.02)` weights; 2.4% of codes saturate; 3.125 effective bits/weight.
- These are microbenchmarks on synthetic weights, not model-quality results.

---

## 7. Training configuration (provisional)

All values below are **starting points to be confirmed after a 20-step probe**.
They are config fields, not hard-coded constants.

| Field | Value |
|---|---|
| precision | bf16 autocast forward, fp32 master weights, fp32 loss math |
| sequence length (offline) | 2,048 |
| prompt cap / generation cap (online) | ≤ 1,024 / 1,024 |
| microbatch | 2 sequences; grad accumulation to **32 sequences per optimizer step** |
| optimizer | AdamW, `betas=(0.9, 0.95)`, `weight_decay=0.0`, `grad_clip=1.0` |
| LR | 1e-5, 10% linear warmup, cosine to 10% of peak |
| steps per Stage-2 run | 150 (Q1/Q2/Q3 identical) |
| rollout refresh | every optimizer step (true on-policy) |
| rollout sampling | `T=0.6`, `top_p=0.95`, `top_k=20`, stop at EOS, no repetition penalty |
| eval cadence | every 25 steps on the fixed 32-batch validation slice |
| seed | 42 (a second seed only if budget allows; then Q1/Q2 only) |

Budget matching: Q1/Q2/Q3 use the same number of optimizer steps and the same
number of **response tokens processed per step** (pad the batch by repeating
prompts if a gold completion is shorter; truncate to 1,024 like rollouts).
Actual token counts are logged; if the mismatch exceeds 10%, it is reported as
a limitation rather than silently normalized.

Reference implementation anchors (not targets): the brief's Q0/Q1/Q2 setup and
the standard distillation recipes. If a loss diverges (non-finite or KL
increasing for 20 consecutive steps), the run stops and is reported as a
failure, not rerun with a different seed.

---

## 8. Evaluation protocol

### 8.1 Every variant, same fixed set

- **Held-out NLL** on 256 × 1,024 FineWeb-Edu tokens (fixed list + hash), with
  correct shift and no padding leakage. Report NLL, perplexity, token count.
- **Teacher–student KL** on the same held-out prefixes: forward KL
  `KL(p_T || p_θ)` at τ=1.0, mean over valid positions; teacher self-KL ≈ 0
  check. (Reverse KL on held-out prefixes is not reported unless added
  deliberately; the direction is always labeled.)
- **Quantization diagnostics**: per-module reconstruction MSE, saturation
  fraction, code histogram, scale distribution; layer-output reconstruction
  error on a fixed batch (`||h_θ - h_T|| / ||h_T||`) for a few selected layers;
  fraction of changed codes vs Q0 after training.
- **Memory**: raw payload, theoretical packed, measured peak allocated/reserved,
  checkpoint size on disk. Labeled separately.
- **Cost**: GPU type, GPU-seconds, tokens processed, throughput where measured.

### 8.2 Generative quality (subsets, headline variants only)

| Benchmark | n (0.6B) | n (1.7B) | Grader |
|---|---|---|---|
| GSM8K | 300 | 400 | final-number exact match |
| MMLU-Redux | 300 | 400 | multiple-choice letter extraction |
| MATH-500 | 250 | 250 | `\boxed{}` extraction, numeric/symbolic match |
| GPQA-Diamond | — | 100 | multiple-choice letter extraction |

- Thinking mode on, **greedy decoding (`T=0`)** for determinism; the same setting
  for every variant, logged. Qwen3's recommended sampling settings are a
  follow-up check, not the primary table.

- Max new tokens capped; parse failures reported separately, never counted as
  correct.
- Generations are cached to disk with token ids, so re-grading is free and
  audit-able.
- LiveCodeBench, IFEval and full-size benchmarks are out of the pilot budget;
  they are listed as follow-ups, not silently omitted.

### 8.3 Statistics

- Report absolute and relative deltas, per task and unweighted mean.
- 300-sample GSM8K has a standard error of roughly ±3 points (p≈0.5); on-policy deltas
  of 1–3 points on one seed are **not** evidence of a real effect. The report
  says this explicitly.
- Bootstrap CI (1,000 resamples) on generative accuracy; KL/NLL reported with
  batch-level standard error.
- One seed is the default. If a second seed is run, it is reported as a separate
  row, not averaged into a false n=2 claim.

### 8.4 Decision thresholds (pre-registered for the pilot)

- Promotion to 1.7B requires: Q1 improves held-out NLL over Q0 by ≥5% relative,
  and the Q0→Q1→Q2 trend is not dominated by a single benchmark.
- A negative result (Q2 ≤ Q1) is an acceptable and fully reported outcome.

---

## 9. Compute and memory

**ESTIMATE** unless stated.

| Item | 0.6B | 1.7B |
|---|---|---|
| unique parameters (config-derived) | ~0.54B | ~1.72B |
| quantized linear parameters | ~0.38B | ~1.41B |
| BF16 weights | ~1.0 GiB | ~3.2 GiB |
| training state (fp32 master + AdamW m/v + grads) | ~6.0 GiB | ~19.2 GiB |
| + BF16 teacher | ~1.1 GiB | ~3.2 GiB |
| activations + chunked logits | ~2–4 GiB | ~6–10 GiB |
| minimum card | 24 GB | 48 GB |
| rollout KV cache (batch 8 × 2k tokens; same GQA geometry) | ~1.8 GiB | ~1.8 GiB |

GPU-hours by gate are in §6.2. The first paid session is expected to be
30–60 minutes for Gate 1+2 on the 0.6B model; the whole 0.6B answer is about
7–10 GPU-hours. No session should exceed a hard 4-hour timeout.

Cost-control protocol per session: one run card (download → run → evaluate →
sync artifacts), checkpoint every 25 steps and on SIGTERM, artifacts synced off
the pod, pod terminated immediately after the artifact checksum is verified.
No idle GPU, no unattended sweeps.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| RTN W3 destroys 0.6B so badly that rollouts are incoherent | Report rollout quality (length, repetition, parse rate) alongside losses; the recovery question is still valid; if Q0 is unusable, Q1/Q2 deltas are interpreted as "recovery from a broken init" |
| On-policy data from a bad student is low-information | Log per-step KL and response length; compare Q2 vs Q1 at matched tokens; note as a limitation |
| Sampled RKL has high variance | Average over ≥4,096 response tokens per step; report `KL_hat` variance across steps; no single-run significance claims |
| Single seed noise | Pre-registered thresholds, bootstrap CIs, "preliminary" label; second seed only for Q1/Q2 if budget allows |
| Benchmark contamination (GSM8K/MATH in pretraining) | Report relative deltas, not absolute capability; state that contamination is assumed |
| Offline/online budget mismatch | Same prompts, steps, response-token target; log actual tokens; report mismatch |
| Chat-template / thinking-mode mismatch across variants | One template function, hash logged per run; a unit test compares tokenization of T0/Q0 inputs |
| Memory blow-up from `[B,T,V]` | Chunked losses; assert peak memory against a budget before step 1 |
| Fake quant mistaken for deployment savings | Three-number memory accounting; report language fixed in the README |
| Data license/gating surprises | License check at Gate 1; datasets that fail are replaced, not used |
| Pod left running | Run cards with `timeout`, terminate after checksum, log GPU-seconds |

---

## 11. Reproducibility contract

Every run writes:

```yaml
run_id: 2026-11-20T10-00-00_q1_0p6b_seed42
git_commit: <sha>
config: configs/offline_qad.yaml
config_hash: <sha256>
versions: {torch: ..., transformers: ..., cuda: ..., python: ...}
gpu: {name: ..., vram_gib: ...}
model: {id: Qwen/Qwen3-0.6B, revision: ..., tokenizer_revision: ...}
data: {source: ..., revision: ..., split_hash: ..., num_train_tokens: ..., num_eval_tokens: ...}
seed: 42
command: ...
artifacts: {checkpoint: ..., metrics_jsonl: ..., eval_json: ...}
gpu_seconds: ...
warnings: [...]
```

`experiments/index.md` links run ids to configs and metrics. Large checkpoints
and datasets are never committed. A `scripts/reproduce.sh` reproduces the
0.6B T0/Q0/Q1/Q2 numbers from a clean environment once Gate 5 passes.

---

## 12. Test plan (beyond the 29 quantizer tests)

| Test | Failure it prevents | When |
|---|---|---|
| causal shift on a hand-built `[B,T,V]` | off-by-one token scoring | Gate 3 |
| rollout offsets, prompt `P=3`, `R=2` | scoring the wrong prefix or the post-final logit | Gate 4 |
| teacher and student score identical ids | teacher scoring its own generations | Gate 4 |
| no gradient path through sampled token ids | silently differentiating through sampling | Gate 4 |
| rollouts change after a weight update | replaying a frozen corpus and claiming on-policy | Gate 4 |
| teacher byte-identical, `grad is None` | teacher contamination | Gate 3 |
| quantized forward active in train **and** eval | evaluating an unquantized student | Gate 2 |
| checkpoint reload → identical logits | lost quant metadata | Gate 3 |
| train/validation overlap by hash | inflated claims | Gate 1 |
| eval deterministic across two runs | sampling noise mistaken for gains | Gate 1 |
| memory fields separated (raw/theory/measured) | fake-quant VRAM claims | Gate 5 |
| loss sign test on a 2-token toy model | reversed KL direction | Gate 3 |

Stop conditions: non-finite loss, `grad` NaN/Inf, teacher `grad` not `None`,
reload mismatch, peak memory over the session budget, or a test failure.
No expensive run starts while any of these is open.

---

## 13. Open questions for the reviewer

These are the decisions I want challenged before implementation continues.

1. **Q1 definition.** I propose offline Q1 uses the *same prompts* as Q2/Q3
   with **gold completions** from OpenThoughts, so that Q1 vs Q2 isolates the
   prefix source. The alternative — arbitrary fixed text (e.g., FineWeb
   passages) — is closer to the brief's literal wording but confounds prompt
   distribution with prefix source. Which is preferred?
2. **Loss temperature.** `τ_loss = 1.0` for all KL objectives, with rollout
   sampling at `τ=0.6`. This avoids a second confound but differs from the
   reference paper's `τ=0.6` loss. Confirm.
3. **Forward-KL truncation.** Full-vocabulary KL (my default) vs top-k teacher
   logits (cheaper). Full vocab is a stronger baseline but costs memory; the
   chunking makes it feasible.
4. **Sampled RKL scope.** Q3 uses the pure surrogate with no CE anchor, per the
   brief's "no CE" note for the first version. Should a small CE anchor be
   allowed as a separate variant?
5. **0.6B generative suite.** Is GSM8K + MMLU-Redux + MATH-500 enough at 0.6B,
   or should IFEval be included and MATH dropped (answer-matching fragility)?
6. **Budget matching rule.** Match optimizer steps and response tokens per step.
   Is token-count matching preferable to step matching, given grading steps are
   the expensive unit?
7. **Promotion criterion to 1.7B.** Is the §8.4 threshold reasonable, or should
   promotion require a specific on-policy effect size?
8. **Seeds.** One seed for all variants plus a second seed only for Q1/Q2, or
   one seed throughout with a "preliminary" label?
9. **Data.** Approve OpenThoughts + FineWeb-Edu, and confirm whether
   MMLU-Redux/GPQA-D gated access is available in this environment.
10. **Scope check.** Given that Stage-1 initialization is excluded, is the
    resulting claim ("on-policy QAD recovery from pure RTN at W3") the one the
    trial expects, or is a stronger initialization needed for credibility?

---

## 14. Immediate next step

Gate 1, entirely local and free:

1. Load `Qwen/Qwen3-0.6B` at a pinned revision on MPS/CPU, record config,
   license, tokenizer special tokens and chat-template hash.
2. Enumerate all parameters and linear modules by role; produce the quantized /
   unquantized coverage table.
3. Build and hash the fixed held-out NLL/KL set (FineWeb-Edu) and the eval
   prompt set.
4. Implement `losses.py` + tests for the causal shift and masking.
5. Produce the T0 baseline (NLL, teacher self-KL, a few generations) locally as
   a smoke test; the official T0 number runs on the GPU in the first paid
   session.

No GPU is needed for any of the above. Gate 2 is the first paid session
(24 GB card, ~1–1.5 h, hard timeout).
