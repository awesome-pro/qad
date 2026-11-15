# PLAN — Qwen3 low-bit weight quantization + online QAD

**Status:** revision 2 — draft for external review (parent researcher agent)
**Author:** implementation agent
**Date:** 2026-11-15
**Revision 2:** incorporates the reviewer's corrections: checkpoint-derived
parameter inventory, corrected diagnostics and exact tail-group accounting,
hidden-state/chunked-LM-head loss design, device and cache handling,
sampling-based Qwen3 evaluation, non-duplicating budget matching, correctness
criteria separated from experimental outcomes, and test-count/reproducibility
cleanup.
**Authoritative source:** `QWEN_LOW_BIT_QAD_RESEARCH_BRIEF.md`. Where this plan and the brief disagree, the brief wins. The published *OnlineQAT* paper (arXiv:2610.09346) is reference material only — this is not a reproduction of it.

Everything below marked **ESTIMATE** is a planning number. Nothing marked *measured* exists yet except the quantizer test results in §6.4.

---

## Summary for the reviewer

- **Question:** does on-policy QAD recover a pure-RTN W3A16 Qwen3 student better than offline QAD, and which factor (prefix source vs loss estimator) matters?
- **Design (primary):** T0 (BF16) / Q0 (RTN) / Q1 (offline forward KL on gold completions) / Q2 (on-policy forward KL). Q1 vs Q2 isolates the prefix source at matched loss, prompts and budget. Q3 (on-policy sampled reverse KL) is an optional third arm, decided after Q1/Q2.
- **Scope:** W3A16 only, no Stage-1 initialization, Qwen3-0.6B first, 1.7B conditional on a pre-registered compute threshold. Estimated 6–9.5 GPU-hours for the primary 0.6B answer; the first paid session is ~1 h on a 24 GB card.
- **Status:** original Gates 0–6 complete; audit corrections (review.md) partly landed on the Mac — strict GSM8K evaluator, matched Q1/Q2 trainer, disjoint val/test splits, best/final checkpoints, statistics, portable weight export. 142 tests collected (141 pass, 1 CUDA-only skip). Corrected GPU runs are pending; see §15 and `GPU_RUNBOOK_AUDIT.md`. No publication claims.
- **Decisions I most want checked:** §13 items 1 (Q1 definition), 2 (loss temperature), 6 (budget matching), 7 (promotion criterion), 10 (whether the resulting claim is strong enough for the trial).
- **Deliberate deviations / judgment calls:** Q1 uses the same prompts with gold completions rather than arbitrary fixed text; generative benchmarks use Qwen3's recommended sampling (no greedy decoding in thinking mode); sampled RKL is an optional third arm, not part of the primary scope.

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
4. Does switching the **loss direction/estimator** (forward KL → sampled
   reverse KL) help (Q3 vs Q2)? Each run samples its own rollouts, so this is a
   between-method comparison, not a paired one.

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
| Q0 initialization | W3A16-G128 RTN is the **aggressive baseline**, not a locked QAD initialization. W3-G32, W4A16-G128 and a stronger W3 initialization remain separate options |
| Trainable params | only the 196 wrapped linear masters (fp32); embeddings, LM head and all norms frozen (`requires_grad=False`) |
| Primary scope | T0/Q0/Q1/Q2. Q3, 1.7B promotion and packed export are subsequent decisions |
| Teacher | same checkpoint, BF16, `eval()`, all parameters `requires_grad_(False)` |
| Offline loss | forward KL from teacher to student, at response-token positions |
| On-policy loss | forward KL on student-visited prefixes (Q2); sampled reverse KL is the optional Q3 arm |
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
| Q3 (optional, subsequent decision) | Q0 + on-policy QAD | yes | student rollouts | sampled reverse KL (REINFORCE) |
| Q4 (conditional) | best Q1/Q2/Q3 + 2nd seed | yes | as chosen | as chosen |

Q1 and Q2 (and Q3 if it is run) start from the **same Q0 checkpoint** and use
the **same prompt pool**, teacher, optimizer, LR schedule, step count and
response-token budget. Q1 is the only variant that does not require rollout
generation. The primary scope stops at Q2; Q3 is decided afterwards.

### 3.2 What each comparison isolates

- Q0 − T0: quantization damage (RTN).
- Q1 − Q0: offline recovery.
- Q2 − Q1: effect of student-visited prefixes with the same loss.
- Q3 − Q2: effect of the loss estimator/direction, but the two runs generate
  their own rollouts, so this is a between-method comparison, not paired.
- Q3 − Q1: total effect, but confounded (prefix + loss); reported as such, not
  as causal evidence.

### 3.3 Loss definitions

All losses are computed on **response tokens only**, after the prompt, with
padding and post-EOS positions masked. Loss temperature `τ_loss = 1.0`
(no temperature scaling) for every KL objective, so Q1/Q2 differ only in prefix
source, and Q3 differs from Q2 only in the estimator (its rollouts are
nevertheless independently generated). Rollout *sampling* temperature is
separate and logged (`τ_sample = 0.6`, Qwen3 recommended settings).

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

### 5.2 Architecture inventory (to be confirmed from the checkpoint)

Qwen3 decouples `head_dim` from `hidden_size`. **For Qwen3-0.6B the query
projection output is 2048, not 1024** (`16 heads × 128 = 2048`), so the
attention projections are not square. Architecture-derived counts:

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---:|---:|
| layers | 28 | 28 |
| hidden size | 1,024 | 2,048 |
| heads × head_dim | 16 × 128 | 16 × 128 |
| q_proj shape | [2048, 1024] | [2048, 2048] |
| kv heads × head_dim | 8 × 128 | 8 × 128 |
| k_proj / v_proj shape | [1024, 1024] | [1024, 2048] |
| o_proj shape | [1024, 2048] | [2048, 2048] |
| MLP intermediate | 3,072 | 6,144 |
| vocab | 151,936 | 151,936 |
| total parameters | 596,049,920 | 1,720,574,976 |
| quantized linear parameters | 440,401,920 | 1,409,286,144 |
| frozen (tied embedding + norms) | 155,648,000 | 311,288,832 |

**Confirmed at Gate 1 from the loaded checkpoint** (`Qwen/Qwen3-0.6B` @
`c1899de289a04d12100db370d81485cdf75e47ca`): 596,049,920 parameters,
440,401,920 quantizable, 155,648,000 frozen; **197 `nn.Linear` modules = 196
quantizable projections + the tied `lm_head`**. The tokenizer vocabulary is
151,643 while the embedding matrix is padded to 151,936 rows; the padded logits
participate in the softmax and are never targets. The 1.7B column stays
architecture-derived until its own inventory is taken.

Qwen3 (both sizes) has 28 layers × 7 quantizable projections. `lm_head.weight`
is tied to `model.embed_tokens.weight`, so the unquantized "embeddings + LM
head" is one tensor (155.6M / 311.2M parameters) and is counted once.

### 5.3 Memory accounting — three distinct numbers

1. **Raw payload** = `sum(numel_quantized × bits / 8)` per module. For W3 this
   is 3/8 byte per weight; no metadata.
2. **Theoretical packed checkpoint** = payload + 16-bit scale per group +
   unquantized parameters (BF16) + the tied embedding. Packing/alignment
   overhead is **unknown** and stated as such.
3. **Measured** = `torch.cuda.max_memory_allocated()` and `reserved` during a
   real run, plus checkpoint file size on disk. Fake-quantized weights and fp32
   master weights **do not** realize (2).

**Exact metadata accounting** uses
`groups = out_features × ceil(in_features / group_size)`, so a tail group costs
a full scale:
`bits_per_weight = (weights × bits + groups × 16 + [asym] groups × 16) / weights`.
Example: `(4, 130)` at `G=128` has 8 groups and **3.246** bits/weight, not the
3.125 of an exact-multiple shape. `storage_bits()` returns the payload / scale /
zero-point / total breakdown and is unit-tested. Parameter-weighted accounting
across included and excluded modules.

**ESTIMATE:** theoretical packed total ≈ **461 MiB** for 0.6B and ≈ **1.09 GiB**
for 1.7B, versus ≈ **1.11 GiB** and ≈ **3.20 GiB** of unique BF16 parameters
(about 2.5× and 2.9×). Not measured; never presented as deployment savings.

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
│   ├── test_quantizer.py          # DONE (43 quantizer tests)
│   ├── test_losses.py             # DONE (19 tests)
│   ├── test_wrapping.py           # DONE (8 tests)
│   ├── test_models.py             # DONE (6 tests)
│   ├── test_teacher_freeze.py
│   ├── test_checkpoint_roundtrip.py
│   ├── test_online_rollout.py
├── experiments/index.md           # registry: run id -> config -> metrics
└── reports/                       # findings template, figures
```

### 6.2 Increments and gates

Primary scope is T0/Q0/Q1/Q2. Q3, the 1.7B promotion and any packed export are
separate decisions after the primary result.

| # | Increment | Acceptance criteria | Where | GPU-h (ESTIMATE) |
|---|---|---|---|---|
| 0 | Reference quantizer + tests | **DONE**: 98 tests collected (97 pass, 1 CUDA-only skipped); quantizer, masks/alignment, chunked-loss equivalence, checkpoint inventory, wrapping, device preservation, cache invalidation, data EOS policy, memory sampler | local CPU | 0 |
| 1 | Model loading, checkpoint inventory, fixed eval sets, T0 baseline | **DONE (local)**: inventory confirmed (§5.2); FineWeb-Edu 256×1,024 tokens and 3,000 OpenThoughts prompts pinned by hash; chunked losses match naive to 1e-6; teacher self-KL computed = 0; local T0/Q0 smoke only, **not reportable**. Official T0 runs in Gate 2 | local MPS | 0 |
| 1.5 | W3A16 degradation diagnosis | **DONE (local)**: disabled-wrap equivalence exact (0.0 logit/NLL diff); BF16/W4A16/W3A16-G32/G64/G128 plus attention-only and MLP-only on 65,472 tokens; teacher self-KL computed from two passes = 0; revisions resolved; memory reporting corrected; EOS-after-truncation fixed | local MPS | 0 |
| 2 | Official baseline + generation diagnostic + gradient/memory probe | **DONE (CUDA)**: full 256-block eval of BF16, W3A16-G128, W3A16-G32, W4A16-G128; generation diagnostic; 3-step probe at W4 and W3 with all assertions true; ~7 min of GPU compute | GPU 24 GB | 1.5–2 |
| 3 | `losses.py` + offline QAD (Q1) + checkpointing | chunked-loss equivalence tests pass on a tiny model (loss and all gradients vs a naive reference); teacher byte-identical, `grad is None`; student grads finite; quantized forward active in train and eval; training objective finite and non-increasing on average over the first 20 steps (diagnostic; a flat curve is reported, not treated as failure); checkpoint reloads and reproduces | GPU 24 GB | 2–3 |
| 4 | Rollouts + on-policy Q2 | index/mask tests pass; teacher and student score identical tokens; no gradient through sampling; cache invalidated between step and rollout; rollouts change after an update; Q2 completes at the matched response-token budget | GPU 24 GB | 2–3 |
| 4b | Q3 sampled reverse KL (optional, separate decision) | same as Gate 4; estimator sign test; `KL_hat` and surrogate logged separately | GPU 24 GB | 1.5–2 |
| 5 | Benchmark subsets + report | T0/Q0/Q1/Q2 table on identical eval; memory method stated; limitations and failures documented; commands documented and re-run once from a clean environment | GPU 24 GB | 1–2 |
| 6 | 1.7B W3A16 (conditional, go/no-go after 5) | same as 2–5 at larger scale | GPU 48 GB min | 4–7 |

Total for a complete 0.6B primary answer: **~6–9.5 GPU-hours (ESTIMATE)**,
excluding failed-run repeats; Q3 adds ~1.5–2 h. **No gate requires QAD to
improve a metric** — see §8.4.

### 6.3 Engineering decisions

1. **Wrapper, not weight rewrite.** `FakeQuantLinear` holds an fp32
   `nn.Parameter`; forward is `F.linear(x, fake_quantize(weight), bias)`.
   Only the 196 wrapped masters are trainable; embeddings, LM head and norms
   have `requires_grad=False`.

2. **Hidden-state chunking, not logit slicing.** Slicing an already
   materialized `[B,T,V]` saves nothing. The design is:
   - run the transformer **once** per batch over the full prompt+response
     tokens, then gather hidden states at the scored response positions:
     `h_θ [N_resp, H]` with autograd, `h_T [N_resp, H]` under `no_grad`;
   - for each chunk of `C=256` response positions, wrap
     `torch.utils.checkpoint` around a function that applies the student and
     teacher LM heads to that chunk, computes log-softmax, and returns the
     chunk loss;
   - sum the chunk losses and call `backward()` **once**.
   Memory is `O(C×V)` plus the hidden states; the transformer graph is
   traversed once in forward and once in backward, and only the LM head is
   recomputed per chunk (≤ one extra LM-head pass per step).
   Rejected alternatives: (a) per-chunk `backward(retain_graph=True)` on a
   shared hidden graph — this re-runs the transformer backward once per chunk,
   up to N× the cost; (b) sequence-chunked full-model forwards without a KV
   cache — they break causality.
   A pre-step guard verifies that `C × V × 4` plus hidden-state bytes fit the
   configured budget, and reports the naive `B × T_resp × V × 4` footprint it
   avoided. The guard aborts before the first optimizer step, not mid-run.

3. **Teacher pass without autograd.** Teacher hidden states are stored for the
   scored positions only; teacher LM-head logits are computed chunk-wise inside
   the checkpointed loss, so no `[N,V]` teacher tensor is ever materialized.

4. **Device-preserving conversion.** `FakeQuantLinear.from_linear` constructs
   the fp32 master on `linear.weight.device`, keeps the bias device/dtype, and
   skips random initialization. Model conversion asserts every wrapped module
   is on the same device as its source layer and that the master is fp32. A
   CUDA-specific test runs in the first GPU session.

5. **Dequant cache invalidation.** No-grad forwards (rollout generation) use a
   cached dequantized weight keyed by the parameter's `_version`; the cache is
   cleared on `.to()` and by an explicit `invalidate_cache()`. The trainer
   calls it after every `optimizer.step()` and before generating new rollouts.
   A stale cache would generate rollouts from pre-update weights and turn
   "on-policy" into a lie.

6. **Checkpointing** stores fp32 master weights, `QuantConfig` dict, optimizer
   state, step, RNG state, and model/tokenizer revisions. Reload equality on a
   fixed batch is a hard gate.

7. **Determinism**: seed everything (`random`, `numpy`, `torch`,
   `torch.cuda.manual_seed_all`); fixed teacher-forced eval batches. Generated
   text uses the fixed sampling settings in §8.2 and is stochastic; bitwise
   reproducibility of sampled generations is **not** claimed. The same seed
   and batch composition are used across variants.

8. **No automatic retry after OOM.** Reduce microbatch, re-measure, or stop.

9. **Run manifest** per brief §8: UTC timestamp, git commit, config hash,
   torch/transformers/CUDA versions, GPU model/VRAM, revisions, token counts,
   seed, exact command, artifact paths, warnings, GPU-seconds.

### 6.4 Already measured (local, no GPU)

- 98 tests collected: 97 pass, 1 CUDA-only test skipped on macOS; Python 3.12,
  torch 2.15.0.
- **Inventory** (from the loaded checkpoint): 596,049,920 parameters,
  440,401,920 quantizable (73.9%), 155,648,000 frozen; 197 `nn.Linear` = 196
  quantizable + tied `lm_head`; `q_proj` is `[2048, 1024]`; tokenizer vocab
  151,643 vs 151,936 embedding rows.
- **Data**: FineWeb-Edu `sample-10BT` 256 × 1,024 tokens
  (`ids_sha256 8676a899…`); 3,000 OpenThoughts prompts (mean 683 prompt tokens,
  max 1,024). **2,979 / 3,000 gold responses are longer than the 1,024-token
  cap** (median response ~7.3k tokens). EOS is appended **only** to the 21
  complete responses; the truncated ones end without EOS (fixed at Gate 1.5).
  Q1 is therefore "fixed, length-capped gold prefixes", not full gold traces.
- **Loss path**: chunked NLL / forward KL / sampled RKL match the naive
  full-logits reference to 1e-6 for losses and gradients on a tiny Qwen3; on the
  real model, chunked NLL matches naive to 1e-6. Teacher self-KL is **computed
  between two independent forward passes** (not assigned) and is 0.0 over
  4,092 tokens.
- **Gate 1.5 diagnosis** (64 blocks, 65,472 scored tokens, bf16, MPS; details in
  `experiments/gate15/DIAGNOSIS.md`):

  | config | NLL | ppl | KL(T‖S) |
  |---|---:|---:|---:|
  | BF16 | 3.110 | 22.4 | 0.000 |
  | W4A16-G128 | 3.424 | 30.7 | 0.405 |
  | W3A16-G32 | 4.569 | 96.5 | 1.700 |
  | W3A16-G64 | 5.408 | 223.2 | 2.614 |
  | W3A16-G128 | 6.626 | 754.4 | 3.919 |
  | W3A16-G128 attention-only | 4.264 | 71.1 | 1.348 |
  | W3A16-G128 MLP-only | 4.565 | 96.1 | 1.663 |

  Controls: the fully wrapped model with quantization disabled reproduces BF16
  logits and NLL exactly (0.0 difference); clipped fraction is 0.0 everywhere.
  Final-layer L2 relative activation error is 0.96 for W3-G128 and 0.41 for W4;
  cosine similarity at layer 27 is 0.724 and 0.932 respectively, so the L2
  number is magnitude-relative and must not be read as direction loss.
  The v2 record adds cosine; v1 is preserved under `experiments/gate15/archive_v1/`.
- **Memory reporting corrected**: process peaks are sampled (MPS lower bound)
  and reported separately from parameter bytes; a teacher+student process is
  never compared with a single-model run. Wrapped student parameter bytes are
  1,977 MiB vs 1,137 MiB BF16 (fp32 masters), not a deployment number.
- **Smoke-tested locally**: the fixed-prompt generation diagnostic separates
  models (2 prompts × 32 tokens: W3-G128 4-gram repetition 0.36 vs 0.00 for
  BF16 and W4), and the 1-step W4 gradient/memory probe passes all assertions
  (loss 0.583, finite gradients, teacher frozen, master weights changed,
  quantization active).
- **Implication:** the W3A16-G128 failure is intrinsic to pure RTN, not a
  wrapper/quantizer bug. W3-G128 is therefore the **aggressive baseline**, not a
  locked QAD initialization. The QAD pipeline should first be verified on a
  tractable initialization (W4A16-G128 is the leading candidate), with W3-G32
  and a stronger W3 initialization preserved as separate options. See §13
  item 9 and the diagnosis document.

---

### 6.5 Gate 2 results (official CUDA, RTX A5000 24 GB)

Full 256-block held-out set (261,888 scored positions), identical data/masks.

| config | NLL | PPL | KL(T‖S) | peak GPU |
|---|---:|---:|---:|---:|
| BF16 | 3.1027 | 22.26 | 0.0000 | 2.16 GiB |
| W3A16-G128 | 6.5848 | 723.98 | 3.8674 | 5.73 GiB |
| W3A16-G32 | 4.5480 | 94.44 | 1.6693 | 5.73 GiB |
| W4A16-G128 | 3.4207 | 30.59 | 0.4057 | 5.73 GiB |

- T0 two-pass self-KL is exactly 0; the CUDA inventory matches the Mac
  inventory exactly; the eval hash matches Gate 1.
- Generation (8 fixed prompts, same settings/seed): 4-gram repetition
  0.047 (BF16), 0.965 (W3-G128), 0.839 (W3-G32), 0.117 (W4); all 8/8 hit the
  256-token cap, so premature-EOS is uninformative at this cap.
- Probe (3 AdamW steps, offline forward KL on gold completions): W4 losses
  0.589 → 0.340 → 0.374, W3 4.519 → 2.904 → 2.633; 9.32 GiB peak for both; all
  assertions true (teacher frozen, finite gradients, quantization active,
  masters changed).
- Details and caveats (including a known `parameter_bytes.note` metadata bug in
  the Q0 JSONs, values correct): `experiments/gate2/README.md`.

---

### 6.6 Gate 3–5 pilot results (W4A16-G128, single seed)

Held-out set (261,888 tokens), forward-KL QAD, no CE term:

| Variant | NLL | PPL | KL(T‖S) | GSM8K (100) |
|---|---:|---:|---:|---:|
| T0 BF16 | 3.1027 | 22.26 | 0.0000 | 33% |
| Q0 W4-G128 RTN | 3.4207 | 30.59 | 0.4057 | 1% |
| Q1-128, 25 steps (offline) | 3.2942 | 26.96 | 0.3009 | 13% |
| Q2, 25 steps (on-policy) | 3.3021 | 27.17 | 0.3124 | 16% |
| Q1, 200 steps (offline) | 3.2688 | 26.28 | 0.2532 | 28% |
| Q2, 200 steps (on-policy) | 3.2676 | 26.25 | 0.2646 | 14% |

- Q1-200 closes about half of the RTN NLL gap; GSM8K recovers from 1% to 28%
  against 33% BF16.
- The 25-step arms are matched at ~102K response tokens and 128-token responses:
  offline is slightly better on NLL/KL, on-policy is 3 GSM8K points better.
  Single seed, 100 examples: suggestive, not decisive.
- At 200 steps / ~819K response tokens, Q1 and Q2 are tied on NLL/PPL, offline
  is better on KL, and offline is twice as accurate on GSM8K (28% vs 14%).
  Q2-200 has the least repetitive generation of the quantized arms (4-gram
  0.050), so repetition and task accuracy dissociate. The 128-token rollout cap
  is the leading hypothesis for Q2's task-quality collapse; replication and a
  rollout-length ablation are required before interpretation.
- Generation repetition returns toward BF16 (4-gram 0.117 Q0 → 0.067 Q1-200 /
  0.053 Q2, BF16 0.047); all configs hit the 256-token cap.
- Cost: Q1-200 313 s / 818,760 response tokens / 14.6 GiB; Q2-25 308 s /
  101,499 tokens / 14.0 GiB; checkpoints 5.3–5.6 GB each (not committed).
Full write-up: `reports/findings.md`.

---

### 6.7 Replication and exploratory 1.7B

> GSM8K values in this section use the legacy parser and are historical; see §15.

**Second seed (43), 25-step matched arms (0.6B):** NLL/KL and the GSM8K ordering
replicate. Q1-128: NLL 3.2938, PPL 26.95, KL 0.3015, GSM8K 16% (seed 42:
3.2942 / 26.96 / 0.3009 / 13%). Q2: 3.2989 / 27.08 / 0.3083 / 18% (seed 42:
3.3021 / 27.17 / 0.3124 / 16%). Offline remains slightly better on NLL/KL;
on-policy remains 2–3 GSM8K points better.

**Second seed, 200-step arms:** Q1 3.2634 / 26.14 / 0.2534 / 28% GSM;
Q2 3.2645 / 26.17 / 0.2640 / 22% GSM. Compared with seed 42 (Q1 3.2688 / 26.28
/ 0.2532 / 28%; Q2 3.2676 / 26.25 / 0.2646 / 14%): Q1/Q2 tied on NLL both seeds,
Q1 better on KL both seeds, Q1 at least as good on GSM8K both seeds. On-policy
is not better, and its GSM8K result is less stable (14% vs 22%).

**Rollout-length ablation (25 steps, ~102,400 tokens):** 128 → 256-token
responses raises Q1 from 26.96 PPL / 0.3009 KL / 13% GSM to 26.81 / 0.2895 /
15%, and Q2 from 27.17 / 0.3124 / 16% to 27.11 / 0.3032 / 18%. The ordering is
unchanged; the 128-token cap is not the whole explanation for Q2's 200-step
GSM8K collapse.

**W3A16-G128 QAD probe (offline):** PPL 723.98 → 83.56 after 25 steps
(NLL 4.4256, KL 1.5954, GSM 3%) and → 59.35 after 200 steps (NLL 4.0834,
KL 1.2008, GSM 0% with 25% parse). Likelihood recovers by an order of magnitude
but generation remains unusable; W4 is still the only viable generative
initialization in this pilot.

**Exploratory 1.7B (A6000 48 GB, single seed):**

| Variant | NLL | PPL | KL | GSM8K |
|---|---:|---:|---:|---:|
| T0 BF16 | 2.8327 | 16.99 | 0.0000 | 36% |
| Q0 W4-G128 | 2.9935 | 19.96 | 0.2516 | 16% |
| Q1-200 | 2.8812 | 17.84 | 0.1839 | 27% |
| Q2-25 | 2.8947 | 18.08 | 0.2091 | 19% |

- W4 RTN costs 2.97 PPL (17.5%, vs 37% at 0.6B); Q1-200 recovers 71% of the
  PPL gap and 55% of the GSM8K gap.
- Q2-25 (102,400 tokens, 14% of Q1's 725,916) improves over Q0 but trails Q1,
  the same ordering as 0.6B.
- Q1 training 572 s / 37.7 GiB peak / 17.5 GB checkpoint; Q2 36.2 GiB.
- Q1 used max_seq_len 1536, so response tokens are fewer than the 0.6B run;
  this is an exploratory arm, not a matched-budget 1.7B comparison.

---

## 7. Training configuration (provisional)

All values below are **starting points to be confirmed after a 20-step memory
and gradient probe**. They are config fields, not hard-coded constants.

| Field | Value |
|---|---|
| precision | bf16 autocast forward, fp32 master weights, fp32 loss math |
| trainable parameters | the 196 wrapped linear masters only (440.4M for 0.6B); embeddings, LM head and all norms frozen |
| sequence length (offline) | 2,048 |
| prompt cap / generation cap (online) | ≤ 1,024 / 1,024 |
| per-step response-token budget | 8,192 |
| microbatch | as many complete sequences as fit the token budget; no duplicated or padding examples; grad accumulation to one optimizer step |
| optimizer | AdamW, `betas=(0.9, 0.95)`, `weight_decay=0.0`, `grad_clip=1.0` |
| LR | 1e-5, 10% linear warmup, cosine to 10% of peak |
| steps per Stage-2 run | 150 (Q1 and Q2 identical; Q3 if run) |
| rollout refresh | every optimizer step (true on-policy) |
| rollout sampling | `T=0.6`, `top_p=0.95`, `top_k=20`, stop at EOS, no repetition penalty |
| eval cadence | every 25 steps on the fixed 32-batch validation slice |
| seed | 42 (a second seed only if budget allows; then Q1/Q2 only) |

**Budget matching without artificial repetition.** Batches are built by
response-token count, not by a fixed prompt count: examples are added until the
next one would exceed the step budget, so a short gold completion is never
duplicated and a long rollout is not padded. Sequences longer than 1,024
response tokens are truncated and flagged. Q1, Q2 (and Q3) target the same
total response tokens and the same number of optimizer steps; actual counts are
logged, and a >5% cross-run deviation is reported as a limitation.

**Q2 and Q3 are not paired.** Each on-policy run samples its own rollouts from
a different student state, so the two variants do not see identical tokens even
with the same prompt order and sampling seed. Q2 vs Q3 is a between-method
comparison, not a paired one. An optional diagnostic scores both objectives on
a frozen Q0 rollout set, but it is labeled off-policy and is not the primary
evidence.

Reference implementation anchors (not targets): the brief's Q0/Q1/Q2 setup and
standard distillation recipes. A non-finite loss stops the run; a flat curve is
reported as an outcome, not a failed test.

---

## 8. Evaluation protocol

### 8.1 Every variant, same fixed set

- **Held-out NLL** on 256 × 1,024 FineWeb-Edu tokens (fixed list + hash), with
  correct shift and no padding leakage. Report NLL, perplexity, token count.
- **Teacher–student KL** on the same held-out prefixes: forward KL
  `KL(p_T || p_θ)` at τ=1.0, mean over valid positions; teacher self-KL ≈ 0
  check. (Reverse KL on held-out prefixes is not reported unless added
  deliberately; the direction is always labeled.)
- **Quantization diagnostics**: per-module reconstruction MSE, **endpoint
  occupancy**, **clipped fraction** (pre-clamp out-of-range), and **zero-level
  fraction**, plus code histogram and scale distribution; layer-output
  reconstruction error on a fixed batch (`||h_θ - h_T|| / ||h_T||`) for a few
  selected layers; fraction of changed codes vs Q0 after training; exact
  group counts and bits-per-weight per module.
- **Memory**: raw payload, theoretical packed, measured peak allocated/reserved,
  checkpoint size on disk. Labeled separately.
- **Cost**: GPU type, GPU-seconds, tokens processed, throughput where measured.

### 8.2 Generative quality (subsets, headline variants only)

| Benchmark | n (0.6B) | n (1.7B) | Mode | Grader |
|---|---|---|---|---|
| GSM8K | 300 | 400 | thinking | final-number exact match |
| MMLU-Redux | 300 | 400 | non-thinking | multiple-choice letter extraction |
| MATH-500 | 250 | 250 | thinking | `\boxed{}` extraction, numeric/symbolic match |
| GPQA-Diamond | — | 100 | thinking | multiple-choice letter extraction |

- **No greedy decoding in thinking mode** (review guidance; Qwen3 recommends
  sampling for thinking mode). Every variant uses the model card's recommended
  settings: thinking `T=0.6, top_p=0.95, top_k=20`; non-thinking
  `T=0.7, top_p=0.8, top_k=20`.
- Mode and sampling settings are fixed per benchmark and **identical across
  T0/Q0/Q1/Q2**; they are logged in the run manifest and hashed with the eval
  manifest.
- A fixed seed and fixed prompt order are used for all variants. Generation is
  still stochastic; accuracy differences carry sampling noise (§8.3).
- Max new tokens: 4,096 (0.6B) / 8,192 (1.7B), identical across variants; parse
  failures are reported separately and never counted as correct.
- Generations are cached to disk with token ids, so re-grading is free.
- LiveCodeBench, IFEval and full-size benchmarks are outside the pilot budget;
  they are listed as follow-ups, not silently omitted.

### 8.3 Statistics

- Report absolute and relative deltas, per task and unweighted mean.
- 300-sample GSM8K has a standard error of roughly ±3 points (p≈0.5);
  on-policy deltas of 1–3 points on one seed are **not** evidence of a real
  effect.
- Generative accuracy carries sampling noise on top of dataset noise; the
  report states the sampling settings and does not claim bitwise
  reproducibility of generated text.
- Bootstrap CI (1,000 resamples) on generative accuracy; KL/NLL reported with
  batch-level standard error.
- One seed is the default. If a second seed is run, it is a separate row, not
  averaged into a false n=2 claim.

### 8.4 Compute-allocation thresholds (not correctness criteria)

These decide whether more GPU hours are justified. They are **not** acceptance
tests for the implementation, and a negative result is a valid, reportable
outcome.

- Promote to 1.7B only if Q1 reduces held-out NLL relative to Q0 by ≥5%
  **and** the Q0→Q1→Q2 trend is not dominated by a single benchmark.
- Q2 ≤ Q1, or Q1 ≈ Q0, is reported as a negative/inconclusive result with the
  same care as a positive one. No rerun-to-significance and no selective
  checkpoint reporting.
- Gate acceptance in §6.2 is mechanical (shapes, gradients, masking, reload,
  memory) and never depends on model quality.

---

## 9. Compute and memory

**ESTIMATE** unless stated; parameter counts are architecture-derived and are
confirmed for 0.6B at Gate 1; the 1.7B column stays architecture-derived.

| Item | 0.6B | 1.7B |
|---|---|---|
| unique parameters | 596,049,920 | 1,720,574,976 |
| trainable (wrapped linear masters) | 440,401,920 | 1,409,286,144 |
| frozen student params (tied embedding + norms) | 155,648,000 | 311,288,832 |
| BF16 unique weights | ~1.11 GiB | ~3.20 GiB |
| training state (fp32 master + grad + AdamW m/v, trainable only) | ~6.57 GiB | ~21.0 GiB |
| frozen student + BF16 teacher | ~2.22 GiB | ~6.40 GiB |
| weights + optimizer total | ~8.8 GiB | ~27.4 GiB |
| activations + one chunk of logits | ~2–4 GiB | ~6–10 GiB |
| minimum card | 24 GB | 48 GB (80 GB comfortable) |
| rollout KV cache (batch 8 × 2k tokens; same GQA geometry) | ~1.8 GiB | ~1.8 GiB |

GPU-hours by gate are in §6.2. The first paid session is Gate 2 (~1–1.5 h:
official T0 plus Q0 on the 0.6B model); the 0.6B primary answer is
~6–9.5 GPU-hours. No session should exceed a hard 4-hour timeout.

Cost-control protocol per session: one run card (download → run → evaluate →
sync artifacts), checkpoint every 25 steps and on SIGTERM, artifacts synced off
the pod, pod terminated immediately after the artifact checksum is verified.
No idle GPU, no unattended sweeps.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| RTN W3 destroys 0.6B so badly that rollouts are incoherent | **Measured locally**: Q0 NLL 6.59 vs T0 3.14 (ppl 727 vs 23) on 32,736 tokens. Gate 2 confirms officially. Report rollout quality (length, repetition, parse rate); if Q0 is unusable, Q1/Q2 is framed as recovery from a broken init, and a W4A16 arm becomes a decision for the reviewer, not a silent substitution |
| On-policy data from a bad student is low-information | Log per-step KL and response length; compare Q2 vs Q1 at matched tokens; note as a limitation |
| Sampled RKL has high variance | Average over ≥4,096 response tokens per step; report `KL_hat` variance across steps; no single-run significance claims |
| Single seed noise | Pre-registered thresholds, bootstrap CIs, "preliminary" label; second seed only for Q1/Q2 if budget allows |
| Benchmark contamination (GSM8K/MATH in pretraining) | Report relative deltas, not absolute capability; state that contamination is assumed |
| Offline/online budget mismatch | Same prompts, steps, response-token target; log actual tokens; report mismatch |
| Chat-template / thinking-mode mismatch across variants | One template function, hash logged per run; a unit test compares tokenization of T0/Q0 inputs |
| Memory blow-up from `[B,T,V]` | Hidden-state chunking with checkpointed LM head; pre-step byte guard; CUDA peak-memory probe before training |
| Chunked loss silently changes the objective | Equivalence test vs naive full-logits loss on a tiny model, including every parameter gradient, before Gate 3 |
| Stale dequant cache turns on-policy rollouts off-policy | Cache keyed by `_version`; explicit invalidation after each `optimizer.step()`; regression test |
| Device mismatch after model conversion | `from_linear` preserves device; conversion assertion; CUDA test in the first GPU session |
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
and datasets are never committed. At Gate 5 a `scripts/reproduce.sh` is written
and run **once** from a clean environment. It is claimed to reproduce numbers
only within the tolerance actually observed for teacher-forced evaluations.
Sampled generations are seeded but not bitwise reproducible, and no such
guarantee is made.

---

## 12. Test plan (beyond the 142 collected tests)

| Test | Failure it prevents | When |
|---|---|---|
| causal shift on a hand-built `[B,T,V]` | off-by-one token scoring | Gate 3 |
| chunked loss and all parameter grads == naive full-logits reference (NLL, FKL, sampled RKL) | chunking silently changing the objective or dropping gradients | Gate 3 |
| chunked path never allocates `[N_resp,V]` | materializing the tensor the chunking was meant to avoid | Gate 3 (CUDA probe) |
| wrapped module device == source layer device | model partially moved during conversion | Gate 2 |
| dequant cache invalidated after an in-place update | rollouts generated from pre-step weights | Gate 2/4 |
| tail-group bits-per-weight exact | understating metadata for odd shapes | Gate 0 (done) |
| endpoint occupancy ≠ clipping | misreading structural saturation as clipping | Gate 0 (done) |
| rollout offsets, prompt `P=3`, `R=2` | scoring the wrong prefix or the post-final logit | Gate 4 |
| teacher and student score identical ids | teacher scoring its own generations | Gate 4 |
| no gradient path through sampled token ids | silently differentiating through sampling | Gate 4 |
| rollouts change after a weight update | replaying a frozen corpus and claiming on-policy | Gate 4 |
| teacher byte-identical, `grad is None` | teacher contamination | Gate 3 |
| quantized forward active in train **and** eval | evaluating an unquantized student | Gate 2 |
| checkpoint reload → identical logits | lost quant metadata | Gate 3 |
| train/validation overlap by hash | inflated claims | Gate 1 |
| teacher-forced eval identical across two runs | silent nondeterminism in NLL/KL | Gate 1 |
| memory fields separated (raw/theory/measured) | fake-quant VRAM claims | Gate 5 |
| loss sign test on a 2-token toy model | reversed KL direction | Gate 3 |

Stop conditions: non-finite loss, `grad` NaN/Inf, teacher `grad` not `None`,
reload mismatch, peak memory over the session budget, or a test failure.
No expensive run starts while any of these is open.

---

## 13. Open questions for the reviewer

Resolved by the revision-2 review: budget matching (now token-count based, with
no duplicated examples) and the promotion threshold (now a compute-allocation
rule, not a correctness gate). Still open:

1. **Q1 definition.** Offline Q1 uses the same prompts with **gold
   completions**, so Q1 vs Q2 isolates the prefix source. Confirm this over
   arbitrary fixed text.
2. **Loss temperature.** `τ_loss = 1.0` for all KL objectives; rollout sampling
   at `τ = 0.6`. Confirm (the alternative is `τ = 0.6` for the loss too).
3. **Forward-KL support.** Full-vocabulary KL with hidden-state chunking;
   top-k teacher truncation would be cheaper but is a different baseline.
   Confirm full vocabulary.
4. **Sampled RKL (Q3).** Pure surrogate with no CE anchor, and run only after
   Q1/Q2 as a separate decision. Confirm the sequencing.
5. **0.6B generative suite.** GSM8K + MMLU-Redux + MATH-500, or drop MATH
   (answer-matching fragility) and add IFEval?
6. **Seeds.** One seed everywhere with a "preliminary" label, or a second seed
   only for Q1/Q2?
7. **Data access.** Confirm OpenThoughts + FineWeb-Edu licenses and whether
   MMLU-Redux / GPQA-D gated access is available.
8. **Scope claim.** With Stage-1 initialization excluded, is "on-policy QAD
   recovery from pure RTN at W3" strong enough for the trial, or is a stronger
   initialization needed? The Gate 1.5 diagnosis (W3 ppl 754 vs W4 ppl 30.7)
   suggests W3 RTN leaves little recoverable margin.
9. **QAD initialization.** Gate 2 confirms the Gate 1.5 ordering on CUDA:
   W3-G128 ppl 724, W3-G32 94.4, W4-G128 30.6, BF16 22.3, and W3-G128
   generation is degenerate repetition (4-gram 0.965 vs 0.047 BF16). Recommend
   **W4A16-G128 as the pipeline-verification arm** (offline Q1 first), with
   W3-G128 retained as the aggressive baseline and W3-G32 available if W3 codes
   are required. Confirm before Stage 2 spending.

---

## 14. Next steps after the pilot

The pilot is complete. The highest-value follow-ups, in order:

1. **Replicate Q1-200 vs Q2-200** with a second seed; the GSM8K 28% vs 14% gap
   is large but single-run.
2. **Rollout-length ablation** for Q2 at fixed token budget (e.g. 4 prompts x
   512 tokens x 25 steps) to test whether 128-token rollouts cause the GSM8K
   collapse while held-out NLL stays flat.
3. **W3 initialization** remains open: either a stronger init (GPTQ/AWQ or
   block-wise QAT) or accept W3-G128 as an unrecoverable baseline; do not
   silently substitute.
4. Only then consider 1.7B, label corruption, or packed INT4 export.

Artifacts are in `experiments/gate2`–`gate5` and `reports/findings.md`.
Checkpoints (5.3–5.6 GB each) remain on the pod and are not committed.

## 15. Audit corrections (review.md, 2026-11-15)

Status legend: **done (Mac)** = code/tests/data landed; **pending GPU** = must
run on a GPU; **blocked** = cannot be completed with available artifacts.

| Priority | Issue | Status |
|---|---|---|
| P0A/B/C | strict GSM8K parser, non-thinking and thinking evaluators, lossless token/text records | **verified at Checkpoint A**: `answer_parsing.py`, `eval_gsm8k_strict.py`, tests (Qwen3 ids 151667/151668 verified) |
| P0D/E | regrade old checkpoints / matched 25-step pair | **blocked for old checkpoints** (deleted); corrected matched seed-42 pair evaluated at Checkpoint A |
| P1 | matched Q1/Q2 schedule, tokens, context | **verified at Checkpoint A** (seed 42): `check_matched.py` passes, 0 prompt-hash mismatches, token ratio 1.0012 |
| P2 | disjoint validation vs final test | **verified at Checkpoint A**: `fineweb_val_ids.npy`, `wikitext_test_ids.npy`, zero block overlap |
| P3 | best/final checkpoint selection with reload | **verified at Checkpoint A**: best step 150 for both arms, best/final saved and hashed |
| P4 | durable checkpoints | **partial**: Q1/Q2 best portable `.safetensors` copied to `checkpoints/gate8/`; `.pt` best/final on pod; old checkpoints lost |
| P5 | API-correctness tests (sampling uses current weights, modes, masks, EOS) | **done (Mac)**: `test_generation_correctness.py`, existing training/freeze tests |
| P6 | realistic memory accounting | **done (Mac)**: `reports/memory_accounting.md`; no packed artifact claimed |
| P7 | statistics | **done (Mac) in evaluator**: bootstrap CI, paired difference CI, McNemar |
| P8 | updated report | **pending reviewer approval**: Checkpoint A report written; old GSM numbers labelled historical |

Old artifacts (`experiments/gate2`–`gate7`, old `reports/findings.md` tables) are
preserved and labelled historical. New corrected runs use `experiments/gate8`.

**Pause point:** Checkpoint A (`reports/checkpoint_a.md`) is ready for reviewer
verification. Seed 43, W3 and corrected 1.7B runs have **not** been started.

---

