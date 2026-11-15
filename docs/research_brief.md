# Qwen Low-Bit Quantization & Online QAD — Research and Engineering Brief

---

## 1. Operating instructions

Read this file as a **research specification**, not a mandate to generate a large codebase immediately.

1. **First inspect this brief, verify relevant library APIs on the intended environment, and present an implementation plan** with dependencies, modules, experiments, tests, compute/memory risks, and unresolved decisions. Ask for review before making substantial architecture changes.
2. Implement **small, reviewable increments**; show the researcher what changed, how to run it, what tests passed, and any scientific assumptions.
3. Prefer the smallest working, testable system over ambitious abstractions. Use established PyTorch/Hugging Face interfaces unless custom functionality is required for the research question.
4. **Never launch a paid GPU instance, long training run, model download of unexpected size, or broad parameter sweep without explicit approval.** Give a cost/VRAM estimate or a dry-run command first. Stop on OOMs or runaway cost rather than retrying blindly.
5. Preserve an **experiment log** (commands, git commit, dependency versions, model/dataset revisions, seed, device, precision, configuration, loss settings, observed metrics, failures).
6. Distinguish **measured** from **estimated**, **implemented** from **proposed**, and **validated** from **unvalidated**. No invented metrics or fabricated baselines.
7. Use a coding agent for implementation efficiency, but keep the quantizer, training-loss semantics, masking, gradients, checkpoint/export, and evaluation logic understandable and auditable by the researcher.
8. Keep credentials out of source, logs, reports, and checkpoints. Respect dataset/model licenses and any private research boundaries.
9. Do not refactor into a framework or build custom GPU kernels until a measurement demonstrates a concrete need.
10. If requirements are ambiguous, **surface the decision and propose a reversible default**, especially for online-KL direction, bit format, module coverage, and checkpoint export.

**Agent's first response should be:** a succinct proposed repository layout, milestone gates, technical choices with reasons, anticipated resource usage, and the *first small implementation diff*. **Do not start by writing the full project.**

---

## 2. Research scope and priorities

### Primary — reproducible baseline and offline QAD

- Select a real pretrained dense Qwen causal LM and pin its exact model revision/tokenizer.
- Establish **BF16 teacher baseline** on a fixed validation/test suite.
- Introduce **W3A16 weight-only quantization** (initial exploratory default) with an explicitly defined low-bit representation, e.g. signed symmetric per-output-row groups of 128 weights.
- Compare the untrained quantized student (PTQ/round-to-nearest baseline) to the teacher.
- Train the quantized student with **fake quantization + straight-through estimator (STE)** using **frozen-teacher distillation** on fixed text prefixes.
- Compare post-training quality using **exactly the same** evaluation pipeline and held-out data.
- Track theoretical effective packed weight storage, actual training GPU memory, and uncertainty about deployment memory.

### Secondary — online/on-policy QAD

- Have the **current quantized student** generate rollouts from fixed prompt sources.
- Evaluate **frozen teacher and trainable student on exactly the same visited prefixes**.
- Update the student using a clearly specified distillation objective and masking.
- Regenerate rollouts after the student changes; distinguish this from repeatedly replaying a frozen corpus of student outputs.
- Compare online to offline under as closely matched an optimization/evaluation budget as is practical.

### Optional explorations (only after the core pipeline works)

- W2A16, W4A16, alternative group sizes, per-channel vs per-group scales.
- GPTQ/AWQ or blockwise-QAT initialization; scale learning/clipping; mixed precision for sensitive modules.
- Forward-KL vs reverse-KL/sampled on-policy objectives, temperature, different rollout sampling settings.
- Real packed checkpoint export, reload, memory measurement, and backend-specific inference benchmarks.

### Deliberate non-goals for the initial pilot

- Reimplementing full GPTQ, AWQ, SGLang, vLLM, or CUDA GEMM kernels.
- Claiming speedups from fake quantization.
- Reproducing a large published benchmark suite at full scale.
- An open-weight public release before the checkpoint, scientific scope, validation, and publication permission are agreed with Lev.
- Treating the example bit format or model size as Tear Labs' actual deployment choice.

---

## 3. Starting defaults, with explicit decision points

| Dimension | Pilot default (changeable) | Why / caveat |
|---|---|---|
| Model | `Qwen/Qwen3-0.6B` | Cheap real causal LM for pipeline correctness. **Not** the agreed Tear Labs checkpoint. |
| Follow-on candidate | `Qwen/Qwen3-1.7B` | Directly relevant to published ultra-low-bit on-policy QAT experiments; requires a fresh resource estimate. |
| Precision | W3A16 (weight-only) | Aggressive but less fragile than 2-bit. Here A16 means FP16/BF16 activations, not KV-cache precision. |
| Weights | Signed symmetric uniform quantizer, `q ∈ [-3, 3]`, with one signed 3-bit code unused | Simple, explicit pilot scheme; **not the only possible INT3 definition**. |
| Grouping | Group size 128 along input dimension, independent per output row | Scales must be tested for shape/broadcasting; handle tail groups or explicitly skip incompatible modules. |
| Scales | Max-absolute group scale, with safe floor; initially no learned zero-point | Simple baseline; frozen/detached scale choice must be documented. |
| Modules | Start with selected linear projection weights; leave embeddings, normalization, and LM head unquantized | Simplifies debugging; report coverage and true effective model bits/weight. |
| Student initialization | Copy pretrained weights exactly, then enable fake quantization | Isolates quantization-induced degradation. |
| Teacher | Same checkpoint in BF16, `eval()` and fully frozen | All teacher outputs calculated without autograd. |
| Offline loss | Forward KL `KL(teacher || student)` over valid next-token positions | Easiest auditable baseline; **not necessarily the loss used by an online-QAT paper**. |
| Online loss | First implement same KL on student-generated prefixes for controlled comparison; separately evaluate sampled reverse-KL if reproducing a paper | Loss direction is a research variable, not an implementation detail. |
| Data | A small legally usable train corpus; separate fixed validation/test texts and rollout prompts | Pin exact source, split, cleaning, license, tokenization. No leakage. |
| Hardware | Approved RunPod CUDA GPU; choose using live availability/cost and peak-memory estimate | Don't hardcode GPU costs/specs. |

**Compatibility warning:** NVIDIA Model Optimizer offers QAT/QAD examples, including W4A16 INT4 weight-only and NVFP4 recipes, but a supplied **NVFP4 QAD recipe is not equivalent to symmetric W3A16/W2A16**. Verify actual format and backend support before reusing APIs or claiming equivalence. `torchao` QAT is another reference, but exact INT3 support/export should be verified rather than assumed.

---

## 4. Quantization specification and numerical invariants

For a linear weight matrix `W` with PyTorch layout `[out_features, in_features]`, group adjacent input features separately **within each output row**.

For signed symmetric `b`-bit weights in the pilot:

```text
qmax  = 2^(b - 1) - 1
qmin  = -qmax
scale = max(abs(W_group)) / qmax  (subject to numerical floor)
q     = clamp(round(W_group / scale), qmin, qmax)
W_hat = q * scale
```

For W3, `qmin=-3`, `qmax=3`. For W2 under the same convention, `qmin=-1`, `qmax=1`. These conventions leave one signed code unused. A different asymmetric or full-range scheme may yield different quality and storage/metadata requirements; **never silently switch schemes**.

### Must-have correctness properties

- Exact original and quantized shapes; correct per-row/per-group broadcasting; no cross-row grouping.
- Finite scales with all-zero groups; explicit behavior for outliers and clipping.
- Quantized integer codes lie in supported range; reconstruction uses the declared scale.
- Fake-quantized forward values equal quantize-dequantize values to numerical tolerance.
- High-precision trainable **master weights** are not overwritten in place by quantized weights.
- STE actually passes nonzero, finite gradients to intended student master weights; no teacher gradients.
- Quantization state/configuration survives save and reload; reloaded student produces matching outputs on a fixed test batch.
- Any excluded modules and any padding/tail-group policy are enumerated in the report.

### Memory accounting — three separate numbers

1. **Raw weight payload**: `sum(number_of_quantized_weights * bits / 8)`.
2. **Theoretical packed checkpoint**: payload + scale/zero-point metadata + unquantized parameters + alignment/packing overhead (where applicable). State if packing overhead is unknown.
3. **Measured runtime/training memory**: device peak memory measured from a real run. Fake-quantized tensors and optimizer state may use full precision and **do not realize theoretical packed-weight VRAM savings**.

For symmetric group quantization with one BF16 scale per `G` weights, approximate effective bits per quantized weight: `b + 16/G`, before padding and other metadata. Use parameter-weighted accounting across all included/excluded modules.

A packing/execution benchmark requires an actual supported packed representation and compatible kernel. Do **not** report 3-bit compression ratio as a 3-bit inference speedup.

---

## 5. Scientific evaluation design

**Core comparisons** (use the same model revision, tokenizer, data, and held-out evaluation):

| ID | System | Purpose |
|---|---|---|
| T0 | BF16 original teacher | Upper reference for this exact architecture/checkpoint |
| Q0 | Fake-quantized W3A16 student before training | Round-to-nearest PTQ reference |
| Q1 | Q0 + offline QAD | Quantization-aware recovery on fixed prefixes |
| Q2 | Q0 + online/on-policy QAD | Recovery on student-visited prefixes; optional if correctly implemented |
| Q3 | Alternative bit width or improved initialization | Extension after Q0/Q1 validity |

**Metrics (always label the dataset, denominator, and measurement method):**

- **Held-out causal language-model NLL** (and perplexity where meaningful) with correct shifted next-token labels and masking.
- **Teacher–student KL** on the same *fixed held-out prefixes*; explicitly identify forward/reverse direction and reduction; teacher self-KL should be ~0 within numerical tolerance.
- **Generative task quality** on a compact, fixed prompt suite: task accuracy when objectively measurable, output examples, and any unsupported/truncated/repetitive responses. Treat subjective quality as anecdotal.
- **Quantization diagnostics**: weight reconstruction MSE by module/group, selected layer-output reconstruction, scale distribution, saturation/zero fraction, quantized-code changes, and excluded modules.
- **Memory**: raw/theoretical bit accounting and measured GPU memory; distinguish allocated, reserved, peak, checkpoint disk, and runtime packed storage.
- **Training cost/performance**: GPU type, peak VRAM, total GPU-seconds or number of processed tokens, and throughput (only if measured reliably).
- **Reproducibility**: at least one replicated run or repeated validation where feasible. Negative and inconclusive results are valid findings.

### Dataset and evaluation policy

- Use **legally permitted** data with documented provenance and licenses; no private data, credentials, or unseen test benchmark answers in training.
- Fix train, validation, and test splits by stable identifiers and hashes. Do not tune on test.
- Record Qwen chat formatting, BOS/EOS behavior, special tokens, and thinking/non-thinking mode where applicable; ensure the same conditions across baselines.
- Use a meaningful fixed held-out evaluation set, not training prefixes. Small pilot results should be described as preliminary.
- For online experiments, log **train-trajectory KL** separately from KL on a *fixed* held-out evaluation set: rollout distributions change over training, so training KL curves across runs/steps are not necessarily directly comparable.
- Report both absolute differences and relative differences; avoid asserting statistical significance from a single noisy run.

### Causal LM logit alignment

For input tokens `ids[:, :T]`, logit at position `t` predicts token `t+1`. Standard next-token NLL compares `logits[:, :-1, :]` with `ids[:, 1:]`, then applies the valid-token / training-target mask. Padding, EOS, and prompt masking must be handled consistently.

For **on-policy rollouts** with prompt length `P` and generated continuation length `R`, the decision distribution for the first generated token comes from model logit index `P-1`; the distribution for generated token `j` (0-indexed) comes from logit index `P-1+j`. If only generated-token decisions are scored, use indices `P-1` through `P+R-2` (inclusive), with per-example EOS/padding masks. Do not accidentally train on the logit after the final generated token unless deliberately included. Add a tiny deterministic unit test for these offsets.

---

## 6. Implementation work packages and acceptance gates (no timelines)

These gates describe **dependencies and evidence**, not dates or delivery promises.

### Gate A — Environment, model loading, and baseline

**Build:** reproducible environment and deterministic baseline CLI; load tokenizer/model; read and record architecture/configuration; enumerate named modules and parameter shapes; print quantization-eligible modules; run BF16 NLL/KL sanity checks and fixed-prompt generations.

**Acceptance:** exact model ID/revision, license and config recorded; forward logits shape `[B,T,V]`; deterministic evaluation on fixed inputs; teacher self-KL approximately zero; baseline metrics persisted to file.

### Gate B — Standalone quantizer and isolated tests

**Build:** pure PyTorch reference quantize/dequantize/fake-quantize functions, per-tensor/per-row/per-group options for testing, W3 chosen default, configurable scale policy, STE, numerical diagnostics.

**Acceptance:** unit tests pass for group counts, extreme values, all-zero groups, negative weights, shape restoration, gradient behavior, reconstruction matching, and serialization. Keep a small mathematical reference test independent of the implementation.

### Gate C — Quantized Qwen PTQ baseline

**Build:** selectively wrap/replace eligible linear layers *without changing the pretrained master weights*; expose config for module inclusion/exclusion; quantize student forward; evaluate frozen initial student; log effective coverage and expected memory.

**Acceptance:** same-shape logits and finite loss; nonzero quantization error in selected weights; excluded modules unchanged; Q0 evaluated on precisely T0's held-out set; restart/reload reproduces Q0.

### Gate D — Offline QAD

**Build:** frozen BF16 teacher, QAT student initialized from teacher, correct teacher/student next-token KL loss, optimizer acting on student high-precision weights, fixed-prefix training data, gradient checkpointing or memory controls as necessary, checkpoint save/resume, fixed validation.

**Acceptance:** teacher parameters unchanged with `.grad is None`; student gradients finite; low-bit forward stays enabled in both train and eval; teacher/student alignment/masks validated; initial Q0 and post-training Q1 compared fairly; best validation checkpoint reloads and reproduces results.

**Training notes:** `zero_grad()` then forward/loss then `backward()` then `optimizer.step()`; ensure no gradients flow through teacher. Decide and document whether STE is identity throughout or clipped. A `detach()` scale may be appropriate for a baseline, but is not the only possible QAT method. Teacher probabilities/student log-probabilities may be computed in FP32 for numerical stability. Full-vocabulary `[B,T,V]` KL can be memory-intensive: inspect shape/cost and implement safe chunking or smaller microbatches before increasing sequence length.

### Gate E — Online/on-policy QAD

**Build:** prompt-source sampler; current student generates rollouts **without autograd**; preserve prompt/continuation boundaries and EOS masks; frozen teacher scores the student-visited prefixes; **recompute the student forward with autograd on those same tokens**; compute the stated KL objective; update student; generate new rollouts after updates; log rollout versions/configs.

**Acceptance:** deterministic tiny-model test checks prefix/logit indices and masks; teacher/student score identical token sequences, not independently generated teacher continuations; no gradient path through discrete token sampling; gradients do reach quantized student via its re-evaluation; rollouts genuinely refresh with the current student; fixed held-out evaluation compares Q1 and Q2.

**Research caveat:** the published *OnlineQAT* method uses a **sampled reverse-KL** signal with its own recipe. A first implementation that uses `KL(teacher || student)` on student prefixes demonstrates **on-policy data collection** but is **not an exact reproduction** of that paper. Label the algorithm accurately and implement/verify the paper-specific objective only as a separate variant.

### Gate F — Validation, reporting, and optional export

**Build:** aggregate plots/tables; record failures and uncertainties; clean one-command repro (environment, dataset preparation, baseline, train, evaluate); optional packed low-bit export if a compatible representation/backend can be verified; open-weight model card draft only if release is authorized.

**Acceptance:** the report identifies T0/Q0/Q1/(Q2), exact revisions/configs, metrics, memory method, unquantized parts, seeds, known limitations, and executable commands. Never mark fake-quant weight savings as measured deployed savings. Never claim an open-weight artifact if only an FP16/BF16 master checkpoint with fake quantization was saved.

---

## 7. Proposed repository structure

This is a **suggestion** for the agent to refine (do not create empty boilerplate files merely to match the tree):

```text
qwen-lowbit-qad/
├── README.md                     # Scientific question, how to reproduce, findings
├── RESEARCH_BRIEF.md             # This specification
├── pyproject.toml                # Pinned dependencies / environment instructions
├── configs/
│   ├── pilot.yaml                # Model, quantizer, data, evaluation defaults
│   ├── offline_qad.yaml          # Optimizer + KL + training settings
│   └── online_qad.yaml           # Rollout + on-policy loss settings
├── src/qwen_qad/
│   ├── models.py                 # Model loading, eligible layer inspection
│   ├── quantization.py           # Reference quantizer, STE, module wrappers
│   ├── data.py                   # Source, tokenization, splits, masks
│   ├── losses.py                 # Full-vocab KL / masked NLL
│   ├── evaluate.py               # Teacher/student evaluation and memory report
│   ├── offline_train.py          # Offline QAD loop
│   ├── rollouts.py               # Student-generated sequences with metadata
│   ├── online_train.py           # On-policy training; optional gate
│   ├── checkpointing.py          # Save/load model, quantizer config, optimizer
│   └── diagnostics.py            # Quant code, scale, gradient stats
├── scripts/
│   ├── baseline.py
│   ├── quantize_eval.py
│   ├── train_offline.py
│   ├── train_online.py
│   └── report.py
├── tests/
│   ├── test_quantizer.py
│   ├── test_ste_gradients.py
│   ├── test_masks_alignment.py
│   ├── test_teacher_freeze.py
│   ├── test_checkpoint_roundtrip.py
│   └── test_online_rollout.py
├── experiments/
│   └── index.md                  # Experiment registry; logs not hard-coded
└── reports/
    └── findings_template.md
```

Prefer simple `argparse` or a modest config loader; no database, distributed controller, serving engine, or observability stack is required for the pilot.

---

## 8. Experiment configuration and reproducibility contract

Every experiment should have a machine-readable configuration equivalent to the following **schema sketch** (illustrative values; agent must validate actual configuration support):

```yaml
model:
  id: Qwen/Qwen3-0.6B
  revision: <pin_exact_commit>
  tokenizer_revision: <pin_exact_commit>
  teacher_dtype: bfloat16
  mode: <document_chat_and_thinking_mode>
quantization:
  scheme: symmetric_uniform_weight_only
  bits: 3
  group_size: 128
  axis: per_output_row_groups_along_input
  qmin: -3
  qmax: 3
  scale_policy: max_abs_detached
  include_modules: <explicit_patterns>
  exclude_modules: <explicit_patterns>
  ste: identity_or_clipped_explicitly
training:
  mode: offline_qad
  loss: forward_kl_teacher_to_student
  temperature: 1.0
  optimizer: adamw
  learning_rate: <chosen_value>
  microbatch_size: <memory_checked_value>
  sequence_length: <memory_checked_value>
  random_seed: 42
data:
  source: <licensed_documented_source>
  source_revision: <pin_or_hash>
  train_split: <stable_split>
  validation_split: <stable_split>
  test_split: <held_out_split>
evaluation:
  masks: valid_next_token_positions
  fixed_eval_suite: <manifest_and_hash>
  metrics: [nll, perplexity, teacher_forward_kl, quantization_diagnostics, memory]
```

**Run manifest minimum:** UTC timestamp of execution (a log, not a project deadline), git commit, config hash, `torch`/`transformers`/CUDA versions, GPU model/VRAM, model and dataset revisions, number of train/eval tokens, seed, exact CLI command, generated artifact paths, and warnings. Run a validation pass after restoring the best saved checkpoint.

Use stable output directories and include reproducible summary JSON/CSV. Do not check in large checkpoints or copyrighted/private datasets. Maintain a short Git-tracked experiment index linking metrics to configs and checkpoints.

---

## 9. Test plan: scientific bugs to catch early

| Test | Failure it prevents |
|---|---|
| Group-scale shape/count on `[out,in]` | Confusing batch/sequence count with weight-group count |
| Hand-computed quantization with exact zeros and negatives | Incorrect scale, clamp, sign, rounding convention |
| Fake forward matches explicit quantize/dequantize | Claiming a quantized forward that actually uses BF16 weights |
| STE gradient is nonzero on chosen in-range examples | `torch.round()` silently blocking learning |
| Teacher remains byte-identical and gradient-free | Teacher contamination or optimizer including teacher |
| Student weight changes only after optimizer step | Misunderstood autograd or accidental in-place modification |
| Train/eval quantization remains enabled | Accidentally evaluating an unquantized student |
| Causal next-token shift/mask on a tiny example | Off-by-one token scoring or padding leakage |
| Online token index test with prompt + EOS | Teacher scores a different prefix or wrong continuation position |
| Saved/reloaded model produces matching logits | Lost quantization metadata or changed module coverage |
| No train/validation overlap by hashes | Inflated claims from leaked data |
| Stable evaluation on fixed seed/split | Mistaking sampling noise for quality improvement |
| Measured vs estimated memory fields separated | Claiming fake-quant VRAM gains as deployment savings |

Treat failing test or nonfinite loss as a **stop condition** for expensive training.

---

## 10. Report template / definition of a useful result

The final short write-up should contain:

1. **Question and hypothesis:** what specifically was tested, with prior art acknowledged.
2. **Setup:** checkpoint/revision, data provenance, quantization configuration, layer coverage, QAD objective, optimizer, GPU, training token count.
3. **Results:** clearly labeled table for T0/Q0/Q1/Q2 on **identical** held-out evaluation; baseline vs best trained checkpoint and uncertainty.
4. **Memory:** theoretical quantized payload, metadata, exclusions, packed export status, measured training/serving peak memory where available.
5. **Diagnostics:** gradients, quantization code/scales, loss curves, failure modes, examples, checkpoint sanity checks.
6. **Interpretation:** what appears to work, what doesn't, what was surprising, and alternatives not yet ruled out.
7. **Reproduction:** commands, commits, configs, dependencies, data hashes, artifact locations, and limitations.
8. **Follow-up hypotheses:** 2–3 high-value discriminating experiments to propose to Lev, not an exhaustive wishlist.

Suggested result table (leave all entries blank until actually measured):

| Variant | Fixed eval NLL ↓ | Teacher KL ↓ | Quality benchmark | Estimated packed weight MiB | Actual checkpoint MiB | Peak GPU MiB | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| T0 BF16 teacher | — | 0 baseline | — | N/A | — | — | |
| Q0 W3A16 before QAD | — | — | — | — | — | — | |
| Q1 W3A16 + offline QAD | — | — | — | — | — | — | |
| Q2 W3A16 + online QAD (optional) | — | — | — | — | — | — | |
