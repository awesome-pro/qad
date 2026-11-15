# Gate 2 — official CUDA baseline, generation diagnostic and training probe

Status: **complete**, one session on an RTX A5000 24 GB (RunPod), 2026-11-15.
Artifacts are byte-identical to the pod copies; see `SHA256SUMS`.

## Environment

- GPU: `NVIDIA RTX A5000`, 24,564 MiB, driver 595.91.07
- torch 2.13.0+cu129 (template), transformers 5.23.0, Python 3.12.3
- Model: `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`
  (resolved model and tokenizer SHA identical)
- Eval: `data/cache/eval/fineweb_eval_ids.npy`, 256 × 1,024 tokens,
  261,888 scored positions, `ids_sha256 8676a899…` (same as local Gate 1)
- Inventory on CUDA matches the Mac inventory exactly:
  596,049,920 parameters, 440,401,920 quantizable, 197 `nn.Linear`
  (196 projections + tied `lm_head`), no warnings.

## Baseline (full held-out set, identical data and masks)

| config | NLL | PPL | KL(T‖S) | peak GPU | teacher/student bytes |
|---|---:|---:|---:|---:|---|
| BF16 (T0) | 3.1027 | 22.26 | 0.0000 | 2.16 GiB | 1.19 / 1.19 GB (same object) |
| W3A16-G128 | 6.5848 | 723.98 | 3.8674 | 5.73 GiB | 1.19 / 2.07 GB |
| W3A16-G32 | 4.5480 | 94.44 | 1.6693 | 5.73 GiB | 1.19 / 2.07 GB |
| W4A16-G128 | 3.4207 | 30.59 | 0.4057 | 5.73 GiB | 1.19 / 2.07 GB |

- T0 teacher self-KL from two independent passes: 0.000e+00 (computed).
- Throughput: 11.2–15.5K tokens/s for a single forward pass on this card.
- These reproduce the local MPS ordering and magnitude almost exactly.

## Generation diagnostic

Same 8 fixed prompts, `T=0.6`, `top_p=0.95`, `top_k=20`, `max_new_tokens=256`,
same seed per prompt across configs.

| config | mean 4-gram repetition | distinct-2 | premature EOS | hit 256-token cap | empty |
|---|---:|---:|---:|---:|---:|
| BF16 | 0.047 | 0.801 | 0/8 | 8/8 | 0/8 |
| W3A16-G128 | 0.965 | 0.025 | 0/8 | 8/8 | 0/8 |
| W3A16-G32 | 0.839 | 0.122 | 0/8 | 8/8 | 0/8 |
| W4A16-G128 | 0.117 | 0.697 | 0/8 | 8/8 | 0/8 |

Saved examples: BF16 and W4 produce readable reasoning; W3-G128 degenerates into
digit repetition; W3-G32 into template/phrase loops. No configuration emitted
EOS within 256 tokens, so premature-EOS rate is uninformative at this cap (all
hit the cap).

## Training probe (3 AdamW steps, offline forward KL on gold completions)

| config | losses (step 0→2) | grad norms | peak GPU | tok/s | assertions |
|---|---|---|---:|---:|---|
| W4A16-G128 | 0.589 → 0.340 → 0.374 | 48.7 → 15.2 → 31.4 | 9.32 GiB | 678 | all true |
| W3A16-G128 | 4.519 → 2.904 → 2.633 | 179.9 → 75.6 → 60.2 | 9.32 GiB | 739 | all true |

Assertions all passed for both: teacher gradients absent, student gradients
finite, quantized forward active, master weights changed, loss finite.

## Observations

1. The 24 GB card is sufficient for the 0.6B baseline and probe, with 9.32 GiB
   peak for the probe and headroom for the eventual QAD batches.
2. W3-G128 RTN is unusable as an initialization: PPL 724 vs 22, and generation
   is degenerate repetition.
3. W4-G128 is the tractable option: PPL 30.6, KL 0.41, readable generations,
   and a probe loss that decreases from 0.589.
4. W3-G32 is a middle point (PPL 94) but still repetitive; useful only if W3
   codes are required.
5. The chunked hidden-state loss and the full training-step plumbing work on
   CUDA: gradients flow to the quantized masters, the teacher is untouched, and
   the dequant cache is invalidated between steps.

## Caveats / known metadata issue

- The `memory.parameter_bytes.note` field in the Q0 JSONs is hardcoded to the
  T0 wording ("teacher and student are the same object for T0"). The numbers
  are correct; the note is wrong. Fixed in `scripts/baseline.py` after the run;
  artifacts are left untouched to preserve their hashes.
- Probe losses are not comparable across steps because each step uses a
  different gold completion and response length (277 / 535 / 151 tokens).
- Generation is stochastic with 8 prompts and one seed; the examples are
  descriptive, not a benchmark.
- Process peak memory includes teacher+student for Q0/probe and is not
  comparable to the single-model T0 number; parameter bytes are reported
  separately.

## Recommended next step

Verify the QAD pipeline on **W4A16-G128** (offline Q1 first, then on-policy Q2),
with W3A16-G128 retained as the aggressive baseline. Do not start full QAD
without a matched-budget plan and a reviewed run card.
