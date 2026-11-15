# GPU runbook — audit corrections (review.md)

**No pod is currently running.** This is the corrected protocol to execute once
a GPU is available. Reuse the pinned model/data; the old checkpoints are gone
and must be retrained (they were deleted after evaluation). Do not overwrite
`experiments/gate2`–`gate7`; write to `gate8`/`gate9`.

## 0. Environment

- 0.6B work: 24 GB card (A5000/4090) is enough.
- 1.7B work: 48 GB card (A6000/L40S).
- Clone `https://github.com/awesome-pro/qad` at the audit commit, set
  `HF_HOME=/workspace/hf`, `python3 -m venv --system-site-packages .venv`,
  install `-e ".[dev]" datasets`, copy `data/`.
- Use **direct TCP SSH**; the `ssh.runpod.io` proxy is interactive-only.

## 1. Data splits (already prepared on the Mac)

```text
data/cache/eval/fineweb_val_ids.npy      128 x 1024 validation blocks
data/cache/eval/wikitext_test_ids.npy    128 x 1024 final-test blocks
data/manifests/gate8_splits_manifest.json
```

Validation and test are disjoint (block-hash intersection 0). Use
`fineweb_val_ids.npy` **only** for checkpoint selection; report final metrics on
`wikitext_test_ids.npy`.

## 2. Strict GSM8K, non-thinking (run first)

```bash
P=.venv/bin/python
R=c1899de289a04d12100db370d81485cdf75e47ca
$P scripts/eval_gsm8k_strict.py --config-name t0_bf16 --mode nonthinking \
  --count 100 --out-prefix experiments/gate8/gsm8k_t0_nonthink
$P scripts/eval_gsm8k_strict.py --config-name q0_w4a16_g128 --mode nonthinking \
  --bits 4 --group-size 128 --count 100 --out-prefix experiments/gate8/gsm8k_q0_w4_nonthink
```

Only explicit final answers count; `truncation_rate`, `completion_rate`,
`parse_status_counts` are reported separately. Keep the 100-question subset
fixed (`--offset 0 --count 100`).

## 3. Matched Q1/Q2, 200 steps (0.6B W4)

Both arms share prompt schedule, prompt IDs/step, continuation cap, steps, LR,
teacher, quantizer and seed. New scripts only; old results stay historical.

```bash
for SEED in 42 43; do
  $P scripts/train_matched.py --arm q1 --steps 200 --prompts-per-step 32 \
    --continuation-cap 128 --max-seq-len 2048 --seed $SEED \
    --val-set data/cache/eval/fineweb_val_ids.npy \
    --run-name m_q1_w4_s${SEED} --out-dir experiments/gate8 --checkpoint-dir checkpoints
  $P scripts/train_matched.py --arm q2 --steps 200 --prompts-per-step 32 \
    --continuation-cap 128 --max-seq-len 2048 --seed $SEED \
    --val-set data/cache/eval/fineweb_val_ids.npy \
    --run-name m_q2_w4_s${SEED} --out-dir experiments/gate8 --checkpoint-dir checkpoints
  $P scripts/check_matched.py --q1 experiments/gate8/m_q1_w4_s${SEED}.json \
    --q2 experiments/gate8/m_q2_w4_s${SEED}.json \
    --out experiments/gate8/match_check_s${SEED}.json
done
```

Q1 and Q2 run sequentially; Q2 needs fresh rollouts each step. Each run saves
`*_best.pt` / `*_best.safetensors` selected on validation NLL, and `*_final.*`.
Best and final must be reported separately. Tokens/step will differ slightly;
the match check requires <5%.

## 4. Full held-out evaluation (validation and final test separately)

```bash
for ARM in q1 q2; do for SEED in 42 43; do
  CK=checkpoints/m_${ARM}_w4_s${SEED}_best.pt
  $P scripts/baseline.py --variant Q0 --checkpoint $CK --bits 4 --group-size 128 \
    --eval data/cache/eval/wikitext_test_ids.npy \
    --out experiments/gate8/test_${ARM}_s${SEED}.json
done; done
# T0 and Q0 on the same final test
$P scripts/baseline.py --variant T0 --eval data/cache/eval/wikitext_test_ids.npy \
  --out experiments/gate8/test_t0.json
$P scripts/baseline.py --variant Q0 --bits 4 --evals data/cache/eval/wikitext_test_ids.npy \
  --out experiments/gate8/test_q0_w4.json
```

(`--evals` is a typo guard: use `--eval`.)

## 5. Strict GSM8K, non-thinking, corrected checkpoints

```bash
$P scripts/eval_gsm8k_strict.py --config-name q1_w4_best --mode nonthinking \
  --bits 4 --checkpoint checkpoints/m_q1_w4_s42_best.pt --count 100 \
  --out-prefix experiments/gate8/gsm8k_q1_w4_nonthink
$P scripts/eval_gsm8k_strict.py --config-name q2_w4_best --mode nonthinking \
  --bits 4 --checkpoint checkpoints/m_q2_w4_s42_best.pt --count 100 \
  --out-prefix experiments/gate8/gsm8k_q2_w4_nonthink
```

Then paired comparison on the same questions:

```bash
$P scripts/compare_gsm8k.py \
  --q1 experiments/gate8/gsm8k_q1_w4_nonthink.generations.jsonl \
  --q2 experiments/gate8/gsm8k_q2_w4_nonthink.generations.jsonl \
  --out experiments/gate8/gsm8k_q1_vs_q2_nonthink.json
```

The output reports paired bootstrap CI and exact McNemar counts.

## 6. Strict GSM8K, thinking subset

Use a small fixed subset to keep the budget controlled:

```bash
$P scripts/eval_gsm8k_strict.py --config-name t0_bf16 --mode thinking \
  --count 20 --max-new-tokens 2048 --generation-batch-size 8 \
  --out-prefix experiments/gate8/gsm8k_t0_think20
$P scripts/eval_gsm8k_strict.py --config-name q1_w4_best --mode thinking \
  --bits 4 --checkpoint checkpoints/m_q1_w4_s42_best.pt --count 20 \
  --max-new-tokens 2048 --generation-batch-size 8 \
  --out-prefix experiments/gate8/gsm8k_q1_w4_think20
```

Report completion/truncation/thinking-closed rates; unfinished reasoning is
never graded.

## 7. W3 QAD (if budget permits)

```bash
$P scripts/train_matched.py --arm q1 --bits 3 --group-size 128 \
  --steps 200 --prompts-per-step 32 --continuation-cap 128 --max-seq-len 2048 \
  --seed 42 --val-set data/cache/eval/fineweb_val_ids.npy \
  --run-name m_q1_w3_s42 --out-dir experiments/gate8 --checkpoint-dir checkpoints
$P scripts/eval_gsm8k_strict.py --config-name q1_w3_best --mode nonthinking \
  --bits 3 --checkpoint checkpoints/m_q1_w3_s42_best.pt --count 100 \
  --out-prefix experiments/gate8/gsm8k_q1_w3_nonthink
```

## 8. Exploratory 1.7B (48 GB pod)

```bash
M=Qwen/Qwen3-1.7B
R1=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
$P scripts/baseline.py --variant T0 --model $M --revision $R1 \
  --eval data/cache/eval/wikitext_test_ids.npy --out experiments/gate8/test_1p7b_t0.json
$P scripts/baseline.py --variant Q0 --model $M --revision $R1 --bits 4 --group-size 128 \
  --eval data/cache/eval/wikitext_test_ids.npy --out experiments/gate8/test_1p7b_q0_w4.json
$P scripts/train_matched.py --arm q1 --model $M --revision $R1 --bits 4 --group-size 128 \
  --steps 200 --prompts-per-step 32 --continuation-cap 128 --max-seq-len 1536 \
  --seed 42 --val-set data/cache/eval/fineweb_val_ids.npy \
  --run-name m_q1_w4_1p7b --out-dir experiments/gate8 --checkpoint-dir checkpoints
```

Label the 1.7B result exploratory: Q2 is not run at a matched budget.

## 9. Preserve artifacts before termination

```bash
sha256sum experiments/gate8/*.json experiments/gate8/*.jsonl \
  checkpoints/*.safetensors > experiments/gate8/SHA256SUMS
```

Copy the JSON/JSONL/generations and **safetensors inference weights** to durable
storage (Mac or object store). Optimizer-state `.pt` files are optional for
resumption and are large. Do not commit multi-GB files to Git.

## 10. Stop conditions

- Any OOM, non-finite loss/gradient, `check_matched.py` mismatch, or
  truncation-rate anomaly: stop and report; do not retry blindly.
- Do not use the final test set for checkpoint selection or hyperparameters.
- Do not claim packed-inference memory savings; no packed artifact is produced.
