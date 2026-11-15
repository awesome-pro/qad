# 1.7B controlled scaling check — handoff runbook

Status: **A6000 not currently available.** Do not start until the pod is up and
the tokenizer compatibility report is re-checked on that pod.

## 0. Preconditions

- 48 GB GPU (A6000/L40S); 0.6B uses 24 GB.
- Pod has the audit commit (`fb58fdf` or later), data splits and prompts.
- Tokenizer compatibility report: `experiments/gate8/tokenizer_compat_1p7b.json`
  — **passed** locally:
  - vocab size/dict equal, chat template equal, EOS/pad equal,
  - `<think>`/`</think>`/`<|im_start|>`/`<|im_end|>` IDs equal,
  - sample encodings equal.
- Re-run `scripts/check_tokenizer_compat.py` on the pod before training.

## 1. BF16 and W4 baselines

```bash
P=.venv/bin/python
M=Qwen/Qwen3-1.7B
R=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
$P scripts/baseline.py --variant T0 --model $M --revision $R --device cuda --dtype bfloat16 \
  --eval data/cache/eval/fineweb_val_ids.npy --out experiments/gate8/val_1p7b_t0.json
$P scripts/baseline.py --variant Q0 --model $M --revision $R --device cuda --dtype bfloat16 \
  --bits 4 --group-size 128 --eval data/cache/eval/fineweb_val_ids.npy \
  --out experiments/gate8/val_1p7b_q0_w4.json
$P scripts/baseline.py --variant T0 --model $M --revision $R --device cuda --dtype bfloat16 \
  --eval data/cache/eval/wikitext_test_ids.npy --out experiments/gate8/test_1p7b_t0.json
$P scripts/baseline.py --variant Q0 --model $M --revision $R --device cuda --dtype bfloat16 \
  --bits 4 --group-size 128 --eval data/cache/eval/wikitext_test_ids.npy \
  --out experiments/gate8/test_1p7b_q0_w4.json
```

## 2. Short memory/gradient probe (mandatory gate)

```bash
$P scripts/probe_training_step.py --model $M --revision $R --device cuda \
  --dtype bfloat16 --bits 4 --group-size 128 --steps 3 --max-seq-len 1536 \
  --out experiments/gate8/probe_1p7b_w4.json
```

Proceed only if: finite loss/gradients, teacher gradients absent, quantized
forward active, peak CUDA memory < 45 GB, and checkpoint save succeeds.

## 3. Corrected offline Q1 (only after probe passes)

```bash
$P scripts/train_matched.py --arm q1 --model $M --revision $R --device cuda \
  --dtype bfloat16 --bits 4 --group-size 128 --steps 200 --prompts-per-step 32 \
  --continuation-cap 128 --max-seq-len 1536 --seed 42 \
  --val-set data/cache/eval/fineweb_val_ids.npy \
  --run-name m_q1_w4_1p7b --out-dir experiments/gate8 --checkpoint-dir checkpoints \
  --no-final-optimizer
$P scripts/baseline.py --variant Q0 --checkpoint checkpoints/m_q1_w4_1p7b_best.pt \
  --bits 4 --group-size 128 --model $M --revision $R --device cuda --dtype bfloat16 \
  --eval data/cache/eval/wikitext_test_ids.npy --out experiments/gate8/test_m_q1_w4_1p7b_verified.json
$P scripts/eval_gsm8k_strict.py --model $M --revision $R --config-name m_q1_w4_1p7b \
  --mode nonthinking --bits 4 --checkpoint checkpoints/m_q1_w4_1p7b_best.pt --count 100 \
  --out-prefix experiments/gate8/gsm8k_m_q1_w4_1p7b_nonthink
```

Use parser **v3.0.0** for grading; regrade with `scripts/regrade_gsm8k.py`.
Select best by validation NLL only; report val and test separately.

## 4. Do not run

- No 1.7B online Q2 in this handoff.
- No MMLU/MATH/broad sweep.
- Do not compare 1.7B against 0.6B as a controlled scaling law without noting
  the differing max sequence length and single seed.

## 5. Stop conditions

OOM, non-finite loss/gradient, tokenizer mismatch, probe >45 GB, or failed
checkpoint save: stop and report; do not retry blindly.
