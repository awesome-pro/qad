# GPU reproduction guide

This guide covers a fresh 24 GB or 48 GB CUDA pod. It contains no
provider-specific addresses or credentials. The 0.6B runs fit in 24 GB; the
1.7B run requires 48 GB.

## 1. Environment

```bash
git clone <repository-url> qad && cd qad
export HF_HOME=/workspace/hf
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e ".[dev]"
```

Use a PyTorch build matching the pod driver. Verify:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Data

The pinned data preparation commands are in `scripts/prepare_data.py`; the
disjoint validation/test split manifest is
`data/manifests/gate8_splits_manifest.json`. Reproduce with:

```bash
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/prepare_splits.py
```

## 3. Baselines

```bash
.venv/bin/python scripts/baseline.py --variant T0 --device cuda \
  --eval data/cache/eval/fineweb_val_ids.npy --out experiments/gate8/val_t0.json
.venv/bin/python scripts/baseline.py --variant Q0 --bits 4 --group-size 128 \
  --device cuda --eval data/cache/eval/wikitext_test_ids.npy \
  --out experiments/gate8/test_q0_w4.json
```

Use `--bits 3 --group-size 128` or `32` for W3.

## 4. Matched QAD

Both arms use the same prompt schedule, 32 prompts/step, a 128-token
continuation cap and 200 steps. Use the same `--seed` and `--val-set` for both.

```bash
# offline Q1
.venv/bin/python scripts/train_matched.py --arm q1 --bits 4 --group-size 128 \
  --steps 200 --prompts-per-step 32 --continuation-cap 128 --max-seq-len 2048 \
  --seed 42 --val-set data/cache/eval/fineweb_val_ids.npy \
  --run-name q1_w4_s42 --out-dir experiments/gate8 --checkpoint-dir checkpoints

# on-policy Q2 (same schedule; only continuation source differs)
.venv/bin/python scripts/train_matched.py --arm q2 --bits 4 --group-size 128 \
  --steps 200 --prompts-per-step 32 --continuation-cap 128 --max-seq-len 2048 \
  --seed 42 --val-set data/cache/eval/fineweb_val_ids.npy \
  --run-name q2_w4_s42 --out-dir experiments/gate8 --checkpoint-dir checkpoints

.venv/bin/python scripts/check_matched.py \
  --q1 experiments/gate8/q1_w4_s42.json \
  --q2 experiments/gate8/q2_w4_s42.json
```

`train_matched.py` saves validation-selected best and final checkpoints plus
portable inference safetensors. Back up and verify before pod cleanup:

```bash
sha256sum checkpoints/q1_w4_s42_best.* > SHA256SUMS
```

## 5. Evaluation

```bash
# held-out test
.venv/bin/python scripts/baseline.py --variant Q0 \
  --checkpoint checkpoints/q1_w4_s42_best.pt --bits 4 --group-size 128 \
  --eval data/cache/eval/wikitext_test_ids.npy --out experiments/gate8/test_q1.json

# strict GSM8K (frozen parser v3.0.0)
.venv/bin/python scripts/eval_gsm8k_strict.py --config-name q1 --mode nonthinking \
  --bits 4 --checkpoint checkpoints/q1_w4_s42_best.pt --count 100 \
  --out-prefix experiments/gate8/gsm8k_q1
```

Parser rules are frozen in `docs/parser_spec_v3.md`; do not change them without
a new version and a full regrade.

## 6. 1.7B prechecks

A 1.7B offline QAD run needs 48 GB. Before training:

```bash
# tokenizer compatibility with the 0.6B reference
.venv/bin/python scripts/check_tokenizer_compat.py --out experiments/gate8/tokenizer.json

# 3-step memory/gradient probe
.venv/bin/python scripts/probe_training_step.py --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --device cuda --dtype bfloat16 --bits 4 --group-size 128 --steps 3 \
  --max-seq-len 1536 --out experiments/gate8/probe_1p7b.json
```

Proceed only if loss/gradients are finite, the teacher has no gradients, the
quantized forward is active, and peak memory is below ~45 GB. Use
`--max-seq-len 1536` for 1.7B; use offline Q1 only unless a separate online
design is approved.

## 7. Checkpoint verification

```bash
.venv/bin/python scripts/verify_checkpoints.py \
  --checkpoint checkpoints/q1_best.pt --safetensors checkpoints/q1_best.safetensors \
  --revision <model-revision> --bits 4 --group-size 128 \
  --eval data/cache/eval/fineweb_val_ids.npy --out verify.json
```

Then copy `.pt` and `.safetensors` off the pod, compare SHA-256, and load them in
a clean process. Do not claim durability until both checks pass.
