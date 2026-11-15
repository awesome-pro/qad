# GPU runbook — Gate 2 (official baseline + generation diagnostic + training probe)

One session, **no QAD training**. Expected ~1.5–2 h; hard timeout 4 h. Terminate
the pod as soon as the artifacts are copied back.

The four baseline configurations are evaluated on the **same full 256-block
held-out set** with identical tokenizer, precision, masks and chunk size:

| name | config | output |
|---|---|---|
| BF16 | no quantization | `experiments/gate2/t0_bf16.json` |
| W3-G128 | W3A16, group 128 (aggressive RTN baseline) | `experiments/gate2/q0_w3a16_g128.json` |
| W3-G32 | W3A16, group 32 | `experiments/gate2/q0_w3a16_g32.json` |
| W4-G128 | W4A16, group 128 | `experiments/gate2/q0_w4a16_g128.json` |

## What to rent

- **1× RTX 4090 24 GB** (Community Cloud is fine). Fallbacks: L40S 48 GB,
  RTX A5000 24 GB.
- **Container disk 50 GB**, **network volume 50 GB mounted at `/workspace`**
  (HF cache and artifacts survive termination).
- Template: current RunPod PyTorch image with **CUDA 12.8+ and driver ≥ 550**
  (or Ubuntu 22.04 + recent driver; PyTorch wheels bundle the CUDA runtime).
- Expose **SSH** (TCP port 22) and paste this public key:

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICXB5vPj+IPWpmcNgKQ2avP95ZzMNkceo64Nu605O4qE qad-runpod
```

Private key stays on the Mac at `.ssh/runpod_ed25519` (gitignored).

## From the Mac

```bash
./scripts/package_for_gpu.sh                 # ~8 MB bundle
scp -i .ssh/runpod_ed25519 -P <port> qad_gpu_bundle.tgz root@<host>:/workspace/
ssh -i .ssh/runpod_ed25519 -p <port> root@<host>
```

## On the pod

```bash
cd /workspace
mkdir -p qad && tar -xzf qad_gpu_bundle.tgz -C qad && cd qad
export HF_HOME=/workspace/hf
bash scripts/bootstrap_gpu.sh              # venv + torch/transformers + CUDA check
REV=c1899de289a04d12100db370d81485cdf75e47ca
mkdir -p experiments/gate2
```

### 1. Inventory + official baseline (four configs)

```bash
.venv/bin/python scripts/inspect_model.py --model Qwen/Qwen3-0.6B --revision $REV \
  --device cuda --dtype bfloat16 --out experiments/gate2/inventory_cuda.json

.venv/bin/python scripts/baseline.py --variant T0 --model Qwen/Qwen3-0.6B --revision $REV \
  --device cuda --dtype bfloat16 --out experiments/gate2/t0_bf16.json

.venv/bin/python scripts/baseline.py --variant Q0 --model Qwen/Qwen3-0.6B --revision $REV \
  --device cuda --dtype bfloat16 --bits 3 --group-size 128 \
  --out experiments/gate2/q0_w3a16_g128.json

.venv/bin/python scripts/baseline.py --variant Q0 --model Qwen/Qwen3-0.6B --revision $REV \
  --device cuda --dtype bfloat16 --bits 3 --group-size 32 \
  --out experiments/gate2/q0_w3a16_g32.json

.venv/bin/python scripts/baseline.py --variant Q0 --model Qwen/Qwen3-0.6B --revision $REV \
  --device cuda --dtype bfloat16 --bits 4 --group-size 128 \
  --out experiments/gate2/q0_w4a16_g128.json
```

### 2. Fixed-prompt generation diagnostic

Identical prompts, sampling settings and seed for all four configs:

```bash
.venv/bin/python scripts/generation_diagnostics.py --model Qwen/Qwen3-0.6B --revision $REV \
  --device cuda --dtype bfloat16 --num-prompts 8 --max-new-tokens 256 \
  --configs bf16,w3a16_g128,w3a16_g32,w4a16_g128 \
  --out experiments/gate2/generation_diagnostics.json
```

### 3. Short gradient/memory probe (no QAD)

A few AdamW steps of offline forward KL on fixed gold completions, to validate
the chunked loss, teacher freezing, gradient flow and CUDA memory before any
real training:

```bash
.venv/bin/python scripts/probe_training_step.py --model Qwen/Qwen3-0.6B --revision $REV \
  --device cuda --dtype bfloat16 --bits 3 --group-size 128 --steps 3 --max-seq-len 1024 \
  --out experiments/gate2/training_probe_w3a16_g128.json
```

Optionally repeat the probe with `--bits 4 --out .../training_probe_w4a16_g128.json`.

### 4. Copy artifacts back and terminate

```bash
mkdir -p /workspace/artifacts/gate2
cp experiments/gate2/*.json experiments/gate2/*.md /workspace/artifacts/gate2/ 2>/dev/null || true
sha256sum /workspace/artifacts/gate2/*
```

Copy the JSON/MD files back to the Mac, verify hashes, then terminate the pod.

## Checks before accepting the run

- inventory totals match `experiments/gate1/inventory_qwen3_0p6b.json`;
- T0 `teacher_self_kl.two_pass_mean` is ~0 (computed, not assigned);
- NLL ordering is sane: W4 < W3-G32 < W3-G128; all finite;
- `memory.peak_bytes` is the CUDA true peak and `memory.parameter_bytes` is
  separate (never compare across processes with different models);
- generation outputs are saved with the same prompts/settings; check
  repetition, premature EOS and the examples before trusting the aggregates;
- probe assertions are all true: teacher gradients absent, student gradients
  finite, quantized forward active, master weights changed;
- every JSON has resolved revisions, quantization config, data hash, git
  commit and GPU/driver information.

## Guardrails

- **No full QAD training in this session.** The probe is a few steps only.
- Stop on OOM, non-finite loss/gradients, inventory mismatch or a failed
  assertion; do not retry blindly.
- Never use `--limit-batches` for a reported number.
- Log GPU type, VRAM, driver and GPU-seconds with every run.
- W3-G128 is the aggressive RTN baseline, not a locked QAD initialization;
  W3-G32, W4-G128 and a stronger W3 initialization remain open options.
