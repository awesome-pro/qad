# GPU runbook — Gate 2 (official T0 + Q0 at W3 and W4)

One session, **no training**. Expected ~1–1.5 h; hard timeout 3 h. Terminate the
pod as soon as the artifacts are copied back.

## What to rent

- **1× RTX 4090 24 GB** (Community Cloud is fine). Fallbacks: L40S 48 GB,
  RTX A5000 24 GB.
- **Container disk 50 GB**, **network volume 50 GB mounted at `/workspace`**
  (HF cache and artifacts survive termination).
- Template: a current RunPod PyTorch image with **CUDA 12.8+ and driver ≥ 550**
  (or plain Ubuntu 22.04 with a recent driver; PyTorch wheels bundle the CUDA
  runtime). Verify with `nvidia-smi`.
- Expose **SSH** (TCP port 22) and paste this public key:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICXB5vPj+IPWpmcNgKQ2avP95ZzMNkceo64Nu605O4qE qad-runpod
```

The private key stays on the Mac at `.ssh/runpod_ed25519` (gitignored).

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
```

Then:

```bash
REV=c1899de289a04d12100db370d81485cdf75e47ca
.venv/bin/python scripts/inspect_model.py --model Qwen/Qwen3-0.6B --revision $REV \
  --device cuda --dtype bfloat16 --out experiments/gate2/inventory_cuda.json

.venv/bin/python scripts/baseline.py --variant T0 --model Qwen/Qwen3-0.6B --revision $REV \
  --device cuda --dtype bfloat16 --out experiments/gate2/t0.json

.venv/bin/python scripts/baseline.py --variant Q0 --model Qwen/Qwen3-0.6B --revision $REV \
  --device cuda --dtype bfloat16 --out experiments/gate2/q0.json
```

Sanity checks before copying back:

- inventory totals match `experiments/gate1/inventory_qwen3_0p6b.json`;
- T0 forward KL is exactly 0; Q0 KL is positive;
- Q0 NLL > T0 NLL, both finite; W4 Q0 is much closer to T0 than W3 Q0;
- T0 `teacher_self_kl.two_pass_mean` is ~0 (computed, not assigned);
- `memory.peak_bytes` is the CUDA true peak, and `memory.parameter_bytes` is
  reported separately (do not compare across processes with different models);
- GPU type/driver and resolved revisions are present in every JSON.

```bash
mkdir -p /workspace/artifacts/gate2
cp experiments/gate2/*.json /workspace/artifacts/gate2/
sha256sum /workspace/artifacts/gate2/*.json
```

Copy the three JSON files back to the Mac, verify the hashes, then **terminate
the pod**.

## Guardrails

- No optimizer step in this session; no benchmark generation yet.
- Stop on OOM, non-finite loss, or inventory mismatch; do not retry blindly.
- Never use `--limit-batches` for a reported number.
- Log GPU type, VRAM, driver and `gpu_seconds` with every run.
