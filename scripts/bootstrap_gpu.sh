#!/usr/bin/env bash
# One-time environment setup on a fresh GPU pod. Run from the repo root.
set -euo pipefail

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "== disk =="
df -h .

python3 --version
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]" datasets

.venv/bin/python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
print("transformers", transformers.__version__)
PY
