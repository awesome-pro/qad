#!/usr/bin/env bash
# Bundle code + prepared data for transfer to a GPU pod (no venv, no model cache).
set -euo pipefail
out="${1:-qad_gpu_bundle.tgz}"
tar \
  --exclude=.venv \
  --exclude=.uv-cache \
  --exclude=.uv-python \
  --exclude=.hf-cache \
  --exclude=.git \
  --exclude=.ssh \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='experiments/runs' \
  --exclude='data/cache/openthoughts' \
  -czf "$out" .
ls -lh "$out"
