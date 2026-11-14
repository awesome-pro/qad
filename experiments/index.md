# Experiment registry

Small, Git-tracked index. Metrics and logs live in the referenced artifacts;
large files and checkpoints are never committed. Local Mac runs are marked
`local` and are not reportable.

| Date (UTC) | Gate | Run | Command | Artifacts | Notes |
|---|---|---|---|---|---|
| 2026-11-15 | 0 | quantizer + losses + wrapping tests | `pytest -q` | — | 76 collected, 75 pass, 1 CUDA-only skip |
| 2026-11-15 | 1 | checkpoint inventory | `scripts/inspect_model.py` | `experiments/gate1/inventory_qwen3_0p6b.json` | 596,049,920 params; 440,401,920 quantizable; 197 Linear (196 + lm_head) |
| 2026-11-15 | 1 | fixed data | `scripts/prepare_data.py` | `data/manifests/gate1_data_manifest.json` | FineWeb 256x1024; OpenThoughts 3,000 prompts, 2,979 truncated gold responses |
| 2026-11-15 | 1 | T0/Q0 smoke (local) | `scripts/baseline.py --limit-batches 8` | `experiments/gate1/local_smoke_t0.json`, `local_smoke_q0.json` | MPS bf16, 32,736 tokens; ppl 23.1 vs 727.4; not reportable |
|  | 2 | official T0/Q0 | `scripts/baseline.py` on GPU | `experiments/gate2/*.json` | **pending first GPU session** |
