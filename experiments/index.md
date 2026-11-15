# Experiment registry

Small, Git-tracked index. Metrics and logs live in the referenced artifacts;
large files and checkpoints are never committed. Local Mac runs are marked
`local` and are not reportable.

| Date (UTC) | Gate | Run | Command | Artifacts | Notes |
|---|---|---|---|---|---|
| 2026-11-15 | 0 | quantizer + losses + wrapping + diagnostics + generation tests | `pytest -q` | — | 98 collected, 97 pass, 1 CUDA-only skip |
| 2026-11-15 | 1 | checkpoint inventory | `scripts/inspect_model.py` | `experiments/gate1/inventory_qwen3_0p6b.json` | 596,049,920 params; 440,401,920 quantizable; 197 Linear (196 + lm_head) |
| 2026-11-15 | 1 | fixed data | `scripts/prepare_data.py` | `data/manifests/gate1_data_manifest.json` | FineWeb 256x1024; OpenThoughts 3,000 prompts, 2,979 truncated gold responses |
| 2026-11-15 | 1 | T0/Q0 smoke (local) | `scripts/baseline.py --limit-batches 8` | `experiments/gate1/local_smoke_t0.json`, `local_smoke_q0.json` | MPS bf16, 32,736 tokens; ppl 23.1 vs 727.4; superseded by Gate 1.5, not reportable |
| 2026-11-15 | 1.5 | W3A16 degradation diagnosis (v2, cosine added) | `scripts/gate15_diagnostics.py --out-dir experiments/gate15/v2` | `experiments/gate15/v2/*`, `experiments/gate15/archive_v1/*`, `DIAGNOSIS.md` | 64 blocks/65,472 tokens, 7 configs. BF16 ppl 22.4; W4 30.7; W3-G32 96.5; G64 223.2; G128 754.4; attn-only 71.1; MLP-only 96.1. Disabled-wrap equivalence 0.0; self-KL 0.0; revised revisions. Recommendation: W3-G128 is the aggressive baseline, not a locked init; preserve W3-G32/W4/stronger-W3 options |
| 2026-11-15 | 1.5 | generation diagnostic smoke (local) | `scripts/generation_diagnostics.py --num-prompts 2 --max-new-tokens 32` | `experiments/gate15/smoke/generation_smoke.json` | MPS; W3-G128 repetition 0.36 vs 0.00 BF16/W4; not reportable |
| 2026-11-15 | 1.5 | training probe smoke (local) | `scripts/probe_training_step.py --bits 4 --steps 1` | `experiments/gate15/smoke/training_probe_smoke.json` | W4, 1 step; loss 0.583, all assertions true; not reportable |
| 2026-11-15 | 2 | official CUDA baseline ×4 | `scripts/baseline.py` on A5000 24GB | `experiments/gate2/{t0_bf16,q0_w3a16_g128,q0_w3a16_g32,q0_w4a16_g128}.json` | 261,888 tokens; PPL 22.26 / 723.98 / 94.44 / 30.59; T0 self-KL 0; hashes verified |
| 2026-11-15 | 2 | generation diagnostic ×4 | `scripts/generation_diagnostics.py` | `experiments/gate2/generation_diagnostics.{json,md}` | 8 fixed prompts; 4-gram repetition 0.047 / 0.965 / 0.839 / 0.117 |
| 2026-11-15 | 2 | gradient/memory probe W4 and W3 | `scripts/probe_training_step.py` | `experiments/gate2/training_probe_*.json` | 3 steps; 9.32 GiB peak; teacher frozen, finite grads, masters changed |
| 2026-11-15 | 3 | Q1 offline QAD, 25-step probe | `scripts/train_offline.py --steps 25` | `experiments/gate3/q1_w4a16_g128_probe.*` | 102,094 tokens; ppl 31.95→28.16; all assertions true |
| 2026-11-15 | 3 | Q1 offline QAD, 200 steps | `scripts/train_offline.py --steps 200` | `experiments/gate3/q1_w4a16_g128_200.*` | 818,760 tokens; ppl 30.65→26.28; GSM8K 28% |
| 2026-11-15 | 3 | Q1 length-matched 25 steps | `scripts/train_offline.py --steps 25 --response-cap 128` | `experiments/gate3/q1_w4a16_g128_len128_25.*` | 102,400 tokens; ppl 30.65→26.96 |
| 2026-11-15 | 4 | Q2 on-policy, 25 steps | `scripts/train_online.py --steps 25 --prompts-per-step 32 --max-new-tokens 128` | `experiments/gate4/q2_w4a16_g128_matched25.*` | 101,499 tokens (99.4% of Q1); ppl 30.65→27.17; GSM8K 16%; rollouts refresh |
| 2026-11-15 | 5 | full held-out + GSM8K + generation for Q1/Q2 | `scripts/baseline.py --checkpoint`, `scripts/eval_gsm8k.py`, `scripts/generation_diagnostics.py --checkpoint` | `experiments/gate5/*` | Q1-200 26.28 ppl / 28% GSM; Q1-128 26.96 / 13%; Q2-25 27.17 / 16% |
