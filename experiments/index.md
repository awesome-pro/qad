# Experiment registry

Small, Git-tracked index. Metrics and logs live in the referenced artifacts;
large files and checkpoints are never committed. Local Mac runs are marked
`local` and are not reportable.

| Date (UTC) | Gate | Run | Command | Artifacts | Notes |
|---|---|---|---|---|---|
| 2026-11-15 | 0 | quantizer + losses + wrapping + diagnostics + generation + audit tests | `pytest -q` | — | 142 collected, 141 pass, 1 CUDA-only skip |
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
| 2026-11-15 | 4 | Q2 on-policy, 200 steps | `scripts/train_online.py --steps 200 --prompts-per-step 32 --max-new-tokens 128` | `experiments/gate4/q2_w4a16_g128_200.*` | 818,374 tokens (99.95% of Q1-200); ppl 30.65→26.25; GSM8K 14%; 2,382 s; 14.0 GiB |
| 2026-11-15 | 5 | full held-out + GSM8K + generation for Q1/Q2 | `scripts/baseline.py --checkpoint`, `scripts/eval_gsm8k.py`, `scripts/generation_diagnostics.py --checkpoint` | `experiments/gate5/*` | Q1-200 26.28 ppl / 28% GSM; Q1-128 26.96 / 13%; Q2-25 27.17 / 16%; Q2-200 26.25 / 14% |
| 2026-11-15 | 6 | seed-43 replication, 25-step matched arms | `scripts/train_offline.py --seed 43`, `train_online.py --seed 43` | `experiments/gate3/q1_len128_25_seed43.*`, `experiments/gate4/q2_matched25_seed43.*`, `experiments/gate5/*seed43*` | Q1 26.95 ppl / 16% GSM; Q2 27.08 / 18%; ordering replicates |
| 2026-11-15 | 7 | exploratory 1.7B W4A16-G128 | `scripts/inspect_model.py`, `baseline.py`, `train_offline.py`, `train_online.py`, `eval_gsm8k.py` on A6000 48GB | `experiments/gate7/*` | T0 16.99 ppl / 36% GSM; Q0 19.96 / 16%; Q1-200 17.84 / 27%; Q2-25 18.08 / 19% |
| 2026-11-15 | 6 | seed-43 replication, 200-step arms | `scripts/train_offline.py --seed 43`, `train_online.py --seed 43` | `experiments/gate3/q1_200_seed43.*`, `experiments/gate4/q2_200_seed43.*`, `experiments/gate5/*seed43*` | Q1 26.14 ppl / 28% GSM; Q2 26.17 / 22%; tied NLL, Q1 better KL/GSM |
| 2026-11-15 | 6 | rollout-length ablation, 256-token | `scripts/train_offline.py --response-cap 256`, `train_online.py --max-new-tokens 256` | `experiments/gate3/q1_len256_25.*`, `experiments/gate4/q2_len256_25.*`, `experiments/gate5/*len256*` | Q1 26.81 ppl / 15% GSM; Q2 27.11 / 18%; ordering unchanged |
| 2026-11-15 | 6 | W3-G128 QAD probe, 25 and 200 steps | `scripts/train_offline.py --bits 3 --group-size 128` | `experiments/gate3/w3g128_*`, `experiments/gate5/*w3g128*` | 723.98 → 83.56 (25) → 59.35 (200) ppl; GSM 3% → 0% (25% parse); not generative |

## Audit (gate8) — Checkpoint A executed; further runs paused for review

Old GSM8K numbers in gate5/gate7 use the legacy parser and are historical. The
corrected protocol is `GPU_RUNBOOK_AUDIT.md`; new runs write to `experiments/gate8`.

| Planned | Purpose | Artifacts |
|---|---|---|
| strict non-thinking GSM8K, 100 questions | T0/Q0/Q1/Q2 accuracy with explicit-answer parsing | **done + parser v3 frozen**: T0 39%, Q0 17%, Q1 33%, Q2 35%; paired p=0.77 (`gate8/regraded_v3/*`) |
| matched Q1/Q2, 200 steps, seeds 42/43 | same prompt schedule, prompt IDs/step, 128-token cap | **seed 42 done**; `check_matched` clean; seed 43 paused (`gate8/m_q*_s42.*`) |
| best/final checkpoints | validation-selected vs final; portable `.safetensors` | `checkpoints/m_*_best.*` |
| final test on WikiText-103 | disjoint test metrics | **done seed 42**: Q1 30.64 PPL, Q2 29.92 (`gate8/test_m_q*`) |
| thinking subset, 10 questions, 1536 tokens | completion/truncation/thinking-closed rates | **done (pilot)**: low completion; not an accuracy benchmark (`gate8/gsm8k_*_think10.*`) |
| 1.7B T0/Q0/Q1 | exploratory corrected arm | **T0/Q0 baselines done; Q1 stopped for Checkpoint A** (`gate8/test_1p7b_*`) |
| 2026-11-15 | 8 | seed-43 matched Q1/Q2 | `scripts/train_matched.py --seed 43` | `experiments/gate8/m_q*_s43.*`, `regraded_v3/*s43*` | prompt hashes match; Q1 30.68 PPL / 35% GSM; Q2 30.72 / 43%; paired +8 pts, p=0.17 |
| 2026-11-15 | 8 | W3 matched Q1 | `scripts/train_matched.py --bits 3 --arm q1` | `experiments/gate8/m_q1_w3_s42.*` | val 62.42 PPL, test 88.24, GSM 0%; not generative |
