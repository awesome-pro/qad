# Checkpoint inventory (final)

**Verified** means local SHA-256 equals the source manifest and a clean-process
restoration check passed. A partial file is not a checkpoint.

| Experiment | Best `.pt` | Best `.safetensors` | Status |
|---|---|---|---|
| 0.6B W4-G128 Q1 seed42 | `3fc424d2…eaf1930` | `a907fb93…24d5f65` | **verified local** (`checkpoints/gate8/`) |
| 0.6B W4-G128 Q2 seed42 | `23c15270…7fd49a6` | `c7b6a27a…78c3a3` | **verified local** (`checkpoints/gate8/`) |
| 0.6B W3-G32 Q1 seed42 | `81505d2a…c4fb23` | `9beafb6d…03ae29` | **verified local** (`checkpoints/gate9_w3g32/`) |
| 1.7B W4-G128 Q1 seed42 | `4c808aab…284e4e` | `e9c354db…a8d105` | **safetensors verified local** (`checkpoints/gate8_1p7b/`, 5.8 GB); `.pt` remains pod-only |
| 0.6B W4 seed43 Q1/Q2 | lost | lost | results preserved; weights unavailable |
| 0.6B W3-G128 Q1 | lost | lost | results preserved; weights unavailable |

Seed-42 and W3-G32 clean-process checks compared all 310 tensors with max
absolute difference 0.0; only the tied `lm_head.weight` is omitted from
safetensors.

1.7B safetensors was assembled from four chunks; local SHA-256 matches the
remote hash, and `experiments/gate8/verify_1p7b.json` confirms the pod `.pt`
and safetensors have identical logits and held-out metrics (max diff 0.0).
The `.pt` optimizer checkpoint (~17 GB) is still pod-only.

## Preserved results without weights

- Seed-43 matched Q1/Q2: `experiments/gate8/m_q*_s43.*`, `regraded_v3/*s43*`.
- W3-G128: `experiments/gate5` summaries and report.
- W3-G32: results and failure analysis.

## Not claimed

No deployment checkpoint, packed INT3/INT4 artifact, or vLLM-ready model
exists. The `.safetensors` files are portable fake-quant master weights for
evaluation, not packed inference artifacts.
