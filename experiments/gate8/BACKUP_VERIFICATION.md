# Checkpoint A backup verification

Date: 2026-11-15 (UTC). Source pod: a private RunPod session (address withheld);
`/workspace/qad/checkpoints`. Destination: local
`checkpoints/gate8/` (gitignored, outside pod ephemeral storage).

## Files copied and verified

| File | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `m_q1_w4_s42_best.pt` | 5,596,425,667 | `3fc424d2e64c38102d2d714842a5db9e2e5589ee8f5e233bc66997a84eaf1930` | hash match |
| `m_q1_w4_s42_best.safetensors` | 2,072,939,240 | `a907fb93bc415fbbf98aff18e460c1f98e7f6e212815e9208f6534aba24d5f65` | hash match |
| `m_q2_w4_s42_best.pt` | 5,596,425,667 | `23c15270278e482d336691653931dc3df27f82850a0de079662404eca7fd49a6` | hash match |
| `m_q2_w4_s42_best.safetensors` | 2,072,939,240 | `c7b6a27a6d3df488184f0a6598adb81f22e54a1363e6dd8fad3119a74ba84fc3` | hash match |

Remote manifest: `SHA256SUMS.remote.txt`; local manifest:
`SHA256SUMS.local.txt`. All four local hashes equal the remote hashes.

## Clean-process restoration

Two fresh Python processes (`scripts/verify_backup_local.py`) loaded each local
`.pt` with `torch.load(map_location="cpu")`, loaded the matching safetensors,
and compared all 310 tensors:

- `verify_backup_q1.json`: `passed: true`, max abs diff **0.0**.
- `verify_backup_q2.json`: `passed: true`, max abs diff **0.0**.
- Only omitted key is the tied `lm_head.weight` (intentionally excluded from
  safetensors to avoid shared storage).

On the pod, `scripts/verify_checkpoints.py` additionally compared logits and
held-out metrics for the same SHA-256 files: max logit diff 0.0, argmax match
1.0, NLL/KL difference 0.0 for both arms (`verify_q1_s42.json`,
`verify_q2_s42.json`). Hash-identical backups therefore restore the same model
that was evaluated.

## Not copied

`m_q1_w4_s42_final.safetensors`, `m_q2_w4_s42_final.safetensors`, and the
`*_final.pt` files do not exist (the runs intentionally skipped the final
optimizer checkpoint). The final safetensors remain on the pod and are listed in
the remote manifest; the selected (best) weights and optimizer states are backed
up. No pod files were deleted or overwritten.
