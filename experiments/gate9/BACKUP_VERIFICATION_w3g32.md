# W3-G32 best checkpoint backup verification

Source: private RunPod session (address withheld), `/workspace/qad/checkpoints`.
Destination: `checkpoints/gate9_w3g32/` (gitignored).

| File | Bytes | SHA-256 |
|---|---:|---|
| `m_q1_w3g32_s42_best.pt` | 5,596,428,443 | `81505d2adf25616bac7f1a5c4eb360177fa2ee2acf05d812b027a4cc32c4fb23` |
| `m_q1_w3g32_s42_best.safetensors` | 2,072,939,240 | `9beafb6d180c86cf747bbf475682f81be2782779053bdd951d81c2a31403ae29` |

Local `SHA256SUMS.local.txt` equals the remote hashes. Clean-process
restoration (`scripts/verify_backup_local.py`): 310 tensors compared, max
absolute difference **0.0**, no missing tensors; only the tied `lm_head.weight`
is intentionally absent from safetensors.

`experiments/gate9/verify_backup_w3g32.json` records the check.
