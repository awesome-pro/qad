# Seed-43 / W3 checkpoint backup status — BLOCKED

Status: **not verified / incomplete.** Do not represent these checkpoints as
durably backed up.

## What exists

The A5000 held:

| File | Remote SHA-256 (captured) |
|---|---|
| `m_q1_w4_s43_best.pt` | `1652e5e11b134a7c18c68e6a3150ce38708ddd788b1947df3dc9c97df1cd56b0` |
| `m_q1_w4_s43_best.safetensors` | `b23cca33e6281296e3d039cd0a3840550e56b6ae5bef67c7145576c24468bfbe` |
| `m_q2_w4_s43_best.pt` | `4ce45e978793c4bba775601c44fba63a692099397a13ceafbac209aeb28aed59` |
| `m_q2_w4_s43_best.safetensors` | `2c40eba0bd2efbc6ae13ff5d50ed37178f23435cc5aceb96340101da9d20dc24` |
| `m_q1_w3_s42_best.pt` | `b9c3827a9533c2d25b1731dc9140bb019c9c07fdb7d93a8108cfb0a6b20f242b` |
| `m_q1_w3_s42_best.safetensors` | `e7229bf226d0a2a0d39f683f30cedc24a7af2413a2d726c23ad95a71a5fb56f7` |

Remote hashes were captured while the pod was alive. Copy attempts were
interrupted by network resets; only partial local files remain
(~231 MB / 248 MB / 259 MB), which are **not valid checkpoints**. A final retry
then failed because the A5000 SSH endpoint began refusing connections.

## What is safe

- All result JSON/JSONL, strict regrades, provenance, integrity reports and
  experiment logs are copied to the Mac and committed.
- The four **seed-42** best checkpoints are fully verified (see
  `experiments/gate8/BACKUP_VERIFICATION.md`).
- No seed-42 or seed-43 pod checkpoint was deleted by the backup script. The
  only earlier deletions were verified seed-42 pod copies and the incomplete
  first W3 attempt, documented in the W3 report.

## Required action

If the A5000 was **stopped** rather than terminated, start the same pod and run
`checkpoints/backup_retry.sh`; it resumes with remote SHA-256 checks. If the pod
was terminated, the seed-43/W3 optimizer and safetensors weights are lost;
their results remain valid and the models can be regenerated, but the backup
requirement cannot be completed.

No claim of clean-process restoration for seed-43/W3 is made.
