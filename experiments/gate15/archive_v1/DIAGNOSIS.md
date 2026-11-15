# Gate 1.5 diagnosis - why is W3A16 RTN so degraded?

Status: **diagnostic, local (MPS, bf16)**. Not the official Gate 2 evaluation.
Machine-readable results: `gate15_diagnostics.json`. The run used the working tree committed as `a28ff1b90124` (the JSON records the HEAD at process start, `3afdd19`, plus `implementation_commit`). Command:

```bash
HF_HOME=.hf-cache .venv/bin/python scripts/gate15_diagnostics.py
```

## Setup

- Checkpoint `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`
  (resolved model and tokenizer SHA identical; recorded in the JSON).
- Tokenizer the same repo revision; dtype bf16; device MPS; seedless
  teacher-forced evaluation.
- Fixed FineWeb-Edu subset: 64 blocks × 1,024 tokens, 65,472 scored positions.
  Identical inputs, masks and chunk size for every configuration.
- Quantizer: the project's symmetric per-group quantizer (max-abs scale,
  16-bit scale, no zero-point), applied to the 196 projection weights only.

## Results

| config | NLL | ppl | KL(T‖S) | quantized params | eff. bits | weight MSE | act rel-err L0 | L13 | L27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 3.1101 | 22.42 | 0.0000 | - | - | - | 0.00 | 0.00 | 0.00 |
| W4A16-G128 | 3.4242 | 30.70 | 0.4050 | 73.9% | 4.125 | 1.21e-05 | 0.168 | 0.099 | 0.410 |
| W3A16-G32 | 4.5690 | 96.45 | 1.7002 | 73.9% | 3.500 | 4.33e-05 | 0.313 | 0.221 | 0.659 |
| W3A16-G64 | 5.4083 | 223.24 | 2.6142 | 73.9% | 3.250 | 5.40e-05 | 0.353 | 0.252 | 0.788 |
| W3A16-G128 | 6.6259 | 754.38 | 3.9186 | 73.9% | 3.125 | 6.52e-05 | 0.387 | 0.321 | 0.957 |
| W3A16-G128, attention only | 4.2640 | 71.10 | 1.3477 | 29.6% | 3.125 | 6.16e-05 | 0.311 | 0.168 | 0.533 |
| W3A16-G128, MLP only | 4.5649 | 96.05 | 1.6630 | 44.3% | 3.125 | 6.76e-05 | 0.230 | 0.203 | 0.659 |

Controls:

- **Quantization disabled on the fully wrapped model** (196 modules, 4 sequences):
  max abs logit difference 0.0, mean 0.0, NLL difference 0.0 vs the original
  BF16 model. The wrapper/quantizer path is not the cause.
- **Teacher self-KL computed from two independent forward passes** (4 batches,
  4,092 tokens): 0.000e+00 mean and max. It is computed, not assigned; a
  perturbed-teacher unit test confirms the function returns positive values.
- **No clipping**: clipped fraction 0.0 for every configuration. Endpoint
  occupancy 2.3–6.5%, zero-level fraction 32–39% at W3. The damage is coarse
  rounding, not outlier clipping.
- **Memory** (process level, MPS sampled peak; lower bound): BF16 2.72 GiB;
  wrapped W3 process 5.57 GiB (teacher + student + teacher cache). Parameter
  bytes tell the real story: teacher 1,137 MiB, wrapped student 1,977 MiB
  (fp32 masters). The two process numbers are not comparable and are reported
  separately, with parameter bytes.

## Observed evidence

1. The degradation is real and monotone in quantizer coarseness: ppl 22.4
   (BF16) → 30.7 (W4) → 96.5 (W3-G32) → 223 (G64) → 754 (G128) on identical
   tokens.
2. Group size is the dominant lever at W3: G128 → G32 changes NLL by −2.06
   (ppl 754 → 96) while effective storage rises only 3.125 → 3.5 bits/weight.
3. Both weight families are strongly affected. Attention-only gives ppl 71 and
   MLP-only ppl 96, versus 22 for BF16. MLP is slightly worse despite being
   44% of parameters against 30% for attention, so MLP quantization is more
   damaging per parameter here.
4. Error compounds with depth: final-layer relative activation error is 0.96
   for full W3-G128, 0.66 for MLP-only, 0.53 for attention-only, and 0.41 even
   for W4. A relative error near 1 means the direction is almost lost.
5. Weight MSE grows as bits/group shrink (1.2e-5 → 6.5e-5), but the NLL damage
   grows much faster: quantization noise is amplified by the network, not a
   simple weight-space effect.
6. The worst weight reconstruction MSE is in early-layer `mlp.gate_proj`
   modules; the worst end-to-end configuration, however, is MLP-only. Both
   statements are in the JSON, per module.

## Hypotheses (not established)

- W3A16 with max-abs per-group scales is intrinsically too coarse for a 0.6B
  model: with 3 levels, the half-step error is a large fraction of the weight
  standard deviation, and the small model has less redundancy than 1.7B.
- The MLP is more sensitive because `down_proj`/`gate_proj` feed every residual
  stream position; attention errors are partly absorbed by the residual and
  normalization.
- QAD from W3 RTN will have a small recoverable margin and poor rollouts;
  W4 RTN leaves a measurable gap (ppl 30.7 vs 22.4, KL 0.405) that QAD has a
  realistic chance to close.
- A stronger initialization (block-wise QAT, GPTQ/AWQ) would close much of the
  W3 gap, but it is outside the brief and was explicitly excluded by the user.

## Recommendation

1. **Do not replace the primary quantizer.** Keep W3A16-G128 RTN as the
   brief-faithful primary arm; the diagnostics show the failure is intrinsic,
   not an implementation bug.
2. **Add W4A16-G128 RTN as a separate, pre-registered variant** in the same
   experiment matrix (same teacher, data, loss, budget, evaluation). It is the
   cheapest setting with a plausible recovery signal, and it isolates
   "can QAD recover this quantization error?" from "is W3 RTN recoverable at
   all?".
3. **Optionally add W3A16-G32** as a middle point if W3 codes are required for
   deployment fidelity; label its 3.5 effective bits honestly.
4. **Do not implement GPTQ/AWQ or block-wise QAT now.** Revisit only if QAD on
   W4 also fails to recover; then the question becomes initialization, not
   on-policy data.
5. In Gate 2, also evaluate **Q0 at W4** and inspect a few W3 Q0 generations
   qualitatively before committing Stage-2 GPU hours; ppl 754 suggests the W3
   rollouts may be incoherent.

## Caveats

- MPS bf16, 64-block subset, one run; GPU numbers may differ slightly. The
  ordering and approximate magnitude are what matter.
- The diagnostic uses the full-precision FineWeb text, not benchmark tasks;
  benchmark degradation may differ.
- No training was run, so this says nothing about post-QAD quality.
