# Gate 1.5 diagnosis — why is W3A16 RTN so degraded?

Status: **diagnostic, local (MPS, bf16)**. Not the official Gate 2 evaluation.

- Current record (v2, with cosine similarity): `v2/gate15_diagnostics.json`,
  `v2/diagnostics.md`.
- Original record (v1, no cosine): `archive_v1/` — preserved unchanged.
- Command: `HF_HOME=.hf-cache .venv/bin/python scripts/gate15_diagnostics.py
  --out-dir experiments/gate15/v2 --limit-batches 16`
- The run used the working tree committed as `7e3e35b` (the v2 JSON records
  `git_commit_at_run_start` and `implementation_commit`).

## Setup

- Checkpoint `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`
  (resolved model and tokenizer SHA identical; recorded in the JSON).
- Tokenizer the same repo revision; dtype bf16; device MPS; teacher-forced
  evaluation.
- Fixed FineWeb-Edu subset: 64 blocks × 1,024 tokens, 65,472 scored positions.
  Identical inputs, masks and chunk size for every configuration.
- Quantizer: the project's symmetric per-group quantizer (max-abs scale,
  16-bit scale, no zero-point), applied to the 196 projection weights only.
- W3 uses codes −3…3 (**seven levels**; codes −4 and +4 do not exist), with
  one signed code unused by construction. Effective storage depends on group
  size only through scale metadata.

## Results

| config | NLL | ppl | KL(T‖S) | quantized params | eff. bits | weight MSE | L2 rel-err L0 | L13 | L27 | cos L27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 3.1101 | 22.42 | 0.0000 | — | — | — | 0.00 | 0.00 | 0.00 | 1.000 |
| W4A16-G128 | 3.4242 | 30.70 | 0.4050 | 73.9% | 4.125 | 1.21e-05 | 0.168 | 0.099 | 0.410 | 0.932 |
| W3A16-G32 | 4.5690 | 96.45 | 1.7002 | 73.9% | 3.500 | 4.33e-05 | 0.313 | 0.221 | 0.659 | 0.830 |
| W3A16-G64 | 5.4083 | 223.24 | 2.6142 | 73.9% | 3.250 | 5.40e-05 | 0.353 | 0.252 | 0.788 | 0.772 |
| W3A16-G128 | 6.6259 | 754.38 | 3.9186 | 73.9% | 3.125 | 6.52e-05 | 0.387 | 0.321 | 0.957 | 0.724 |
| W3A16-G128, attention only | 4.2640 | 71.10 | 1.3477 | 29.6% | 3.125 | 6.16e-05 | 0.311 | 0.168 | 0.533 | 0.854 |
| W3A16-G128, MLP only | 4.5649 | 96.05 | 1.6630 | 44.3% | 3.125 | 6.76e-05 | 0.230 | 0.203 | 0.659 | 0.844 |

Controls:

- **Quantization disabled on the fully wrapped model** (196 modules, 4 sequences):
  max abs logit difference 0.0, mean 0.0, NLL difference 0.0 vs the original
  BF16 model. The wrapper/quantizer path is not the cause.
- **Teacher self-KL computed from two independent forward passes** (4 batches,
  4,092 tokens): 0.000e+00 mean and max. A perturbed-teacher unit test confirms
  the function returns positive values when distributions differ.
- **No clipping**: clipped fraction 0.0 for every configuration. Endpoint
  occupancy 2.3–6.5%, zero-level fraction 32–39% at W3. The damage is coarse
  rounding, not outlier clipping.
- **Memory** (process level, MPS sampled peak; lower bound): BF16 2.72 GiB;
  wrapped W3 process 5.57 GiB (teacher + student + teacher cache). Parameter
  bytes tell the real story: teacher 1,137 MiB, wrapped student 1,977 MiB
  (fp32 masters). The process numbers are not comparable and are reported
  separately, with parameter bytes.

## Observed evidence

1. The degradation is real and monotone in quantizer coarseness: ppl 22.4
   (BF16) → 30.7 (W4) → 96.5 (W3-G32) → 223 (G64) → 754 (G128) on identical
   tokens.
2. Group size is the dominant lever at W3: G128 → G32 changes NLL by −2.06
   (ppl 754 → 96) while effective storage rises only 3.125 → 3.5 bits/weight.
3. Both weight families are strongly affected. In **absolute** NLL terms,
   attention-only gives 4.264 and MLP-only 4.565, versus 3.110 for BF16. The
   two ablations cover different fractions of the model (29.6% vs 44.3% of
   parameters), so these numbers do **not** establish a per-parameter
   sensitivity ordering; a matched-coverage or per-module swap would be needed
   for that.
4. L2 relative activation error grows with depth (0.387 → 0.957 from layer 0 to
   27 for W3-G128), but this is a **magnitude-relative** metric, not an angular
   one. Cosine similarity at layer 27 is 0.724 for W3-G128 and 0.932 for W4:
   the activations remain largely direction-aligned even where the relative L2
   error is large. No direction-loss claim is made from the L2 numbers.
5. Weight MSE grows as bits/group shrink (1.2e-5 → 6.5e-5), but NLL damage
   grows much faster: quantization noise is amplified by the network.
6. The worst weight reconstruction MSE is in early-layer `mlp.gate_proj`
   modules (recorded per module in the JSON); the worst end-to-end
   configuration is MLP-only, but see the coverage caveat above.

## Hypotheses (not established)

- W3A16 with max-abs per-group scales may be intrinsically too coarse for a
  0.6B model: with seven levels the half-step error is still a large fraction
  of the weight standard deviation, and the small model has less redundancy
  than 1.7B.
- The MLP may be more sensitive because its outputs enter every residual
  position; this requires a matched-coverage ablation before being claimed.
- QAD from W3 RTN will have a small recoverable margin and poor rollouts;
  W4 RTN leaves a measurable gap (ppl 30.7 vs 22.4, KL 0.405) that QAD has a
  realistic chance to close.
- A stronger initialization (block-wise QAT, GPTQ/AWQ) would close much of the
  W3 gap, but it is outside the brief and was explicitly excluded by the user.

## Recommendation

1. **Treat W3A16-G128 as the aggressive RTN baseline**, not as a locked QAD
   initialization. It is the brief-faithful hard case and should be preserved.
2. **Preserve W3-G32, W4A16-G128 and a stronger W3 initialization as separate
   options.** W4-G128 is the leading candidate for verifying the QAD pipeline
   because it leaves a measurable recovery gap; W3-G32 is the middle point if
   W3 codes are required; a stronger W3 init is a later, separate decision.
3. **Do not silently substitute** any arm. Each is pre-registered with its own
   config and reported on the same evaluation.
4. **Do not implement GPTQ/AWQ or block-wise QAT now.** Revisit only if QAD on
   W4 also fails to recover; then the question becomes initialization, not
   on-policy data.
5. **Gate 2** runs the official CUDA baseline for BF16, W3-G128, W3-G32 and
   W4-G128, a fixed-prompt generation diagnostic (repetition, premature EOS,
   examples), and a short gradient/memory probe. No full QAD training until the
   probe is reviewed.

## Caveats

- MPS bf16, 64-block subset, one run; GPU numbers may differ slightly. The
  ordering and approximate magnitude are what matter.
- The diagnostic uses full-precision FineWeb text, not benchmark tasks;
  benchmark degradation may differ.
- No training was run, so this says nothing about post-QAD quality.
