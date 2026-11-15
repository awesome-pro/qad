# Gate 1.5 diagnostics

- model `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`
- device `mps`, dtype `bfloat16`, eval blocks 64 × 1024
- git `3afdd195faa4`, created 2026-11-15T01:58:21.202448+00:00

| config | NLL | ppl | KL(T||S) | quantized params | eff bits | weight MSE | act L0 | act L13 | act L27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 3.1101 | 22.42 | 0.0000 | 0.0% | — | — | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| w4a16_g128 | 3.4242 | 30.70 | 0.4050 | 73.9% | 4.125 | 1.21e-05 | 1.68e-01 | 9.91e-02 | 4.10e-01 |
| w3a16_g32 | 4.5690 | 96.45 | 1.7002 | 73.9% | 3.500 | 4.33e-05 | 3.13e-01 | 2.21e-01 | 6.59e-01 |
| w3a16_g64 | 5.4083 | 223.24 | 2.6142 | 73.9% | 3.250 | 5.40e-05 | 3.53e-01 | 2.52e-01 | 7.88e-01 |
| w3a16_g128 | 6.6259 | 754.38 | 3.9186 | 73.9% | 3.125 | 6.52e-05 | 3.87e-01 | 3.21e-01 | 9.57e-01 |
| w3a16_g128_attn | 4.2640 | 71.10 | 1.3477 | 29.6% | 3.125 | 6.16e-05 | 3.11e-01 | 1.68e-01 | 5.33e-01 |
| w3a16_g128_mlp | 4.5649 | 96.05 | 1.6630 | 44.3% | 3.125 | 6.76e-05 | 2.30e-01 | 2.03e-01 | 6.59e-01 |

## Quantization-disabled equivalence

- wrapped modules: 196
- sequences: 4
- max abs logit diff: 0.000e+00
- mean abs logit diff: 0.000e+00
- NLL teacher: 3.147981
- NLL wrapped/disabled: 3.147981
- NLL abs diff: 0.000e+00

## Teacher self-KL (two independent passes)

- mean: 0.000e+00
- max batch: 0.000e+00

Full machine-readable results: `gate15_diagnostics.json`.
