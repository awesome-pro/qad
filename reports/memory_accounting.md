# Theoretical storage accounting

These are **theoretical packed-model sizes** computed from parameter shapes and the declared quantizer metadata. They are not measured packed artifacts and do not imply inference speedups. Measured training memory is reported separately in `reports/archive/findings.md`.

| model | config | total params | quantized params | payload MiB | metadata MiB | total GiB | eff bits/weight | vs BF16 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-0.6B | BF16 | 596,049,920 | 440,401,920 | 840.0 | 0.0 | 1.11 | 16.000 | 1.00x |
| Qwen/Qwen3-0.6B | W4A16-G128 | 596,049,920 | 440,401,920 | 210.0 | 6.6 | 0.50 | 7.226 | 2.21x |
| Qwen/Qwen3-0.6B | W3A16-G32 | 596,049,920 | 440,401,920 | 157.5 | 26.2 | 0.47 | 6.764 | 2.37x |
| Qwen/Qwen3-0.6B | W3A16-G128 | 596,049,920 | 440,401,920 | 157.5 | 6.6 | 0.45 | 6.487 | 2.47x |
| Qwen/Qwen3-1.7B | BF16 | 1,720,574,976 | 1,409,286,144 | 2688.0 | 0.0 | 3.20 | 16.000 | 1.00x |
| Qwen/Qwen3-1.7B | W4A16-G128 | 1,720,574,976 | 1,409,286,144 | 672.0 | 21.0 | 1.26 | 6.273 | 2.55x |
| Qwen/Qwen3-1.7B | W3A16-G32 | 1,720,574,976 | 1,409,286,144 | 504.0 | 84.0 | 1.15 | 5.762 | 2.78x |
| Qwen/Qwen3-1.7B | W3A16-G128 | 1,720,574,976 | 1,409,286,144 | 504.0 | 21.0 | 1.09 | 5.454 | 2.93x |

Metadata assumes one 16-bit scale per group and no zero-point (symmetric scheme); alignment/packing overhead is unknown and excluded.
Unquantized parameters (embedding, tied LM head, norms) are counted at 16 bits.
