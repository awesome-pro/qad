# Final correctness and methodology audit

## Verified invariants

| Check | Result |
|---|---|
| Matched Q1/Q2 seed 42 | `matched: true`; token ratio 1.00118; 0 prompt-hash mismatches |
| Matched Q1/Q2 seed 43 | `matched: true`; token ratio 1.00138; 0 prompt-hash mismatches |
| Parser | frozen v3.0.0 (`docs/parser_spec_v3.md`) |
| Canonical result integrity | seed-42 (6 artifacts + paired), seed-43 (5), 1.7B (3), W3-G32 (1): all **passed** |
| Per-example vs summaries | recomputed from JSONL; counts, statuses, correctness agree |
| Provenance | metrics/parser/source hashes/checkpoint SHA agree |
| Revisions | 0.6B `c1899de2…`; 1.7B `70d244cc…` |
| Quantization | W4-G128 / W3-G128 / W3-G32 as labelled |
| Seeds / best steps | run JSON records seed and validation-selected best step |
| Validation/test splits | disjoint; block-hash intersection 0 |
| Best-checkpoint selection | validation NLL only; test never used |
| Failed run handling | first W3 attempt (disk-full) labelled failed; only retry reported |
| Storage/memory separation | theoretical packed sizes, safetensors sizes and measured GPU peaks reported separately |
| Unit tests | **175 collected, 173 passed, 2 environment skips** |

## GSM8K parser caveat

Some 1.7B Q0 outputs use:

```text
#### Final Answer:
#### 18
```

Parser v3 treats the first line as a malformed marker and returns `ambiguous`
because it conflicts with the valid second marker. In 1.7B Q0 this affects
**9/100** examples. If the second marker were accepted, at most 5 of those 9
would be correct, moving Q0 from 44% to 49% — still below BF16 58% and above
Q1 42%. The conclusion is unchanged.

The same pattern is rare in the other runs: seed-42 Q2 1/100, seed-43 Q2 1/100.
Parser v3 was **not** changed; a future correction should be separately
versioned and all historical metrics must remain labelled as v3.

## Measurement distinctions preserved

- **Strict GSM8K accuracy** = parsed final answer equals gold on the fixed
  100-example subset.
- **Parsing/completion metrics** = marker presence, parsed-answer rate,
  completion, truncation.
- **Teacher-forced NLL/PPL/KL** = held-out language-modelling metrics.
- **Open-ended generation diagnostics** = fixed-prompt repetition/quality,
  not accuracy.

These are reported separately and are not interchangeable.

## Remaining limitations

- Single seed for W3 and 1.7B; two seeds only for 0.6B W4.
- 100 GSM8K examples; confidence intervals are wide.
- Thinking subsets are 10-question diagnostics only.
- W3 failure cannot be causally attributed to quantization vs formatting.
- Fake quantization does not imply deployed speed or memory savings.
- Seed-43/W3-G128 weights are lost; only results remain.
- The 1.7B checkpoint backup is incomplete as of this audit; see checkpoint
  inventory before relying on it.
