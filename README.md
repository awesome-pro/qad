# Low-bit QAD for Qwen3

Research question: **can quantization-aware distillation (QAD) recover a
weight-only low-bit Qwen3 model, and does on-policy distillation on the
student's own rollouts beat matched offline distillation?**

This is a completed research pilot, not a deployment artifact. It uses fake
quantization to measure accuracy; it does not produce packed INT3/INT4 weights
or claim serving speedups.

**Read the final report: [`reports/FINAL_RESEARCH_REPORT.md`](reports/FINAL_RESEARCH_REPORT.md).**

## Main result

Qwen3-0.6B, W4A16-G128, disjoint WikiText test, strict GSM8K (100 questions):

| Variant | Test PPL | GSM8K |
|---|---:|---:|
| BF16 | 25.52 | 39% |
| W4 RTN | 36.79 | 17% |
| Offline QAD, seed 42 | 30.64 | 33% |
| Offline QAD, seed 43 | 30.68 | 35% |
| Online QAD, seed 42 | 29.92 | 35% |
| Online QAD, seed 43 | 30.72 | 43% |

Offline QAD recovers most of the W4 likelihood damage. Online QAD has no
statistically established advantage and costs extra rollout compute.

Aggressive W3 is not generatively usable even after QAD. W3-G128: 88.2 test
PPL, 0% GSM8K. W3-G32 improves to 51.4 test PPL but remains 0%; its failures are
dominated by premature control-token termination.

## What was tested

- Qwen3-0.6B and Qwen3-1.7B, BF16 teacher.
- W3/W4 symmetric per-group fake quantization; FP32 master weights + STE.
- Offline forward-KL QAD on gold completions.
- On-policy forward-KL QAD with fresh student rollouts.
- Two matched 0.6B seeds, matched prompt schedules, validation-selected
  checkpoints, disjoint validation/test splits.
- Strict GSM8K with frozen parser v3.0.0; parser and data manifests included.

## What remains unproven

- Whether on-policy QAD helps with longer rollouts or different objectives.
- Whether stronger W3 initialization (GPTQ/AWQ or block-wise QAT) restores
  generation. This is the strongest next question; it was not implemented.
- Whether results generalize beyond W4-G128, Qwen3, or the fixed 100-example
  GSM8K subset.
- Any packed inference speed or memory claim.

## Reproduce

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

# data (requires the pinned HF datasets; see scripts/prepare_data.py)
python scripts/prepare_data.py
python scripts/prepare_splits.py

# baselines and matched QAD
python scripts/baseline.py --variant T0 --device cuda --eval data/cache/eval/wikitext_test_ids.npy --out /tmp/t0.json
python scripts/baseline.py --variant Q0 --bits 4 --group-size 128 --device cuda \
  --eval data/cache/eval/wikitext_test_ids.npy --out /tmp/q0.json
python scripts/train_matched.py --arm q1 --bits 4 --group-size 128 --steps 200 \
  --prompts-per-step 32 --continuation-cap 128 --val-set data/cache/eval/fineweb_val_ids.npy \
  --run-name q1 --out-dir experiments/gate8 --checkpoint-dir checkpoints
python scripts/eval_gsm8k_strict.py --mode nonthinking --config-name q1 \
  --bits 4 --checkpoint checkpoints/q1_best.pt --count 100 --out-prefix /tmp/gsm8k_q1
```

Exact seeds, hashes and run configurations are in the run JSONs under
`experiments/gate8/` and `experiments/gate9/`. Checkpoint availability is in
[`reports/checkpoint_inventory.md`](reports/checkpoint_inventory.md).

## Documentation

- Final report: [`reports/FINAL_RESEARCH_REPORT.md`](reports/FINAL_RESEARCH_REPORT.md)
- Results audit: [`reports/final_audit.md`](reports/final_audit.md)
- W3 failure analysis: [`reports/w3_failure_analysis.md`](reports/w3_failure_analysis.md)
- Parser spec: [`docs/parser_spec_v3.md`](docs/parser_spec_v3.md)
- 1.7B handoff: [`GPU_RUNBOOK_1P7B.md`](GPU_RUNBOOK_1P7B.md)
- Historical pilot: [`reports/findings.md`](reports/findings.md) (legacy parser; labelled)
- Experiment registry: [`experiments/index.md`](experiments/index.md)

## Tests

```bash
.venv/bin/python -m pytest -q
```

175 collected, 173 passed, 2 environment skips (CUDA-only quantizer test;
pinned-tokenizer test when not cached).
