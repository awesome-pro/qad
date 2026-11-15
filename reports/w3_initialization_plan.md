# W3 initialization plan

## What we learned

- W3-G128 QAD: val 62.4 / test 88.2 PPL, 0% strict GSM8K.
- W3-G32 QAD: val 40.8 / test 51.4 PPL, still 0% GSM8K (31% completion,
  1% marker presence).
- W4 QAD: test 30.7 PPL, 33–35% GSM8K.

Granularity is a real lever for likelihood but not sufficient for generation.
The remaining hypothesis is **initialization**, not more QAD steps.

## Candidate order

1. **GPTQ W3 initialization.** Calibrate on the pinned OpenThoughts/FineWeb
   prompts, quantize W3-G128 with second-order weight updates, dequantize into
   the trainable masters, then run the same matched Q1-200 QAD. Compare
   against W3-G32 and W4.
2. **AWQ W3 initialization.** Activation-aware channel scaling, same QAD
   protocol. If GPTQ is unstable or unavailable for Qwen3 with the pinned
   transformers, AWQ is the alternate.
3. **Block-wise QAT initialization.** Larger engineering effort; only after
   GPTQ/AWQ. This is the brief's Stage-1 idea and must be explicitly approved.
4. Optional control: **W3-G16** to extend the granularity curve; cheap but
   unlikely to fix generation by itself.

## Required protocol for any candidate

- Same model revision, data splits, 200 steps, 32 prompts/step, 128-token cap,
  matched schedule, validation-selected best checkpoint, frozen parser v3.
- Report val/test PPL, KL, strict GSM8K (non-thinking 100), and targeted
  generation diagnostics. Thinking subset remains optional.
- Back up best `.pt` + safetensors immediately with SHA-256 manifests and
  clean-process restoration before any pod cleanup.
- No broad benchmark sweep; one candidate at a time.

## Cost estimate

- GPTQ/AWQ integration: 1–2 days engineering + ~1 GPU-hour per W3 run.
- Block-wise QAT: 2–3 days engineering + ~2 GPU-hours.
- W3-G16 control: ~30 minutes.

## Decision

Start with GPTQ/AWQ; use W3-G32 as the reference. Do not run more W3 QAD steps
without a new initialization.
