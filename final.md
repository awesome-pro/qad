**We are officially ending the experimental exploration phase and entering final verification, artifact preservation, documentation, and presentation.**

The objective is to produce a technically rigorous, reproducible, polished research artifact that demonstrates meaningful progress toward the trial's goals.

## Strict scope

**Do not implement or run:**

- GPTQ or AWQ initialization.
- Block-wise QAT.
- Additional QAD training.
- New model-size experiments.
- New benchmarks or large inference sweeps.
- Another GSM8K parser revision without a genuinely established correctness issue.
- Unnecessary refactoring or new infrastructure.

GPTQ/AWQ is a proposed next research direction, not a prerequisite for finishing this pilot.

Use existing artifacts for analysis wherever possible. besides both the A5000 & A6000 are still  completely available. use when needed.  Once we are done, let me know. I will terminate both.

---

## P0 — Secure all available research checkpoints

This is the highest operational priority.

Review `reports/checkpoint_backup_status.md` and verify the actual checkpoint state.

### 1. Qwen3-1.7B

The 1.7B best checkpoint reportedly remains on the A6000, but its backup was incomplete.

- Check whether the existing pod and checkpoint files are still accessible.
- Verify the files and their SHA-256 hashes.
- Prioritize durably preserving `m_q1_w4_1p7b_best.safetensors` (approximately 6.26 GB).
- If feasible, also preserve the best `.pt` checkpoint containing optimizer state (approximately 17.5 GB).
- Prefer a resumable transfer or a direct pod-to-object-storage upload over repeated unreliable full-file transfers.
- If an external storage destination or credentials are required, request them rather than inventing a destination.
- Verify the copied file's hash against the source.
- Record its storage location, size, hash, and restoration status.

Do not terminate or delete the pod/checkpoint before verification.

### 2. Other checkpoints

Verify the existing status without redoing work unnecessarily:

- Seed-42 matched Q1/Q2: previously backed up and verified.
- W3-G32: backed up and clean-process verified.
- Seed-43 matched Q1/Q2: checkpoint files reported lost; experiment results preserved.
- W3-G128: checkpoint files reported lost; experiment results preserved.

If lost checkpoints cannot be recovered, document that clearly. **Do not retrain simply to recreate them.**

### Acceptance

Produce a final checkpoint inventory distinguishing:

- Verified local/durable backups.
- Available only on a pod.
- Unavailable/lost.
- Experiment results preserved without model weights.

Never claim backup success without verified bytes and hashes.

---

## P1 — Complete the W3-G32 failure analysis using existing outputs

We have found an important behavior that deserves accurate documentation.

The W3-G32 QAD experiment produced:

- Validation PPL: 40.78.
- WikiText test PPL: 51.43.
- GSM8K: 0/100.
- 31% natural completion, 69% truncation.
- 0% successfully parsed final answers.

However, inspection of the saved generations shows that **30 of the 31 naturally completed responses ended within 10 tokens**, frequently producing only malformed control-token sequences such as `</think>`, `<think>`, or `<tool_call>`.

This is premature termination, not recovery of usable generation.

### Required investigation

Use the existing saved generation JSONLs and training data.

**A. Classify W3-G32 generation failures.**

For all 100 GSM8K examples, report:

- Premature termination with special/control tokens.
- Truncation without a final answer.
- Repetitive or incoherent generation.
- Unexpected thinking/chat-control markers.
- Malformed final-answer formatting.
- Other identifiable categories.

Categories may overlap; state the counting methodology.

Include 3–5 representative anonymized examples, without implying that they represent the whole distribution.

**B. Audit training/evaluation formatting.**

The training data was prepared using Qwen3's chat template with `enable_thinking=True`, while the primary strict GSM8K evaluation uses `enable_thinking=False`.

Inspect:

- Actual decoded training prompt IDs.
- Gold completion prefixes and endings.
- Whether the gold responses contain thinking markers or other chat-control tokens.
- Whether the assistant continuations follow the expected chat-template format.
- Whether response truncation and EOS are handled correctly.
- Whether this training/evaluation mode mismatch plausibly contributes to W3 failures.

Importantly, distinguish:

1. A proven formatting or tokenization bug.
2. A plausible training-distribution mismatch.
3. Damage caused by aggressive quantization that has not been causally isolated.

Do not claim that the mode mismatch caused the failures unless the existing evidence establishes that.

**C. Compare against existing BF16/W4/W3-G128 outputs.**

Use the existing generation records to compare completion behavior, repetition, special-token emission, and truncated responses.

No model loading or GPU inference is necessary unless saved artifacts cannot answer an essential question.

### Acceptance

Produce a concise `reports/w3_failure_analysis.md` with quantitative evidence, representative outputs, hypotheses, limitations, and a proposed future experiment.

Do not implement the future experiment.

---

## P2 — Final correctness and methodology verification

Perform a focused audit; do not restart the full research process.

Verify the following:

1. Q1 and Q2 matched-run checks pass for both seeds 42 and 43, using the corrected matching checker.
2. The frozen GSM8K parser v3.0.0 and canonical result files are consistent.
3. Per-example counts agree with summary files, paired comparisons, and provenance.
4. Qwen model revisions, quantization settings, seeds, checkpoint hashes, and evaluation data hashes are correctly recorded.
5. Validation and test splits remain disjoint.
6. Best-checkpoint selection used validation metrics only.
7. Failed or interrupted training runs are correctly distinguished from completed runs.
8. The report does not confuse theoretical packed storage, actual safetensors size, and measured GPU memory.
9. Automated unit tests pass in the available environment.

### Small evaluator caveat

Some generated outputs use formatting such as:

`#### Final Answer:` followed by `#### 18`.

Parser v3 may reject these because the first line resembles a malformed numeric answer marker.

Conduct a small manual audit of the existing output classifications across models, including correctly parsed and rejected examples.

**Do not change the frozen parser just to improve scores.** If a genuine issue is confirmed, document its possible impact and propose a separately versioned correction. Do not silently modify historical metrics.

Also ensure we distinguish:

- Strict GSM8K accuracy on the fixed 100-example subset.
- Parsing/completion performance.
- Teacher-forced NLL/perplexity.
- Actual open-ended generation quality.

These measure different properties.

### Acceptance

Produce a short final audit record listing verified invariants, tests executed, remaining limitations, and any unresolved correctness concerns.

Fix only clear, low-risk bugs necessary for trustworthy reporting.

---

## P3 — Produce one authoritative final research report

We currently have research information spread across many reports and experiment gates.

Create a new primary document:

`reports/FINAL_RESEARCH_REPORT.md`

Make it concise, scientifically rigorous, and easy for an ML researcher or founder to read.

Suggested sections:

### 1. Executive summary

Explain in a few paragraphs:

- The research question.
- What was implemented.
- The most important results.
- What failed.
- The strongest proposed next research question.

### 2. Experimental setup

Document:

- Qwen3-0.6B and Qwen3-1.7B revisions.
- BF16 teacher.
- W3/W4 quantization configurations.
- Quantized module coverage.
- FP32 master weights and STE.
- Offline forward-KL distillation.
- On-policy forward-KL distillation with fresh student rollouts.
- Training data, validation, independent test data.
- Optimizer, training budgets, and GPU hardware.

Explicitly state that the online implementation uses forward KL on student-generated prefixes and is not an exact reproduction of every OnlineQAT algorithm.

### 3. Main results

Present the key comparisons.

**Qwen3-0.6B, W4-G128:**

- BF16: test PPL 25.52, GSM8K 39%.
- RTN: test PPL 36.79, GSM8K 17%.
- Offline QAD seed 42: PPL 30.64, GSM8K 33%.
- Offline QAD seed 43: PPL 30.68, GSM8K 35%.
- Online QAD seed 42: PPL 29.92, GSM8K 35%.
- Online QAD seed 43: PPL 30.72, GSM8K 43%.

Clarify that online QAD does not demonstrate a statistically established advantage and incurs additional rollout compute.

**W3 experiments:**

- W3-G128: test PPL 88.24 after QAD, GSM8K 0%.
- W3-G32: test PPL 51.43 after QAD, GSM8K 0%.
- Discuss the real improvement in likelihood from smaller groups and the remaining generation failure.

**Qwen3-1.7B, W4-G128:**

- BF16: test PPL 19.53, GSM8K 58%.
- RTN: test PPL 25.41, GSM8K 44%.
- Offline QAD: test PPL 19.47, GSM8K 42%.

Emphasize that the near-BF16 test perplexity does not imply recovered reasoning accuracy.

### 4. Main scientific findings

Highlight:

1. Offline QAD meaningfully recovers low-bit quantization damage.
2. On-policy forward-KL QAD has no established advantage under the tested conditions.
3. Reducing W3 group size improves likelihood but does not restore usable generation.
4. Teacher-forced likelihood and generative task quality can diverge substantially.
5. Aggressive W3 may exhibit malformed thinking/control-token behavior and premature termination.

### 5. Limitations

Include:

- Only two matched seeds for 0.6B W4.
- Only 100 GSM8K examples per run.
- 1.7B offline-only and single-seed experiment.
- Fixed 128-token training continuations.
- Thinking-mode subsets are diagnostic only.
- No causal proof of the cause of W3 generation failures.
- Fake quantization rather than deployed packed INT3/INT4 weights.
- Checkpoint availability limitations.
- Results cannot automatically generalize to larger models or alternative quantizers.

### 6. Proposed future research

Describe—but do not implement:

- Stronger W3 initialization such as GPTQ/AWQ or block-wise QAT.
- Preserving chat/thinking-control behavior under aggressive quantization.
- Studying on-policy QAD with longer rollouts or alternative loss objectives.
- Exporting and validating a truly packed low-bit representation.

Keep this section short, explicitly separating hypotheses from established results.

### Figures

Create a small number of useful plots:

- W4 teacher/RTN/offline/online quality recovery.
- W3-G128 versus W3-G32 likelihood and generation outcomes.
- Offline versus online accuracy across both seeds.
- Optional 1.7B likelihood-versus-GSM8K comparison.

Use existing JSON results. Never fabricate or interpolate values.

Avoid decorative plots that don't explain a research finding.

---

## P4 — Polish the public GitHub repository

Update `README.md` so someone visiting the repository immediately understands:

- What research question this project investigates.
- Why aggressive quantization and online QAD matter.
- The main result in a compact table.
- What was tested and what remains unproven.
- Where to read the final report.
- How to reproduce the environment, baselines, training, and evaluation.

Clean up contradictory or stale status claims in (keep everything latest):

- `README.md`
- `plan.md`
- `experiments/index.md`
- Historical reports, where necessary

Do not rewrite or delete historical experiment results. Mark obsolete results as historical and point readers to the canonical corrected reports.

Ensure the code is presentable and that no API keys, host connection details, private credentials, or temporary local paths are exposed in tracked files.

A fresh-clone documentation check or dependency/test smoke check is useful, but avoid downloading models or running training solely to demonstrate documentation quality.

**This is a finalization pass, not another research phase. Stop when the deliverables are trustworthy, coherent, and ready to present.**