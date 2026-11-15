# Task: Final Research Audit, Corrections, and Validation — Qwen Low-Bit QAD

## Context

Repository: https://github.com/awesome-pro/qad

Reference commit: `f18f59976b24b083f28322406a11cc911560c8d3`

Primary report: `reports/findings.md`

We have completed a substantial research pilot investigating low-bit Qwen quantization and quantization-aware distillation (QAD), including offline forward-KL distillation and on-policy forward-KL distillation.

Experiments have been run on Qwen3-0.6B and Qwen3-1.7B, including W3/W4 quantization, 25-step and 200-step training, second-seed replication, and rollout-length ablations.

**The project is largely implemented, but an external review identified several issues that must be corrected before we consider the findings scientifically reliable.**

Your task is to audit the existing implementation, fix these issues, execute the necessary validation experiments, and update the documentation.

Do not start the project over. Reuse the existing implementation, checkpoints, datasets, and experiment infrastructure wherever possible.

Preserve all original results as historical artifacts. New experiments must have distinct run names and output locations.

---

# Priority 0 — Fix the GSM8K evaluation

**This is the highest-priority issue.**

### Problem

The existing `scripts/eval_gsm8k.py` has two major limitations:

1. It evaluates Qwen3 in thinking mode with `max_new_tokens=512`. Nearly all generations hit the token limit before completing their reasoning.
2. When the output does not contain an explicit `####` answer, `extract_prediction()` falls back to the last numerical value anywhere in the text.

This means an intermediate calculation from an unfinished reasoning trace can be incorrectly classified as the model's final answer.

The existing GSM8K results (e.g., Q1-200 28% versus Q2-200 14%) must therefore be considered exploratory, not validated reasoning accuracy.

### Required fixes

**A. Implement a strict answer parser.**

The new parser must:

- Distinguish reasoning content from final-answer content.
- Never automatically use the last number from arbitrary unfinished reasoning.
- Support explicitly formatted final answers, including `#### number`.
- Correctly handle decimals, negative values, commas, and other relevant numeric formatting.
- Identify missing or ambiguous final answers.
- Report malformed/incomplete generations as such.

**B. Implement a non-thinking-mode benchmark.**

Use Qwen3's official chat-template mechanism with `enable_thinking=False`.

- Use consistent prompts and generation settings for every variant.
- Request a final numeric answer in a clearly defined format.
- Grade only an explicitly identified final answer.
- Record completion, truncation, parsing, and correctness separately.

**C. Implement a separate thinking-mode benchmark.**

- Use `enable_thinking=True`.
- Increase the generation cap sufficiently to permit substantially longer reasoning.
- Track whether the model closes its thinking section and produces a final answer.
- Distinguish model-generated EOS or end-of-message tokens from reaching the generation cap.
- Report the actual completion rate.
- Do not treat incomplete reasoning as a final answer.

Use the correct special-token and chat-template semantics for the pinned Qwen3 tokenizer. Verify them with tests rather than assuming plain-text markers always behave the same way.

**D. Store enough information for independent regrading.**

For every example, save:

- Example identifier and question identifier.
- Model/checkpoint and generation configuration.
- Full generated token IDs or an equivalent lossless artifact.
- Complete decoded output, not just the first 600 characters.
- Extracted final answer.
- Gold answer.
- Whether generation finished naturally.
- Whether the answer was explicit and valid.
- Whether the answer was correct.

Avoid storing only truncated text snippets.

**E. Re-evaluate the relevant checkpoints.**

At minimum:

- BF16 teacher.
- W4-G128 RTN baseline.
- Q1-25 and Q2-25 (matched-length pair).
- Q1-200 and Q2-200.
- Seed-43 Q1-200 and Q2-200.
- W3-G128 Q1-200, if the evaluation budget permits.
- Qwen3-1.7B baseline and Q1-200, as a separate exploratory comparison.

Prioritize non-thinking evaluation first, followed by a correctly evaluated thinking-mode subset.

If feasible, expand beyond 100 GSM8K questions. Keep a fixed, predeclared subset for controlled comparisons, and do not repeatedly select checkpoints based on that subset.

### Acceptance criteria

- The parser never extracts an intermediate number from unfinished reasoning as a final answer.
- Truncation and completion are correctly distinguished.
- Tests cover completed answers, incomplete thinking, misleading intermediate numbers, malformed outputs, and missing final answers.
- The evaluation can be reproduced from saved outputs.
- The new report clearly labels the old GSM8K results as invalid for strict final-answer accuracy.

---

# Priority 1 — Correct the offline-versus-online comparison

### Problem

The existing 200-step Q1/Q2 comparison matches optimizer steps and response-token budgets, but it does not completely isolate the source of training prefixes.

Current differences include:

- Q1 uses gold continuations up to 1,024 tokens.
- Q2 uses student-generated continuations up to 128 tokens.
- Q1 sees approximately 800 prompt instances over 200 steps.
- Q2 sees approximately 6,400 prompt instances.
- Context lengths, number of forward passes, and prompt coverage therefore differ.

Consequently, the 200-step results cannot be interpreted as a clean causal measurement of on-policy versus offline training.

### Required corrections

**A. Add a rigorously matched experiment configuration.**

Both training arms should use:

- The same initial quantized checkpoint.
- The same teacher.
- The same quantizer and group size.
- The same loss direction and temperature.
- The same learning rate, optimizer, and schedule.
- The same number of optimizer steps.
- The same seed-controlled prompt order.
- The same prompt IDs per step.
- The same continuation-length cap.
- Approximately equal numbers of scored response tokens.

The essential difference must be:

Q1: teacher/student forward KL on fixed gold continuations.

Q2: teacher/student forward KL on continuations generated by the current student.

Prompt-instance counts and actual token counts should be logged, not assumed.

**B. Verify prompt-level matching.**

The current scripts use different batching and traversal logic.

Ensure that Q1 and Q2 genuinely receive the same prompt identifiers at corresponding training steps.

Store per-step prompt IDs or hashes and add an automated validation.

**C. Add a length-matched 200-step comparison.**

The existing 25-step experiment already provides a useful matched-length comparison.

Extend this methodology to the longer training configuration. Start with the existing 128-token continuation length.

A 256-token comparison is optional if resources permit.

Do not silently pad, duplicate, or otherwise alter examples simply to force identical token counts.

**D. Record training efficiency separately.**

Measure:

- Total optimizer steps.
- Number of prompt instances.
- Unique prompts encountered.
- Scored response tokens.
- Prompt/prefill tokens processed.
- Total model tokens processed.
- Rollout generation time.
- Teacher forward time, where instrumentable.
- Student training time.
- End-to-end GPU time.
- Peak allocated GPU memory.

The existing online/offline runtime ratio is interesting, but it should not be described as the inherent overhead of online QAD because the workloads differ.

**E. Distinguish equal-token and equal-compute comparisons.**

Equal-response-token comparisons answer one research question.

Equal-GPU-compute comparisons answer another.

Implement equal-token matching first. If a compute-matched experiment is not feasible, document it as a limitation rather than pretending both are equivalent.

### Acceptance criteria

- An automated comparison verifies that the intended configurations match.
- The primary experimental difference is the source of continuation tokens.
- Both training arms have the same prompt-visit schedule.
- All residual mismatches are quantified.
- Any previous causal claims are revised appropriately.

---

# Priority 2 — Establish independent validation and test sets

### Problem

The existing training scripts periodically evaluate using an initial subset of `fineweb_eval_ids.npy`.

The final evaluation uses all 256 blocks from the same file.

Therefore, the final evaluation includes examples previously used for validation and training-progress inspection.

These are not training-gradient examples, but they are not completely independent final-test examples either.

### Required changes

1. Create a deterministic, disjoint validation set and test set.
2. Prefer a newly prepared independent final evaluation set instead of simply relabeling a previously inspected subset.
3. Pin dataset revisions and store content hashes.
4. Verify no sequence/block overlap between validation and test data.
5. Keep evaluation sequences separate from QAD training completions.
6. Use validation only for checkpoint selection and hyperparameter decisions.
7. Report final metrics on the held-out test set.

The previously reported FineWeb metrics must be preserved but correctly labeled as results from the original evaluation protocol.

### Acceptance criteria

- No overlap between the new validation and test sets.
- Explicit hashes and manifests.
- Training code never uses final-test metrics for checkpoint selection.
- Final metrics are reproducible.

---

# Priority 3 — Improve checkpoint selection and restoration

### Problem

The 200-step offline QAD training logs show that intermediate checkpoints sometimes have better validation NLL than the final checkpoint.

The implementation currently emphasizes saving the final checkpoint.

### Required changes

1. Save checkpoints at evaluation intervals.
2. Track the best checkpoint using a declared primary validation metric.
3. Retain the final checkpoint separately.
4. Record the selection metric, selected step, and checkpoint hash.
5. Evaluate both best and final checkpoints when useful.
6. Confirm checkpoint reload reproduces logits and evaluation metrics.
7. Save all necessary model/quantizer metadata for reconstruction.
8. Avoid evaluating or selecting checkpoints using final-test performance.

For Q1 and Q2, use identical checkpoint-selection rules.

### Acceptance criteria

- Best and final checkpoints are distinguishable.
- Reloaded checkpoints reproduce their recorded results within numerical tolerance.
- No checkpoint is selected based on final-test data.

---

# Priority 4 — Preserve trained weights and verify reproducibility

### Problem

The research report states that checkpoints remain on the GPU pod and are not committed to GitHub.

That is sensible, but there is a risk that the weights may be lost when the pod is terminated.

### Required changes

- Verify that all important trained checkpoints still exist.
- Preserve the key Q1/Q2 checkpoints in durable storage.
- Record SHA-256 hashes and byte sizes.
- Save model revision, quantization config, optimizer config, and training metadata.
- Confirm that the checkpoint can be restored in a clean process.
- Prefer safe, portable inference weights such as `safetensors` where appropriate; keep optimizer state separately for training resumption.
- Document the process for reconstructing the student from the original Qwen checkpoint and trained master weights.

Do not commit multi-GB checkpoints to the normal Git repository.

Do not publish trained weights or present them as an official Tear Labs release without explicit agreement.

### Acceptance criteria

- Important checkpoints are durably preserved.
- Clean-process restoration succeeds.
- Artifacts have hashes and metadata.
- Reproduction instructions do not depend on a still-running pod.

---

# Priority 5 — Audit generation and on-policy training correctness

The existing implementation appears conceptually correct, but verify these properties explicitly.

### Required checks

1. The teacher remains frozen, has no gradients, and stays in evaluation mode.
2. The student is sampled from its current quantized parameters after each optimizer update.
3. Cached dequantized weights are invalidated correctly.
4. Fresh rollouts are not incorrectly reused across optimizer steps.
5. Left padding and attention masks do not leak into teacher/student loss calculations.
6. Loss positions align exactly with the sampled response tokens.
7. `train()` and `eval()` modes are handled consistently during training, generation, and validation.
8. Sampling seeds and generation configurations are recorded.
9. EOS, end-of-message, and truncation behavior are handled correctly.
10. Gradient norms and losses are finite; master weights receive nonzero updates.

Use focused tests, including a tiny causal model where a controlled weight update changes the sampled distribution.

A changing rollout hash alone is not sufficient proof that the current weights are being used.

---

# Priority 6 — Clarify memory and quantization claims

### Required changes

Separate these measurements explicitly:

**A. Actual research/training memory**

- CUDA peak allocated memory.
- CUDA peak reserved memory, if useful.
- Master-weight storage.
- Optimizer state.
- Teacher parameters.
- Checkpoint size.

**B. Theoretical compressed-model storage**

- Quantized-weight payload.
- Group scales.
- Unquantized embedding/LM-head/norm parameters.
- Effective bits per weight.
- Whole-model effective size.

**C. Actual packed inference storage**

Report only if a real packed implementation is created and measured.

Our current fake quantization does not demonstrate actual packed-weight memory savings or inference acceleration.

Do not claim that training GPU memory or runtime should improve merely because the forward pass uses simulated W3/W4 quantization.

If practical, provide a small table comparing theoretical whole-model storage for BF16, W4-G128, W3-G32, and W3-G128.

A real packed model export is optional and should be treated as a separate workstream, not invented as a completed result.

---

# Priority 7 — Statistical and reporting quality

### Required changes

1. Preserve the two training seeds and clearly identify where only one seed exists.
2. Add uncertainty estimates for benchmark accuracy.
3. Use paired per-question comparisons for Q1 versus Q2, where the same test questions were evaluated.
4. Avoid claiming that a 2–3 percentage-point advantage on 100 questions is established.
5. Distinguish statistical uncertainty from variation between training seeds and generation samples.
6. Report exact evaluation configurations and sample counts.
7. Distinguish observations, interpretations, and hypotheses.

Avoid claims such as:

- "Online QAD is worse in general."
- "The 3-bit student is fundamentally unrecoverable."
- "QAD restores full reasoning ability."
- "Fake quantization produces deployment memory savings."

Unless new experiments genuinely establish those conclusions.

For W3-G128, the defensible result is that likelihood improved substantially with offline QAD but quality remained far below BF16/W4 under the tested recipe.

For Qwen3-1.7B, clearly label the experiments exploratory because offline and online budgets are unmatched.

---

# Priority 8 — Update the final research report

Update:

- `reports/findings.md`
- `README.md`
- `plan.md`
- `experiments/index.md`
- Relevant reproduction instructions

Remove stale statements such as "No QAD training has run" or scope descriptions that incorrectly imply the main completed W4 experiments were W3.

The final report should contain:

1. Research objective.
2. Models, quantization configurations, and training methods.
3. Dataset provenance and train/validation/test separation.
4. Verified quantization baselines.
5. Offline-QAD results.
6. On-policy-QAD results.
7. Fair matched comparisons.
8. Corrected generation benchmarks.
9. Training efficiency and memory accounting.
10. W3 findings.
11. Exploratory 1.7B findings.
12. Negative findings and limitations.
13. Reproduction instructions.
14. Remaining research questions.

Provide a concise executive summary suitable for discussion with a research engineer or founder.

**The final document should clearly separate what has been experimentally established from what remains a hypothesis.**

---

# Implementation and execution rules

- Review the relevant existing code and artifacts before making modifications.
- Reuse the current repository architecture.
- Implement fixes in reviewable, logically separated commits.
- Add regression tests for every correctness bug.
- Preserve historical results; never silently overwrite or rewrite earlier experiment artifacts.
- Give new evaluations distinct run names and manifests.
- Do not fabricate metrics or infer successful results from expected behavior.
- Check for source-data overlap and checkpoint/config mismatches.
- Keep GPU usage controlled and recorded.
- Run cheap correctness checks before expensive experiments.
- Do not launch a large training sweep automatically. Establish the corrected protocol and complete small validation runs first.
- Existing checkpoints should be reused for evaluation corrections; only retrain when the methodological correction requires it.
- There is no deadline or timeline requirement. Optimize for rigor, correctness, and efficient execution.

# Completion criteria

Consider this task complete only when:

1. The GSM8K grader has been corrected and the required checkpoints have been re-evaluated.
2. The matched offline/online comparison has been implemented, tested, and executed on the agreed primary configuration.
3. The new validation and final-test data are disjoint.
4. Best-checkpoint handling is implemented and verified.
5. Important checkpoints are durably preserved and reload correctly.
6. All relevant automated tests pass.
7. The findings report reflects the corrected results, including any conclusions that change.
8. Reproduction commands and experiment artifacts are documented.

If any criterion cannot be completed because of missing checkpoints, unavailable data, or compute restrictions, explicitly mark it incomplete and document the reason. Do not claim completion without evidence.

# Expected output

At the end, provide:

- A summary of bugs and methodological issues found.
- Exact files and commits changed.
- Test results.
- Experiments actually executed and their outputs.
- Updated comparison tables.
- Revised scientific conclusions.
- Remaining limitations.
- The most valuable next research question.

The goal is not to make online QAD look better or to force a positive result.

**The goal is to make the research trustworthy, reproducible, and defensible.**