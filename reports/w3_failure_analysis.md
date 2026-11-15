# W3-G32 generation failure analysis

All counts use the saved 100-example strict non-thinking GSM8K generations for
W3-G32 Q1-200 and the existing BF16/W4 records. No model was loaded.

## 1. Headline

30 of the 31 naturally completed responses ended within **10 tokens**, and all
31 consisted only of control-token sequences. This is premature termination,
not recovered generation.

| Measure | W3-G32 | W4 Q1 seed42 | BF16 |
|---|---:|---:|---:|
| Natural completion | 31% | 64% | 76% |
| Truncated at cap | 69% | 36% | 24% |
| Finished in ≤10 tokens | **30/100** | 0/100 | 0/100 |
| Any control marker | **99%** | 0% | 0% |
| Missing final marker | **99%** | 34% | 36% |
| Malformed marker | 1% | 3% | 5% |
| 4-gram repetition | **9.3%** | 3.7% | 2.7% |
| Mean response tokens | 178.5 | 205.4 | 185.2 |

## 2. Categories (overlapping)

Method: each response may satisfy several categories; percentages are of 100.

1. **Premature control-token termination - 30/100.** Finished naturally,
   ≤10 tokens, only `<think>`/`</think>`/`<tool_call>` sequences.
2. **Control-token loop after EOS-like close - 1/100.** A 21-token sequence of
   repeated `<think>`/`</think>`/`<tool_call>`.
3. **Truncated without a final answer - 69/100.** Hit the 256-token cap; none
   supplied a parsable final answer.
4. **Repetitive/incoherent generation - 69/100 truncated; mean 4-gram
   repetition 9.3%.** Qualitative inspection shows circular restatement.
5. **Malformed final-answer formatting - 1/100.**
6. **No terminal answer - 99/100.**

Representative outputs (anonymized; not representative of all failures):

```text
</think>

</think>

<think>

<think>

</think>

<think>

<tool_call>

<think>
```

The one non-short completion is a long control-token sequence, not reasoning.
Examples are illustrative only; the quantitative table is the evidence.

## 3. Training/evaluation formatting audit

Data (`data/cache/prompts/openthoughts_prompts.jsonl`, 3,000 records):

- **3,000/3,000** gold responses start with `<|begin_of_thought|>`.
- **0/3,000** contain `<think>` or `</think>`.
- **2,979/3,000** gold responses are truncated at 1,024 tokens; only **21**
  contain EOS.
- All training prompts end with `<|im_start|>assistant\n` (the
  `enable_thinking=True` generation prefix).
- The primary GSM8K evaluation uses `enable_thinking=False`, which pre-fills
  `<|im_start|>assistant\n<think>\n\n</think>\n\n`.

So the training targets use OpenThoughts control tokens and are usually
unfinished, while the evaluation prompt starts with a closed empty thinking
block. This is a **proven formatting/distribution mismatch**.

It is not proven to cause the W3 failure: W4 was trained on the same data and
emits no control tokens. The most defensible statement is that aggressive
quantization plus this mismatch plausibly makes the student fall into
control-token loops; the causal contribution is not isolated.

## 4. Hypotheses

1. W3 capacity/quantization damage makes the model unable to sustain a
   multi-token reasoning trajectory; it falls back to high-probability chat
   control tokens. (Consistent with W3-G128 and W3-G32 both failing.)
2. The training/eval mode mismatch and unfinished OpenThoughts targets create
   strong control-token priors that W4 can suppress but W3 cannot.
3. The 128-token training continuations and 256-token evaluation cap do not
   explain 10-token termination by themselves.

## 5. Proposed future experiment (not implemented)

Rebuild the training targets with Qwen-native thinking format: gold responses
begin with `<think>`, end with `</think>` plus a final answer, and are not
truncated mid-reasoning. Retrain one W3-G32 QAD run with the same matched
protocol. If control-token termination persists, the failure is primarily
quantization capacity, not formatting. This is a targeted follow-up, not a
broad sweep.

## 6. Limitations

- Single seed and one W3 checkpoint; 100 GSM8K examples.
- No W3-G128 generation JSONL was preserved, so its control-token rate cannot
  be measured (only its summary and 0% accuracy).
- Categories overlap; the counts are descriptive, not a causal model.
- The training data audit proves a mismatch, not its effect size.
