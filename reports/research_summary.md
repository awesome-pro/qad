# Research summary — W4 recovery, offline vs online QAD, W3 failure

## 1. What is established

Parser v3.0.0 strict GSM8K; disjoint FineWeb validation / WikiText test; W4A16-G128;
matched Q1/Q2 prompt schedules; two seeds for the 0.6B W4 comparison.

### W4 recovery (0.6B, strict non-thinking, 100 questions)

| Variant | Seed | Test PPL | GSM8K |
|---|---:|---:|---:|
| T0 BF16 | — | 25.52 | 39% |
| Q0 W4 RTN | — | 36.79 | 17% |
| Q1-200 offline | 42 | 30.64 | 33% |
| Q1-200 offline | 43 | 30.68 | 35% |
| Q2-200 online | 42 | 29.92 | 35% |
| Q2-200 online | 43 | 30.72 | 43% |

Offline Q1 recovers most of the W4 test-PPL damage (36.8 → ~30.7) and about
half to two-thirds of the GSM8K gap (17% → 33–35% vs T0 39%).

### Offline vs online

- Test likelihood is tied: seed 42 Q2 slightly better, seed 43 Q1 slightly
  better.
- GSM8K: Q2 is +2 points (p=0.77) and +8 points (p=0.17) across seeds. No
  statistically established advantage; the sign is consistent but the effect
  is seed-dependent.
- Conclusion: **no reliable evidence that on-policy QAD beats matched offline
  QAD at this scale/rollout budget.** It is not worse on likelihood and may
  help generation for some seeds, but the current evidence does not justify
  preferring it.

### W3 failure

| Run | Val PPL | Test PPL | GSM8K |
|---|---:|---:|---:|
| W3 Q0 RTN | 724 | — | ~0–1% |
| W3 Q1-25 | 83.6 | — | 3% |
| W3 Q1-200 (matched) | 62.4 | 88.2 | **0%** (99% truncation) |

QAD recovers W3 held-out likelihood by an order of magnitude but does not
produce usable generative behavior. W3 is far from W4 (test 88 vs 31 PPL) and
does not emit terminal answers. Likelihood recovery is not sufficient for
aggressive quantization.

## 2. Interpretation

- W4 is the tractable operating point; offline QAD is the reliable recovery
  method.
- Online QAD does not reliably improve over matched offline QAD; the remaining
  seed-43 GSM signal is interesting but not decisive.
- W3 failure is most plausibly a representation-capacity/initialization problem,
  not only an optimization-budget problem: QAD can fit teacher conditionals on
  held-out text while the 3-bit student still degenerates in long generation.

## 3. Highest-value next experiment: stronger W3 initialization

The unresolved question is how to make truly aggressive quantization preserve
useful behavior. Candidates, in recommended order:

1. **W3-G32** — smaller groups reduce quantization error; same QAD recipe,
   directly tests whether group granularity closes the W3 generation gap.
2. **GPTQ/AWQ W3 initialization** — data-aware rounding before QAD; tests
   whether better initialization, not more QAD, is the missing ingredient.
3. **Block-wise QAT initialization** — closest to the brief's Stage-1 idea, but
   larger engineering cost; only after (1)/(2).

Evaluate with the same matched protocol: validation-selected checkpoint,
disjoint WikiText test, strict GSM8K non-thinking, and the fixed generation
diagnostic. Do not add broad benchmarks before these comparisons.

## 4. 1.7B

Worthwhile as a controlled scaling check: T0/W4 baselines, tokenizer
compatibility, memory/gradient probe, then corrected offline Q1 only. No online
sweep in the handoff. This tests whether the W4 recovery pattern persists with
scale, without displacing the W3-init research question.

## 5. Limitations

Single corrected seed until more replicates; 100 GSM8K questions; thinking mode
only a 10-question diagnostic; WikiText domain shift/contamination; old
28%/14% gap not causally decomposed; fake quantization gives no serving speedup.
Seed-43/W3 best-checkpoint durability is currently blocked (pod unreachable);
results are preserved but the weights are not verified backups.
