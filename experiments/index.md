# Experiment registry

Canonical corrected results are in `experiments/gate8` and `experiments/gate9`.
Historical artifacts (`gate0`–`gate7`) are preserved on disk but their GSM8K
numbers used the legacy parser and are not canonical.

## Canonical runs

| Run | Artifacts | Result |
|---|---|---|
| 0.6B W4 seed42 Q1/Q2 | `gate8/m_q*_w4_s42.*`, `regraded_v3/*s42*` | Q1 30.64 PPL / 33% GSM; Q2 29.92 / 35%; paired +2, p=0.77 |
| 0.6B W4 seed43 Q1/Q2 | `gate8/m_q*_w4_s43.*`, `regraded_v3/*s43*` | Q1 30.68 / 35%; Q2 30.72 / 43%; paired +8, p=0.17 |
| 0.6B W3-G128 Q1 | `gate5/gsm8k_w3g128_200.json` | test 88.24 PPL; GSM 0% |
| 0.6B W3-G32 Q1 | `gate9/*`, `reports/w3g32_results.md` | test 51.43 PPL; GSM 0%; failure analysis |
| 1.7B W4 Q1 | `gate8/*1p7b*`, `reports/1p7b_results.md` | test 19.47 PPL; GSM 42% |

## Method/infrastructure records

| Item | Location |
|---|---|
| Quantizer/loss/wrapping tests | `tests/` (175 collected) |
| Data manifest and splits | `data/manifests/gate8_splits_manifest.json` |
| Parser frozen v3.0.0 | `docs/parser_spec_v3.md` |
| Matched-run checker | `scripts/check_matched.py` |
| Result integrity | `experiments/gate8/regraded_v3/integrity_report*.json`, `experiments/gate9/regraded_v3/integrity_report*.json` |
| Theoretical vs measured memory | `reports/memory_accounting.md` |
| Final report | `reports/FINAL_RESEARCH_REPORT.md` |

## Historical pilot (superseded GSM8K)

`gate1`–`gate7` contain the first pilot: baseline, W3 degradation diagnosis,
offline/online QAD at 25/200 steps, seed replication, rollout-length ablation and
an independent 1.7B check. Their GSM8K numbers used a permissive parser and a
512-token cap; they are labelled historical in `reports/findings.md`. Do not mix
them with canonical strict results.
