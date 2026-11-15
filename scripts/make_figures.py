"""Generate final-report figures directly from canonical JSON metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def metric(path: str, key: str) -> float:
    return float(json.loads(Path(path).read_text())["metrics"][key])


def bars_with_labels(ax, labels, values, colors, ylabel, fmt="{:.1f}", ylim=None):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ylim if ylim is not None else max(values) * 1.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                fmt.format(value), ha="center", va="bottom", fontsize=8)


def two_panel(labels, ppl, gsm, title, filename, colors):
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.2))
    bars_with_labels(a, labels, ppl, colors, "WikiText test PPL (lower is better)", fmt="{:.2f}")
    bars_with_labels(b, labels, gsm, colors, "Strict GSM8K accuracy (%)", ylim=65)
    a.set_title("Test perplexity")
    b.set_title("GSM8K")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


# 0.6B W4
labels = ["BF16", "W4 RTN", "Q1 s42", "Q1 s43", "Q2 s42", "Q2 s43"]
ppl = [
    metric("experiments/gate8/test_t0.json", "perplexity"),
    metric("experiments/gate8/test_q0_w4.json", "perplexity"),
    metric("experiments/gate8/test_m_q1_w4_s42_verified.json", "perplexity"),
    metric("experiments/gate8/test_m_q1_w4_s43_verified.json", "perplexity"),
    metric("experiments/gate8/test_m_q2_w4_s42_verified.json", "perplexity"),
    metric("experiments/gate8/test_m_q2_w4_s43_verified.json", "perplexity"),
]
gsm_paths = [
    "experiments/gate8/regraded_v3/gsm8k_t0_nonthink.json",
    "experiments/gate8/regraded_v3/gsm8k_q0_w4_nonthink.json",
    "experiments/gate8/regraded_v3/gsm8k_m_q1_w4_s42_nonthink.json",
    "experiments/gate8/regraded_v3/gsm8k_m_q1_w4_s43_nonthink.json",
    "experiments/gate8/regraded_v3/gsm8k_m_q2_w4_s42_nonthink.json",
    "experiments/gate8/regraded_v3/gsm8k_m_q2_w4_s43_nonthink.json",
]
gsm = [metric(p, "accuracy") * 100 for p in gsm_paths]
two_panel(labels, ppl, gsm, "Qwen3-0.6B W4-G128: test likelihood and strict GSM8K",
          "fig_w4_recovery.png", ["#4d4d4d", "#b04a4a", "#2f6f9f", "#6aa84f", "#2f6f9f", "#6aa84f"])

# W3 vs W4
labels = ["W3-G128", "W3-G32", "W4 Q1", "BF16"]
ppl = [
    metric("experiments/gate8/test_m_q1_w3_s42_verified.json", "perplexity"),
    metric("experiments/gate9/test_m_q1_w3g32_s42_verified.json", "perplexity"),
    metric("experiments/gate8/test_m_q1_w4_s42_verified.json", "perplexity"),
    metric("experiments/gate8/test_t0.json", "perplexity"),
]
gsm = [
    metric("experiments/gate8/regraded_v3/gsm8k_m_q1_w3_s42_nonthink.json", "accuracy") * 100,
    metric("experiments/gate9/regraded_v3/gsm8k_m_q1_w3g32_s42_nonthink.json", "accuracy") * 100,
    metric("experiments/gate8/regraded_v3/gsm8k_m_q1_w4_s42_nonthink.json", "accuracy") * 100,
    metric("experiments/gate8/regraded_v3/gsm8k_t0_nonthink.json", "accuracy") * 100,
]
two_panel(labels, ppl, gsm, "W3 group size vs W4 recovery (0.6B, seed 42)",
          "fig_w3_granularity.png", ["#8c2f2f", "#c47f2f", "#2f6f9f", "#4d4d4d"])

# offline vs online
fig, ax = plt.subplots(figsize=(5, 3))
x = np.arange(2); w = 0.35
q1 = [metric("experiments/gate8/regraded_v3/gsm8k_m_q1_w4_s42_nonthink.json", "accuracy") * 100,
      metric("experiments/gate8/regraded_v3/gsm8k_m_q1_w4_s43_nonthink.json", "accuracy") * 100]
q2 = [metric("experiments/gate8/regraded_v3/gsm8k_m_q2_w4_s42_nonthink.json", "accuracy") * 100,
      metric("experiments/gate8/regraded_v3/gsm8k_m_q2_w4_s43_nonthink.json", "accuracy") * 100]
ax.bar(x - w / 2, q1, w, label="Offline Q1", color="#2f6f9f")
ax.bar(x + w / 2, q2, w, label="Online Q2", color="#6aa84f")
for i, (a, b) in enumerate(zip(q1, q2)):
    ax.text(i - w / 2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + w / 2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(["seed 42", "seed 43"])
ax.set_ylabel("Strict GSM8K accuracy (%)"); ax.set_ylim(0, 50)
ax.set_title("Offline vs online QAD on GSM8K")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_offline_online.png", dpi=180); plt.close(fig)

# 1.7B
labels = ["BF16", "W4 RTN", "W4 Q1"]
ppl = [
    metric("experiments/gate8/test_1p7b_t0.json", "perplexity"),
    metric("experiments/gate8/test_1p7b_q0_w4.json", "perplexity"),
    metric("experiments/gate8/test_m_q1_w4_1p7b_verified.json", "perplexity"),
]
gsm = [
    metric("experiments/gate8/regraded_v3/gsm8k_1p7b_t0_nonthink.json", "accuracy") * 100,
    metric("experiments/gate8/regraded_v3/gsm8k_1p7b_q0_nonthink.json", "accuracy") * 100,
    metric("experiments/gate8/regraded_v3/gsm8k_m_q1_w4_1p7b_nonthink.json", "accuracy") * 100,
]
two_panel(labels, ppl, gsm, "Qwen3-1.7B W4-G128 controlled check",
          "fig_1p7b.png", ["#4d4d4d", "#b04a4a", "#2f6f9f"])
