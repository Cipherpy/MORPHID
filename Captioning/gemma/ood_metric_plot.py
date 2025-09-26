#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------- INPUT ----------------
data = {
    "Metric":   ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L"],
    "ID mean":  [0.879,    0.816,    0.765,    0.723,    0.836],
    "OOD mean": [0.761,    0.634,    0.545,    0.475,    0.698],
    # Optional: 95% CI tuples; set to None or omit if unavailable
    "ID_CI":    [(0.875, 0.883), (0.810, 0.821), (0.758, 0.771), (0.716, 0.730), (0.830, 0.842)],
    "OOD_CI":   [(0.756, 0.766), (0.629, 0.640), (0.539, 0.552), (0.468, 0.482), (0.693, 0.703)],
}
df = pd.DataFrame(data)
df["Gap (G)"] = df["ID mean"] - df["OOD mean"]
df["rho"]     = df["OOD mean"] / df["ID mean"]
df["% drop vs ID"] = 100.0 * df["Gap (G)"] / df["ID mean"]

# ---------------- STYLE ----------------
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
})

COLOR_ID  = "#1f77b4"  # blue
COLOR_OOD = "#d62728"  # red
COLOR_GAP = "#7f7f7f"  # gray
COLOR_RHO = "#2ca02c"  # green

# ==========================================================
# FIGURE 1: Grouped BAR chart for ID vs OOD means (+ optional CI)
# ==========================================================
def plot_grouped_means(df: pd.DataFrame, fname="fig_means_grouped.png"):
    metrics = df["Metric"].tolist()
    x = np.arange(len(metrics))
    w = 0.36

    id_vals  = df["ID mean"].values
    ood_vals = df["OOD mean"].values
    gap_vals = df["Gap (G)"].values

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    b1 = ax.bar(x - w/2, id_vals,  width=w, label="ID mean",  color=COLOR_ID,  alpha=0.95, zorder=3)
    b2 = ax.bar(x + w/2, ood_vals, width=w, label="OOD mean", color=COLOR_OOD, alpha=0.90, zorder=3)

    # Optional CI whiskers if present
    if "ID_CI" in df.columns:
        for i, (lo, hi) in enumerate(df["ID_CI"]):
            if lo is None or hi is None: continue
            ax.errorbar(x[i] - w/2, id_vals[i],
                        yerr=[[id_vals[i]-lo], [hi-id_vals[i]]],
                        fmt='none', ecolor='k', elinewidth=0.8, capsize=2, zorder=4)
    if "OOD_CI" in df.columns:
        for i, (lo, hi) in enumerate(df["OOD_CI"]):
            if lo is None or hi is None: continue
            ax.errorbar(x[i] + w/2, ood_vals[i],
                        yerr=[[ood_vals[i]-lo], [hi-ood_vals[i]]],
                        fmt='none', ecolor='k', elinewidth=0.8, capsize=2, zorder=4)

    # Δ label above pairs
    for i in range(len(x)):
        ymax = max(id_vals[i], ood_vals[i])
        ax.text(x[i], ymax + 0.035, f"Δ={gap_vals[i]:.3f}",
                ha="center", va="bottom", fontsize=8, color=COLOR_GAP)

    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Caption–reference similarity (BLEU / ROUGE)")
    ax.set_xlabel("Metric")
    ax.set_title("ID vs OOD means (grouped bars)", pad=8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", transparent=True)
    plt.show()

# ==========================================================
# FIGURE 2: Horizontal LOLLIPOP for Stability (ρ = OOD/ID)
# ==========================================================
def plot_stability_lollipop(df: pd.DataFrame, fname="fig_stability_lollipop.png"):
    metrics = df["Metric"].tolist()
    rho = df["rho"].values
    y = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    # Reference lines
    ax.axvline(1.0, color="black", lw=1.0, alpha=0.6)
    ax.axvline(0.8, color="#bbbbbb", lw=0.8, ls="--", alpha=0.8)

    # Stems + markers
    for yi, r in zip(y, rho):
        ax.plot([0, r], [yi, yi], color="#bbbbbb", lw=2, zorder=1)
    ax.scatter(rho, y, s=70, color=COLOR_RHO, zorder=3, label="ρ = OOD/ID")

    # Labels near markers
    for yi, r in zip(y, rho):
        off = 0.01 if r < 0.98 else -0.01
        ha  = "left" if r < 0.98 else "right"
        ax.text(r + off, yi, f"{r:.3f}", ha=ha, va="center", fontsize=8)

    ax.set_yticks(y, metrics)
    ax.set_xlim(0.0, max(1.05, rho.max() + 0.1))
    ax.set_xlabel("Stability ratio (ρ)")
    ax.set_title("Stability across metrics (lollipop)", pad=8)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", transparent=True)
    plt.show()

# ==========================================================
# FIGURE 3: Signed WATERFALL for Gap (Δ = ID − OOD) + % drop labels
# ==========================================================
def plot_gap_waterfall(df: pd.DataFrame, fname="fig_gap_waterfall.png"):
    # If any Δ could be negative in the future, this handles both signs.
    metrics = df["Metric"].tolist()
    gap = df["Gap (G)"].values
    pct = df["% drop vs ID"].values

    fig, ax = plt.subplots(figsize=(6.8, 3.8))

    colors = [COLOR_GAP if g >= 0 else "#bdbdbd" for g in gap]
    bars = ax.bar(metrics, gap, color=colors, alpha=0.85, zorder=3)

    # Annotate Δ and % drop
    for rect, g, p in zip(bars, gap, pct):
        x = rect.get_x() + rect.get_width()/2
        y = rect.get_height()
        va = "bottom" if g >= 0 else "top"
        pad = 0.015 if g >= 0 else -0.015
        ax.text(x, y + pad, f"Δ={g:.3f}\n({p:.1f}% of ID)", ha="center", va=va, fontsize=8)

    # Zero baseline for signed view
    ax.axhline(0, color="black", lw=1.0, alpha=0.6)
    ylim_top = max(0.05, gap.max()) + 0.12
    ylim_bot = min(0.0, gap.min() - 0.05)
    ax.set_ylim(ylim_bot, ylim_top)

    ax.set_ylabel("Gap (Δ = ID − OOD)")
    ax.set_title("Generalization gap (signed waterfall)", pad=8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", transparent=True)
    plt.show()

# ---------------- RUN ----------------
if __name__ == "__main__":
    plot_grouped_means(df, "fig_means_grouped.png")
    plot_stability_lollipop(df, "fig_stability_lollipop.png")
    plot_gap_waterfall(df, "fig_gap_waterfall.png")
