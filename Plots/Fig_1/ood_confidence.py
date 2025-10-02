import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Data ----------
data = {
    "OOD subset": ["Deep sea", "Shallow marine", "Freshwater"],
    "N": [113, 473, 20],
    "TPR_OOD": [0.283, 0.186, 0.000],
    "Avg.conf": [0.671, 0.820, 0.975],
}
df = pd.DataFrame(data)
y = np.arange(len(df))

# Global font settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 7
})

# Pastel colors
COLOR_N   = "#AEC6CF"   # pastel blue for N
COLOR_TPR = "#FFB347"   # pastel orange for TPR
COLOR_CONF= "#77DD77"   # pastel green for Confidence
EDGE      = "#444444"   # dark gray edges

# ---------- Helper: dual-axis bar plot ----------
def plot_dual_bar(df, metric_col, metric_label, metric_color, out_png):
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=1200)

    # Primary axis: N as bars
    bars_n = ax.barh(y, df["N"], height=0.45, color=COLOR_N, edgecolor=EDGE,
                     label="Sample Count (N)", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["OOD subset"])
    ax.invert_yaxis()
    ax.set_xlabel("Sample Count (N)")

    # Secondary axis: metric as bars (scaled 0–1)
    ax2 = ax.twiny()
    bars_m = ax2.barh(y, df[metric_col], height=0.25, color=metric_color, edgecolor=EDGE,
                      label=metric_label, zorder=3)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel(f"{metric_label} (0–1)")

    # Legends
    handles = [bars_n, bars_m]
    labels  = ["Sample Count (N)", metric_label]
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",          # anchor position
        bbox_to_anchor=(0.5, -0.15), # move below the axis (0.5 = centered, -0.15 = below)
        fontsize=7,
        ncol=2                       # two columns for compact layout
    )

    # Grid
    ax.xaxis.grid(True, linestyle=":", linewidth=0.6, color="gray", alpha=0.5, zorder=1)
    ax2.xaxis.grid(False)

    # Clean spines
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    # Annotations
    # for i, (n, m) in enumerate(zip(df["N"], df[metric_col])):
    #     ax.text(n, i - 0.15, f"{n}", va="center", ha="left", color=EDGE, fontsize=7)
    #     ax2.text(m, i + 0.15, f"{m:.2f}", va="center", ha="left", color=EDGE, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_png, dpi=1200, bbox_inches="tight", transparent=True)
    plt.close(fig)

# ---------- Generate & save ----------
plot_dual_bar(df, "TPR_OOD", "TPR (OOD)", COLOR_TPR,
              "/home/reshma/MORPHID/Plots/Fig_1/plots/ood_dualbar_TPR.png")

plot_dual_bar(df, "Avg.conf", "Average Confidence", COLOR_CONF,
              "/home/reshma/MORPHID/Plots/Fig_1/plots/ood_dualbar_Confidence.png")
