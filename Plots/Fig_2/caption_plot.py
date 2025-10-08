#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# ───────────── Config (Nature-style) ─────────────
FIG_W_MM, FIG_H_MM = 180, 160   # ~two-column width
FIG_W_IN, FIG_H_IN = FIG_W_MM/25.4, FIG_H_MM/25.4

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 1200,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize":7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,    # editable text in Illustrator
    "ps.fonttype": 42
})

# ───────────── Inputs ─────────────
CSV_FILE = "/home/reshma/MORPHID/Plots/Fig_2/caption_scores_llama.csv"
LABEL_COL = "actual_label"
METRICS   = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L"]
OUTDIR    = "fig_species_scores"
os.makedirs(OUTDIR, exist_ok=True)

# ───────────── Load & aggregate ─────────────
df = pd.read_csv(CSV_FILE)
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
species_stats = df.groupby(LABEL_COL)[METRICS].mean().sort_index()
species_counts = df.groupby(LABEL_COL).size().rename("N")

# helper: italicize genus + epithet, keep “Genus sp.” as “Genus sp.”
def italicize_taxon(name: str) -> str:
    txt = name.strip()
    parts = txt.split(" ", 1)
    if len(parts) == 2 and parts[1].strip().lower().startswith("sp"):
        return r"$\it{" + parts[0] + r"}$ " + parts[1]
    elif len(parts) == 2:
        return r"$\it{" + parts[0] + r"}$ " + r"$\it{" + parts[1] + r"}$"
    else:
        return r"$\it{" + txt + r"}$"

# save the species × metric mean table
species_stats.to_csv(os.path.join(OUTDIR, "species_scores_matrix.csv"), float_format="%.4f")

# ───────────── Option A: Clustered heatmap ─────────────
# Normalize per-metric for clustering (z-score), but annotate with real values.
Z = species_stats.apply(lambda col: zscore(col, nan_policy="omit"))
# Keep same index/columns
annot_vals = species_stats.copy()

# Use seaborn clustermap (creates its own fig); we align font sizes & style
cg = sns.clustermap(
    Z,
    figsize=(FIG_W_IN, FIG_H_IN),
    cmap="viridis", center=0.0,
    linewidths=0.3, linecolor="#f2f2f2",
    row_cluster=True, col_cluster=False,  # cluster species only; metrics keep order
    cbar_kws={"label": "Z-score (within metric)"},
)

# Put true mean values as text annotations (smaller font, 2 decimals)
ax = cg.ax_heatmap
for i, sp in enumerate(Z.index):
    for j, m in enumerate(Z.columns):
        val = annot_vals.loc[sp, m]
        ax.text(j+0.5, i+0.5, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color="black")

# Replace y tick labels with italics (must map new y order from clustermap)
new_yticks = [italicize_taxon(txt.get_text()) for txt in ax.get_yticklabels()]
ax.set_yticklabels(new_yticks, rotation=0)

ax.set_xlabel("Metric", labelpad=4)
ax.set_ylabel("Species name", labelpad=4)
# cg.fig.suptitle("A  |  Species-wise caption scores (clustered by profile)", x=0.0, y=1.02, ha="left", fontsize=7)
cg.fig.tight_layout()
cg.fig.savefig(os.path.join(OUTDIR, "llama_A_clustered_heatmap.png"), bbox_inches="tight",transparent=True)
cg.fig.savefig(os.path.join(OUTDIR, "llama_A_clustered_heatmap.pdf"), transparent=True, bbox_inches="tight")


