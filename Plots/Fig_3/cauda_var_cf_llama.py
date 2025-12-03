#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dot-matrix Confusion Plot (Row-normalized)
- Green = correct (diagonal), Red = wrong (off-diagonal)
- Bubble size ∝ row-normalized value (0–1)
- Transparent, high-resolution PNG output + CSV of normalized CM
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ========================= STYLE =========================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 9,        # slightly smaller base font
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.title_fontsize": 9,
    "legend.fontsize": 9,
})

# ========================= CONFIG =========================
CSV_IN   = "/home/reshma/MORPHID/Plots/Fig_3/output_filtered copy.csv"
COL_GT   = "Description"
COL_PRED = "generated_caption"

OUT_DIR        = "plots/"
OUT_NORM_CSV   = os.path.join(OUT_DIR, "cauda_cm_normalized_llama.csv")
OUT_PNG        = os.path.join(OUT_DIR, "cauda_dotmatrix_correct_vs_wrong_llama.png")

# Plot sizing & styles
# 🔹 Make dots smaller (so they fit tiny cells)
DOT_MIN   = 3       # min marker area (pt^2)
DOT_MAX   = 40      # max marker area (pt^2)

# 🔹 Much smaller per-cell scaling
FIG_DX    = 0.03    # width scale per column (very small)
FIG_DY    = 0.03    # height scale per row (very small)

LABEL_PAD = 4       # slightly smaller padding
GRID_ALPHA = 0.06
SKIP_BELOW = 1e-6   # do not draw bubbles for (near-)zero values

COLOR_CORRECT = "#1f9d55"  # green
COLOR_WRONG   = "#d64545"  # red
GRID_COLOR    = (0, 0, 0, GRID_ALPHA)

# ===================== HELPERS ============================
def extract_raw_ostium(text: str) -> str:
    """Extract exact raw text after 'Cauda:' up to the next period or end."""
    if pd.isna(text):
        return ""
    m = re.search(r"Cauda:\s*([^\.]*)", str(text), flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""

def clean_feature_label(s: str) -> str:
    """
    Clean cauda feature phrases for plotting:
    - remove 'tubular' (and its following comma/space)
    - remove trailing 'curved'
    - remove leading commas
    - trim to max 15 characters
    """
    if pd.isna(s):
        return ""
    s = str(s).strip()

    # remove 'tubular' (case-insensitive) and optional following comma/space
    s = re.sub(r'\b[Tt]ubular\b,?\s*', '', s)

    # remove trailing 'curved' (e.g., 'strongly curved' -> 'strongly')
    s = re.sub(r'\s*[Cc]urved\.?$', '', s)

    # remove leading commas and surrounding spaces
    s = s.lstrip(",;:- ").strip()

    # normalise spaces
    s = re.sub(r'\s+', ' ', s)

    # final strip
    s = s.strip()

    # limit character length to 15
    if len(s) > 15:
        s = s[:15]

    return s

def row_normalize(df_counts: pd.DataFrame) -> pd.DataFrame:
    """Row-normalize a count matrix to [0,1]."""
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = df_counts.div(df_counts.sum(axis=1).replace(0, np.nan), axis=0)
    return norm.fillna(0.0)

# ===================== LOAD & PREP ========================
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_IN)

# Parse GT and predicted raw phrases
df["cauda_gt_raw"]  = df[COL_GT].apply(extract_raw_ostium)
df["cauda_gen_raw"] = df[COL_PRED].apply(extract_raw_ostium)

filtered = df[
    (df["cauda_gt_raw"]  != "") &
    (df["cauda_gen_raw"] != "")
].copy()

# Clean labels (remove 'tubular', trailing 'curved', leading comma, and limit to 15 chars)
filtered["cauda_gt_label"]  = filtered["cauda_gt_raw"].apply(clean_feature_label)
filtered["cauda_gen_label"] = filtered["cauda_gen_raw"].apply(clean_feature_label)

# Drop rows that became empty after cleaning
filtered = filtered[
    (filtered["cauda_gt_label"]  != "") &
    (filtered["cauda_gen_label"] != "")
].copy()

# Crosstab (counts) on cleaned labels
cm_counts = pd.crosstab(
    filtered["cauda_gt_label"],
    filtered["cauda_gen_label"],
    dropna=False
)

# Order rows/cols by total counts for readability
row_order = cm_counts.sum(axis=1).sort_values(ascending=False).index.tolist()
col_order = cm_counts.sum(axis=0).sort_values(ascending=False).index.tolist()
cm_counts = cm_counts.loc[row_order, col_order]

# Row-normalized matrix
cm_norm = row_normalize(cm_counts)
cm_norm.to_csv(OUT_NORM_CSV, index=True)

rows = cm_norm.index.tolist()
cols = cm_norm.columns.tolist()
vals = cm_norm.values

# ===================== PLOT ===============================
# 🔹 Remove huge 10x8" minimum → allow tight figure
fig_w = max(9.0, FIG_DX * len(cols))   # tiny base figure, scaled by columns
fig_h = max(3.0, FIG_DY * len(rows))   # tiny base figure, scaled by rows

plt.figure(figsize=(fig_w, fig_h), dpi=1500)
ax = plt.gca()
ax.set_facecolor("white")

# faint grid
for x in range(len(cols) + 1):
    ax.axvline(x - 0.5, color=GRID_COLOR, lw=0.4, zorder=0)
for y in range(len(rows) + 1):
    ax.axhline(y - 0.5, color=GRID_COLOR, lw=0.4, zorder=0)

# Scatter each (non-zero) cell
ys, xs = np.indices(vals.shape)
xs = xs.ravel()
ys = ys.ravel()
v  = vals.ravel()

# sizes ∝ value
sizes = DOT_MIN + (DOT_MAX - DOT_MIN) * v

# colors by correct vs wrong
plot_x, plot_y, plot_sizes, plot_colors = [], [], [], []
for xi, yi, vi, si in zip(xs, ys, v, sizes):
    if vi <= SKIP_BELOW:
        continue
    rlab, clab = rows[yi], cols[xi]
    is_diag = (rlab == clab)
    plot_x.append(xi)
    plot_y.append(yi)
    plot_sizes.append(si)
    plot_colors.append(COLOR_CORRECT if is_diag else COLOR_WRONG)

ax.scatter(
    plot_x, plot_y,
    s=plot_sizes,
    c=plot_colors,
    marker='o',
    linewidths=0,
    alpha=0.9
)

# ticks & labels
ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(rows)))
ax.set_xticklabels(cols, rotation=45, ha="right")
ax.set_yticklabels(rows)

# invert y so first row is at top
ax.invert_yaxis()

# spines
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.6)
    spine.set_alpha(0.3)

# ===================== LEGEND 1 (Bubble size) =====================
legend_levels = [0.25, 0.50, 0.75, 1.00]
size_handles = [
    Line2D([], [], marker='o', linestyle='',
           markersize=np.sqrt(DOT_MIN + (DOT_MAX - DOT_MIN) * lv),
           color="black", label=f"{lv:.2f}")
    for lv in legend_levels
]

leg1 = ax.legend(
    handles=size_handles,
    title="Reference count",
    frameon=True,
    loc="upper right",
    bbox_to_anchor=(1.3, 0.95),
    labelspacing=0.25,
    handlelength=2.0,
    borderpad=0.6
)

frame1 = leg1.get_frame()
frame1.set_edgecolor("#444444")
frame1.set_facecolor("#f2f2f7")
frame1.set_boxstyle("round,pad=1,rounding_size=3")

ax.add_artist(leg1)

# ===================== LEGEND 2 (Correct/Wrong) =====================
color_handles = [
    Line2D([], [], marker='o', linestyle='', color=COLOR_CORRECT, markersize=8, label="Correct"),
    Line2D([], [], marker='o', linestyle='', color=COLOR_WRONG,   markersize=8, label="Wrong"),
]

leg2 = ax.legend(
    handles=color_handles,
    title="Prediction Type",
    frameon=True,
    loc="upper right",
    bbox_to_anchor=(1.3, 0.25),
    handlelength=2.0,
    labelspacing=0.25,
    borderpad=0.6
)

frame2 = leg2.get_frame()
frame2.set_edgecolor("black")
frame2.set_facecolor("white")
frame2.set_boxstyle("round,pad=0.9,rounding_size=3")

# ===================== FINAL LABELS & AXIS LIMITS =====================
ax.set_xlabel("Generated features", labelpad=LABEL_PAD)
ax.set_ylabel("Reference features", labelpad=LABEL_PAD)

# 🔹 Clamp axes so there is no extra empty row/column around the matrix
ax.set_xlim(-0.5, len(cols) - 0.5)
ax.set_ylim(len(rows) - 0.5, -0.5)

plt.tight_layout()

# 🔹 Use tight bbox but with a small padding
plt.savefig(OUT_PNG, dpi=1500, transparent=True, bbox_inches="tight", pad_inches=0.2)
plt.close()

print(f"Saved:\n- Normalized CM: {OUT_NORM_CSV}\n- Figure: {OUT_PNG}")
