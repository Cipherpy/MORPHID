#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ostium feature evaluation & confusion matrix (Gemma)

Preprocessing:
- Extract 'Ostium:' text from GT and generated captions
- Build crosstab and row-normalize

New:
- Map hallucinated generated feature "blunt" -> "HALLUCINATE"
- Compute classification metrics (accuracy, precision, recall, F1, classification report)
  using the raw ostium phrase pairs.
- Plot a pastel blue–green confusion matrix with per-class recall bars
  (row-normalized), saved to OUT_PNG.
- For plotting, use compact code labels for shapes:
    funnel-like -> FN-SHAPE___
    tubular     -> TB-SHAPE___
    discoidal   -> DS-SHAPE___
    HALLUCINATE -> HALLUCINATE
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)

# ========================= STYLE =========================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.title_fontsize": 10,
    "legend.fontsize": 10,
})

# ========================= CONFIG =========================
CSV_IN   = "/home/reshma/MORPHID/Plots/Fig_3/output_filtered copy.csv"
COL_GT   = "Description"
COL_PRED = "generated_caption"

OUT_DIR        = "plots/"
OUT_NORM_CSV   = os.path.join(OUT_DIR, "ostium_cm_normalized.csv")
OUT_PNG        = os.path.join(OUT_DIR, "ostium_dotmatrix_correct_vs_wrong_llama.png")

# Extra outputs for metrics
OUT_REPORT_TXT   = os.path.join(OUT_DIR, "ostium_classification_report_llama.txt")
OUT_REPORT_CSV   = os.path.join(OUT_DIR, "ostium_classification_report_llama.csv")
OUT_SUMMARY_JSON = os.path.join(OUT_DIR, "ostium_summary_metrics_llama.json")

# Plot styling (used for fig sizing)
FIG_DX    = 0.40
FIG_DY    = 0.35
LABEL_PAD = 6
GRID_ALPHA = 0.06
GRID_COLOR = (0, 0, 0, GRID_ALPHA)

# Code labels just for plotting
PLOT_LABEL_MAP = {
    "funnel-like": "FN___",  # funnel
    "tubular":     "TB___",
    "discoidal":   "DS___",
    "not visible": "NOTVISIBLE_",
}

# ===================== HELPERS ============================
def extract_raw_ostium(text: str) -> str:
    """Extract exact raw text after 'Ostium:' up to the next period or end."""
    if pd.isna(text):
        return ""
    m = re.search(r"Ostium:\s*([^\.]*)", str(text), flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""

def clean_generated_feature(x: str) -> str:
    """
    Convert hallucinated generated features to standard labels.
    - 'blunt' (any case) -> 'HALLUCINATE'
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() == "blunt":
        return "HALLUCINATE"
    return s

def row_normalize(df_counts: pd.DataFrame) -> pd.DataFrame:
    """Row-normalize a count matrix to [0,1]."""
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = df_counts.div(df_counts.sum(axis=1).replace(0, np.nan), axis=0)
    return norm.fillna(0.0)

def pastel_bluegreen_cmap():
    """Pastel blue–green colormap."""
    stops = [
        (0.00, "#ffffff"),
        (0.05, "#f2fdfa"),
        (0.10, "#dff8f3"),
        (0.20, "#baf0e4"),
        (0.35, "#8fe0d2"),
        (0.50, "#64cdc2"),
        (0.70, "#39b3ae"),
        (0.85, "#2593a3"),
        (1.00, "#1d678f"),
    ]
    cmap = LinearSegmentedColormap.from_list("pastel_bluegreen_rich", stops)
    cmap.set_bad("#cbd5e1")   # gray for masked zeros
    cmap.set_under("#ffffff")
    return cmap

def plot_ostium_confusion_with_recall(
    cm_norm: pd.DataFrame,
    filename: str,
    dpi: int = 1500
):
    """
    Plot confusion matrix given a row-normalized pandas DataFrame cm_norm:
    - Index = rows = reference features
    - Columns = predicted features (can include predicted-only like 'HALLUCINATE')
    - Values = [0,1], row-normalized

    Adds per-row recall bars on the right:
    - Recall for a row = diagonal value if the same label exists as a column,
      else 0.
    """
    rows = list(cm_norm.index)
    cols = list(cm_norm.columns)
    vals = cm_norm.values  # shape: (n_rows, n_cols)

    # Plot labels: map to codes where available
    rows_plot = [PLOT_LABEL_MAP.get(r, r) for r in rows]
    cols_plot = [PLOT_LABEL_MAP.get(c, c) for c in cols]

    n_rows, n_cols = vals.shape
    cmap = pastel_bluegreen_cmap()

    # Compute recall per row from diagonal where possible
    recalls = []
    for i, rlab in enumerate(rows):
        if rlab in cols:
            j = cols.index(rlab)
            recalls.append(vals[i, j])
        else:
            recalls.append(0.0)
    recalls = np.array(recalls)

    # For color scaling, ignore exact zeros
    zero_mask = (vals == 0.0)
    nonzero_vals = vals[~zero_mask]
    if nonzero_vals.size > 0:
        vmax = max(np.percentile(nonzero_vals, 98), 0.1)
    else:
        vmax = 1.0
    vmin = 0.001

    fig_w = max(8.0, 0.25 * n_cols + 4)
    fig_h = max(6.0, 0.25 * n_rows + 2)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[10, 1], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])

    # Mask zeros for nicer appearance
    cm_masked = np.ma.masked_array(vals, mask=zero_mask)

    im = ax.imshow(
        cm_masked * 100,
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin * 100,
        vmax=vmax * 100,
        extent=(-0.5, n_cols - 0.5, n_rows - 0.5, -0.5)
    )

    # Ticks & labels (x on top)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(cols_plot, rotation=0, fontsize=8)
    ax.set_yticklabels(rows_plot, fontsize=8)

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # ax.set_title("Generated features", fontsize=10, pad=6)
    # ax.set_ylabel("Reference features", fontsize=10)

    # Minor gridlines
    ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Annotate non-zero cells
    if nonzero_vals.size > 0:
        max_val = float(np.nanmax(nonzero_vals))
    else:
        max_val = 0.0
    thr = 0.6 * max_val

    for i in range(n_rows):
        for j in range(n_cols):
            if vals[i, j] > 0:
                color = "white" if vals[i, j] > thr else "black"
                ax.text(
                    j, i, f"{vals[i, j] * 100:.1f}%",
                    ha="center", va="center",
                    fontsize=7, color=color
                )

    # Right-side recall bars (one per row)
    ax_bar = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_bar.barh(
        np.arange(n_rows),
        [1.0] * n_rows,
        height=0.2,
        color="#e5e7eb",
        edgecolor="none"
    )
    ax_bar.barh(
        np.arange(n_rows),
        recalls,
        height=0.2,
        color="#1d678f",
        edgecolor="none"
    )

    for i, val in enumerate(recalls):
        ax_bar.text(
            0.02, i - 0.20,
            f"{val:.2f}",
            va="bottom", ha="left",
            fontsize=7,
            color="#1e293b"
        )

    ax_bar.set_xlim(0, 1.05)
    ax_bar.yaxis.set_visible(False)
    ax_bar.set_xticks([0, 0.5, 1.0])
    ax_bar.set_xticklabels(["0", "0.5", "1"], fontsize=7, color="#1e293b")
    ax_bar.set_xlabel("Recall", fontsize=8, color="#1e293b")
    for spine in ["top", "right", "bottom", "left"]:
        ax_bar.spines[spine].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.savefig(filename, bbox_inches="tight", dpi=dpi, transparent=True)
    plt.close(fig)

# ===================== LOAD & PREP ========================
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_IN)

# Parse GT and predicted raw phrases
df["ostium_gt_raw"]  = df[COL_GT].apply(extract_raw_ostium)
df["ostium_gen_raw"] = (
    df[COL_PRED]
    .apply(extract_raw_ostium)
    .apply(clean_generated_feature)  # "blunt" -> "HALLUCINATE"
)

filtered = df[
    (df["ostium_gt_raw"]  != "") &
    (df["ostium_gen_raw"] != "")
].copy()

# Crosstab (counts): rows = GT, cols = predicted
cm_counts = pd.crosstab(
    filtered["ostium_gt_raw"],
    filtered["ostium_gen_raw"],
    dropna=False
)

# Order rows/cols by total counts for readability
row_order = cm_counts.sum(axis=1).sort_values(ascending=False).index.tolist()
col_order = cm_counts.sum(axis=0).sort_values(ascending=False).index.tolist()
cm_counts = cm_counts.loc[row_order, col_order]

# Row-normalized matrix (what we plot)
cm_norm = row_normalize(cm_counts)
cm_norm.to_csv(OUT_NORM_CSV, index=True)

rows = cm_norm.index.tolist()
cols = cm_norm.columns.tolist()

# ===================== METRICS ============================
# Use raw ostium phrases for metrics
y_true = filtered["ostium_gt_raw"].values
y_pred = filtered["ostium_gen_raw"].values

# Labels limited to rows (true classes) for macro metrics
class_names = rows

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(
    y_true, y_pred, labels=class_names,
    average="macro", zero_division=0
)
rec  = recall_score(
    y_true, y_pred, labels=class_names,
    average="macro", zero_division=0
)
f1   = f1_score(
    y_true, y_pred, labels=class_names,
    average="macro", zero_division=0
)

print("\n=== Ostium Feature Metrics (Gemma) ===")
print(f"Top-1 Accuracy    : {acc:.4f}")
print(f"Precision (macro) : {prec:.4f}")
print(f"Recall  (macro)   : {rec:.4f}")
print(f"F1-score (macro)  : {f1:.4f}")

# Classification report (per true class)
report_txt = classification_report(
    y_true, y_pred,
    labels=class_names,
    target_names=class_names,
    zero_division=0
)
with open(OUT_REPORT_TXT, "w") as f:
    f.write(report_txt)

report_dict = classification_report(
    y_true, y_pred,
    labels=class_names,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)
pd.DataFrame(report_dict).transpose().to_csv(OUT_REPORT_CSV)

# Summary JSON
summary = {
    "num_classes": len(class_names),
    "classes": class_names,
    "metrics": {
        "accuracy_top1": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
    }
}
with open(OUT_SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2)

# ===================== PLOT ===============================
plot_ostium_confusion_with_recall(
    cm_norm=cm_norm,
    filename=OUT_PNG
)

print(
    f"Saved:\n"
    f"- Normalized CM: {OUT_NORM_CSV}\n"
    f"- Confusion matrix figure: {OUT_PNG}\n"
    f"- Classification report TXT: {OUT_REPORT_TXT}\n"
    f"- Classification report CSV: {OUT_REPORT_CSV}\n"
    f"- Summary metrics JSON: {OUT_SUMMARY_JSON}"
)
