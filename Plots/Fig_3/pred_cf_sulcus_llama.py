#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sulcus acusticus feature evaluation & confusion matrix (LLaMA)

Preprocessing:
- Same as original script:
  * Extract 'Sulcus acusticus:' text
  * Map to fixed short labels via LABEL_MAP
  * Build crosstab and row-normalize

New:
- Compute classification metrics (accuracy, precision, recall, F1, classification report)
  using the short label pairs.
- Plot a pastel blue–green confusion matrix with per-class recall bars
  (row-normalized), saved to OUT_PNG.
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
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

# ========================= STYLE =========================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 10,        # base font size (between 5–7 pt)
    "axes.labelsize": 10,   # axis labels
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.title_fontsize": 10,
    "legend.fontsize": 10,
})

# ========================= CONFIG =========================
CSV_IN   = "/home/reshma/MORPHID/Plots/Fig_3/output_filtered copy.csv"
COL_GT   = "Description"
COL_PRED = "generated_caption"

OUT_DIR        = "llama/plots"
OUT_NORM_CSV   = os.path.join(OUT_DIR, "sulcus_acusticus_cm_normalized_llama.csv")
OUT_PNG        = os.path.join(OUT_DIR, "sulcus_acusticus_dotmatrix_correct_vs_wrong_llama.png")

# Extra outputs for metrics
OUT_REPORT_TXT   = os.path.join(OUT_DIR, "sulcus_acusticus_classification_report_llama.txt")
OUT_REPORT_CSV   = os.path.join(OUT_DIR, "sulcus_acusticus_classification_report_llama.csv")
OUT_SUMMARY_JSON = os.path.join(OUT_DIR, "sulcus_acusticus_summary_metrics_llama.json")

# Plot sizing & styles (kept for reference)
DOT_MIN   = 8
DOT_MAX   = 180
FIG_DX    = 0.40
FIG_DY    = 0.35
LABEL_PAD = 6
GRID_ALPHA = 0.06
SKIP_BELOW = 1e-6

COLOR_CORRECT = "#1f9d55"  # green (kept)
COLOR_WRONG   = "#d64545"  # red (kept)
GRID_COLOR    = (0, 0, 0, GRID_ALPHA)

# ===================== HELPERS ============================
def extract_raw_sulcus_acusticus(text: str) -> str:
    """Extract exact raw text after 'Sulcus acusticus:' up to the next period or end."""
    if pd.isna(text):
        return ""
    m = re.search(r"Sulcus\s+acusticus:\s*([^\.]*)", str(text), flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""

# mapping from full phrase -> short fixed-length label (11 chars)
LABEL_MAP = {
    "heterosulcoid, ostial, median":        "HS-OS-MED__",
    "heterosulcoid, ostial, inframedian":   "HS-OS-INF__",
    "heterosulcoid, ostial, supramedian":   "HS-OS-SUP__",
    "pseudo-archaesulcoid, mesial":         "PA-MESIAL__",
    "pseudo-archaesulcoid, ostial, median": "PA-OS-MED__",
    "homosulcoid, para-ostial, median":     "HO-PO-MED__",
    "homosulcoid, mesial, median":          "HO-MESIAL-MD",
    "heterosulcoid, ostio-caudal, median":  "HS-OC-MED__",
    "Not visible":                          "NOTVISIBLE_",
}

def to_short_label(raw: str) -> str:
    """
    Convert raw sulcus description to short code:
    - use LABEL_MAP if available
    - otherwise truncate raw text to max 11 characters
    """
    if pd.isna(raw):
        return ""
    raw = str(raw).strip()
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    # fallback: remove commas, collapse spaces, then truncate
    tmp = re.sub(r"[,\s]+", " ", raw)
    tmp = tmp.strip()
    if len(tmp) > 11:
        tmp = tmp[:11]
    return tmp

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

def plot_sulcus_confusion_with_recall(
    y_true,
    y_pred,
    class_names,
    filename,
    dpi=1500
):
    """
    Pastel blue–green confusion matrix with per-class recall bars.
    - Rows = reference (true) sulcus features (short labels)
    - Cols = generated (predicted) sulcus features (short labels)
    - Values = row-normalized (%)
    - Right side: horizontal bars showing recall (diagonal / row total)
    """

    # Confusion matrix normalized by true label (row-normalized)
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=class_names,
        normalize="true"
    )

    # Per-class recall = diagonal of row-normalized CM
    recalls = np.diag(cm)

    zero_mask = (cm == 0.0)
    cm_masked = np.ma.masked_array(cm, mask=zero_mask)

    cmap = pastel_bluegreen_cmap()

    n_classes = len(class_names)
    fig_w = max(8.0, 0.25 * n_classes + 4)
    fig_h = max(6.0, 0.25 * n_classes + 2)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[10, 1], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])

    # choose vmax based on non-zero cells
    if np.any(~zero_mask):
        vmax = np.percentile(cm[~zero_mask], 98)
        vmax = max(vmax, 0.1)
    else:
        vmax = 1.0
    vmin = 0.001

    # Show as percent
    im = ax.imshow(
        cm_masked * 100,
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin * 100,
        vmax=vmax * 100,
        extent=(-0.5, n_classes - 0.5, n_classes - 0.5, -0.5)
    )

    # Ticks & labels (x on top)
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.set_title("Generated features", fontsize=10, pad=6)
    ax.set_ylabel("Reference features", fontsize=10)

    # Minor gridlines
    ax.set_xticks(np.arange(-.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_classes, 1), minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Annotate non-zero cells
    if np.any(~zero_mask):
        max_val = np.nanmax(cm)
    else:
        max_val = 0.0
    thr = 0.6 * max_val

    for i in range(n_classes):
        for j in range(n_classes):
            if cm[i, j] > 0:
                color = "white" if cm[i, j] > thr else "black"
                ax.text(
                    j, i, f"{cm[i, j] * 100:.1f}%",
                    ha="center", va="center",
                    fontsize=7, color=color
                )

    # Colorbar placed above the matrix
    fig.canvas.draw()
    # bbox = ax.get_position()
    # cb_width  = 0.20
    # cb_height = 0.015
    # cb_left   = bbox.x0
    # cb_bottom = bbox.y1 + cb_height * 1.8
    # cb_ax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])
    # cbar = plt.colorbar(im, cax=cb_ax, orientation="horizontal")
    # cbar.ax.tick_params(labelsize=7, pad=1)
    # cbar.set_label("Percent (row-normalized)", fontsize=7, labelpad=2)

    # Right-side recall bars
    ax_bar = fig.add_subplot(gs[0, 1], sharey=ax)
    # Background bar (1.0)
    ax_bar.barh(
        np.arange(n_classes),
        [1.0] * n_classes,
        height=0.2,
        color="#e5e7eb",
        edgecolor="none"
    )
    # Actual recall
    ax_bar.barh(
        np.arange(n_classes),
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
df["sulcus_acusticus_gt_raw"]  = df[COL_GT].apply(extract_raw_sulcus_acusticus)
df["sulcus_acusticus_gen_raw"] = df[COL_PRED].apply(extract_raw_sulcus_acusticus)

filtered = df[
    (df["sulcus_acusticus_gt_raw"]  != "") &
    (df["sulcus_acusticus_gen_raw"] != "")
].copy()

# Apply short labels
filtered["sulcus_acusticus_gt_lbl"]  = filtered["sulcus_acusticus_gt_raw"].apply(to_short_label)
filtered["sulcus_acusticus_gen_lbl"] = filtered["sulcus_acusticus_gen_raw"].apply(to_short_label)

# Drop rows that became empty after mapping (just in case)
filtered = filtered[
    (filtered["sulcus_acusticus_gt_lbl"]  != "") &
    (filtered["sulcus_acusticus_gen_lbl"] != "")
].copy()

# Crosstab (counts) using the short labels
cm_counts = pd.crosstab(
    filtered["sulcus_acusticus_gt_lbl"],
    filtered["sulcus_acusticus_gen_lbl"],
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

# ===================== METRICS (NEW) ======================
# Use the short-label pairs to compute metrics
y_true = filtered["sulcus_acusticus_gt_lbl"].values
y_pred = filtered["sulcus_acusticus_gen_lbl"].values

# Class names = sorted union of all labels seen in true or predicted
class_names = sorted(list(set(y_true) | set(y_pred)))

# Overall metrics (macro)
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, labels=class_names,
                       average="macro", zero_division=0)
rec  = recall_score(y_true, y_pred, labels=class_names,
                    average="macro", zero_division=0)
f1   = f1_score(y_true, y_pred, labels=class_names,
                average="macro", zero_division=0)

print("\n=== Sulcus acusticus Feature Metrics (LLaMA) ===")
print(f"Top-1 Accuracy    : {acc:.4f}")
print(f"Precision (macro) : {prec:.4f}")
print(f"Recall  (macro)   : {rec:.4f}")
print(f"F1-score (macro)  : {f1:.4f}")

# Classification report
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

# ===================== PLOT (NEW STYLE) ===================
plot_sulcus_confusion_with_recall(
    y_true=y_true,
    y_pred=y_pred,
    class_names=class_names,
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
