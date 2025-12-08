#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sulcus acusticus feature evaluation & confusion matrix (Gemma)

Preprocessing:
- Extract 'Sulcus acusticus:' text
- Map to fixed short labels via LABEL_MAP
- Build crosstab and row-normalize

New:
- Ignore 'HALLUCINATE' in reference (ground-truth) features:
  it can appear only in generated features (prediction columns, not rows).
- Compute classification metrics (accuracy, precision, recall, F1, classification report)
  using the short label pairs (excluding HALLUCINATE from true labels).
- Plot a pastel blue–green confusion matrix with per-class recall bars
  (row-normalized), saved to OUT_PNG, using the asymmetric crosstab
  (rows = true labels, cols = predicted labels, incl. HALLUCINATE).
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
    "font.size": 10,        # base font size
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

OUT_DIR        = "plots"
OUT_NORM_CSV   = os.path.join(OUT_DIR, "sulcus_acusticus_cm_normalized_llama.csv")
OUT_PNG        = os.path.join(OUT_DIR, "sulcus_acusticus_dotmatrix_correct_vs_wrong_llama.png")
    
# Extra outputs for metrics
OUT_REPORT_TXT   = os.path.join(OUT_DIR, "sulcus_acusticus_classification_report_llama.txt")
OUT_REPORT_CSV   = os.path.join(OUT_DIR, "sulcus_acusticus_classification_report_llama.csv")
OUT_SUMMARY_JSON = os.path.join(OUT_DIR, "sulcus_acusticus_summary_metrics_llama.json")

GRID_ALPHA = 0.06
GRID_COLOR = (0, 0, 0, GRID_ALPHA)

# ===================== HELPERS ============================
def extract_raw_sulcus_acusticus(text: str) -> str:
    """Extract exact raw text after 'Sulcus acusticus:' up to the next period or end."""
    if pd.isna(text):
        return ""
    m = re.search(r"Sulcus\s+acusticus:\s*([^\.]*)", str(text), flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""

# 11-char labels for all features (including hallucinated one)
LABEL_MAP = {
    "heterosulcoid, ostial, median":        "HS-OS-MED__",
    "heterosulcoid, ostial, inframedian":   "HS-OS-INF__",
    "heterosulcoid, ostial, supramedian":   "HS-OS-SUP__",
    "pseudo-archaesulcoid, mesial":         "PA-MESIAL__",
    "pseudo-archaesulcoid, ostial, median": "PA-OS-MED__",
    "homosulcoid, para-ostial, median":     "HO-PO-MED__",
    "heterosulcoid, ostio-caudal, median":  "HS-OC-MED__",
    "heterosulcoid, ostial, median to supramedian": "HS-OS-M2S__",
    "homosulcoid, mesial, median":          "HO-MESIAL-MD",  # short code
    "Not visible":                          "NOTVISIBLE_",
    # hallucinated / wrong feature:
    "crenate ventral margins":              "HALLUCINATE",
}

def to_short_label(raw: str) -> str:
    """
    Convert raw sulcus description to short code:
    - use LABEL_MAP if available
    - otherwise truncate cleaned raw text to max 11 characters
    """
    if pd.isna(raw):
        return ""
    raw = str(raw).strip()
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    # fallback: remove commas, collapse spaces, truncate
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
    """Pastel blue–green colormap (same style as other features)."""
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

def plot_sulcus_confusion_with_recall_cm(
    cm_norm: pd.DataFrame,
    filename: str,
    dpi: int = 1500
):
    """
    Plot confusion matrix using an already row-normalized crosstab (cm_norm):
    - cm_norm.index   = reference (true) labels (NO HALLUCINATE)
    - cm_norm.columns = predicted labels (can include HALLUCINATE)
    - Values          = [0,1], row-normalized

    Right side: horizontal bars showing per-row recall (diagonal value where
    the same label exists as a predicted column; 0 otherwise).
    """
    rows = list(cm_norm.index)
    cols = list(cm_norm.columns)
    vals = cm_norm.values

    n_rows, n_cols = vals.shape
    cmap = pastel_bluegreen_cmap()

    # Per-row recall from diagonal where possible
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

    # --- FIGURE SIZE: still adaptive but consistent ---
    fig_w = max(8.0, 0.25 * n_cols + 4)
    fig_h = max(6.0, 0.25 * n_rows + 2)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[10, 1], wspace=0.02)

    # ================== MAIN CONFUSION MATRIX ==================
    ax = fig.add_subplot(gs[0, 0])

    cm_masked = np.ma.masked_array(vals, mask=zero_mask)

    im = ax.imshow(
        cm_masked * 100,
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin * 100,
        vmax=vmax * 100,
        extent=(-0.5, n_cols - 0.5, n_rows - 0.5, -0.5),
        aspect="auto"
    )

    # Ticks & labels (x on top)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticklabels(rows, fontsize=8)

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(
        axis='x',
        which='major',
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        length=0
    )
    ax.tick_params(
        axis='y',
        which='major',
        left=True,
        right=False,
        length=0
    )
    # ax.set_title("Generated features", fontsize=10, pad=6)
    # ax.set_ylabel("Reference features", fontsize=10)

    # Explicitly lock y-limits so each row is exactly 1 unit high
    ax.set_ylim(n_rows - 0.5, -0.5)

    # Minor gridlines (cell boundaries only)
    ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
        top=False,
        right=False,
        length=0
    )
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

    # ================== RIGHT-SIDE RECALL BARS ==================
    # NOTE: no sharey; we manually match the y-limits so it works
    # for any number of rows (5, 9, etc.)
    ax_bar = fig.add_subplot(gs[0, 1])

    # y-positions that match the centers of the CM rows
    y_pos = np.arange(n_rows)

    # Background bar (1.0)
    ax_bar.barh(
        y_pos,
        [1.0] * n_rows,
        height=0.2,               # ~cell height; works for any n_rows
        color="#e5e7eb",
        edgecolor="none"
    )
    # Foreground recall bar
    ax_bar.barh(
        y_pos,
        recalls,
        height=0.2,
        color="#1d678f",
        edgecolor="none"
    )

    # Numeric recall labels (centered on each row)
    for y, val in zip(y_pos, recalls):
        ax_bar.text(
            0.02, y-0.20,
            f"{val:.2f}",
            va="center", ha="left",
            fontsize=7,
            color="#1e293b"
        )

    ax_bar.set_xlim(0, 1.05)

    # Match y-limits to main axis → guaranteed alignment
    ax_bar.set_ylim(n_rows - 0.5, -0.5)

    ax_bar.yaxis.set_visible(False)
    ax_bar.set_xticks([0, 0.5, 1.0])
    ax_bar.set_xticklabels(["0", "0.5", "1"], fontsize=7, color="#1e293b")
    ax_bar.set_xlabel("Recall", fontsize=8, color="#1e293b")
    for spine in ["top", "right", "bottom", "left"]:
        ax_bar.spines[spine].set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.savefig(filename, bbox_inches="tight", dpi=dpi)
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

# Drop rows that became empty
filtered = filtered[
    (filtered["sulcus_acusticus_gt_lbl"]  != "") &
    (filtered["sulcus_acusticus_gen_lbl"] != "")
].copy()

# ===== NEW: Remove HALLUCINATE from reference labels (GT only) =====
IGNORE_REF = {"HALLUCINATE"}
before = len(filtered)
filtered = filtered[~filtered["sulcus_acusticus_gt_lbl"].isin(IGNORE_REF)].copy()
after = len(filtered)
print(f"Removed {before - after} rows with hallucinated reference labels.")

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

# Row-normalized matrix for plotting
cm_norm = row_normalize(cm_counts)
cm_norm.to_csv(OUT_NORM_CSV, index=True)

rows = cm_norm.index.tolist()
cols = cm_norm.columns.tolist()
vals = cm_norm.values

# ===================== METRICS ============================
# Use the short-label pairs to compute metrics (excluding HALLUCINATE in GT)
y_true = filtered["sulcus_acusticus_gt_lbl"].values
y_pred = filtered["sulcus_acusticus_gen_lbl"].values

# Class names = unique true labels only (no HALLUCINATE)
class_names = sorted(list(pd.unique(y_true)))

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

print("\n=== Sulcus acusticus Feature Metrics (Gemma) ===")
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

# ===================== PLOT (USING CM NORM) ===============
plot_sulcus_confusion_with_recall_cm(
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
