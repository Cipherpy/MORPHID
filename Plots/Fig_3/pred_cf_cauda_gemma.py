#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cauda feature evaluation & confusion matrix (Gemma)

Preprocessing:
- Extract 'Cauda:' text from GT and generated captions
- Clean feature labels (remove 'tubular', trailing 'curved', leading commas, limit length)
- Build crosstab and row-normalize (saved as CSV)

New:
- Compute classification metrics (accuracy, precision, recall, F1, classification report)
  using the cleaned label pairs.
- Plot a pastel blue–green confusion matrix with per-class recall bars
  (row-normalized via sklearn.confusion_matrix), saved to OUT_PNG.
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
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.title_fontsize": 9,
    "legend.fontsize": 9,
})

# ========================= CONFIG =========================
CSV_IN   = "/home/reshma/MORPHID/Plots/Fig_3/paired_captions_minimall_modi_gemma.csv"
COL_GT   = "Description"
COL_PRED = "generated_caption"

OUT_DIR        = "plots/"
OUT_NORM_CSV   = os.path.join(OUT_DIR, "cauda_cm_normalized_gemma.csv")
OUT_PNG        = os.path.join(OUT_DIR, "cauda_confusion_with_recall_gemma.png")

OUT_REPORT_TXT   = os.path.join(OUT_DIR, "cauda_classification_report_gemma.txt")
OUT_REPORT_CSV   = os.path.join(OUT_DIR, "cauda_classification_report_gemma.csv")
OUT_SUMMARY_JSON = os.path.join(OUT_DIR, "cauda_summary_metrics_gemma.json")

# ===================== HELPERS ============================
def extract_raw_cauda(text: str) -> str:
    """Extract exact raw text after 'Cauda:' up to the next period or end."""
    if pd.isna(text):
        return ""
    m = re.search(r"Cauda:\s*([^\.]*)", str(text), flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""


def clean_feature_label(s: str) -> str:
    """
    Clean cauda feature phrases for modeling & plotting:
    - remove 'tubular' (and its following comma/space)
    - remove trailing 'curved'
    - remove leading commas / punctuation
    - normalize spaces
    - trim to max 15 characters
    """
    if pd.isna(s):
        return ""
    s = str(s).strip()

    # remove 'tubular' (case-insensitive) and optional following comma/space
    s = re.sub(r'\b[Tt]ubular\b,?\s*', '', s)

    # remove trailing 'curved' (e.g. 'strongly curved' -> 'strongly')
    s = re.sub(r'\s*[Cc]urved\.?$', '', s)

    # remove leading commas / punctuation
    s = s.lstrip(",;:- ").strip()

    # normalize spaces
    s = re.sub(r'\s+', ' ', s).strip()

    # limit character length
    if len(s) > 15:
        s = s[:15]

    return s


def row_normalize(df_counts: pd.DataFrame) -> pd.DataFrame:
    """Row-normalize a count matrix to [0,1]."""
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = df_counts.div(df_counts.sum(axis=1).replace(0, np.nan), axis=0)
    return norm.fillna(0.0)


def pastel_bluegreen_cmap():
    """Pastel blue–green colormap (same style as ostium / sulcus)."""
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


def plot_cauda_confusion_with_recall(
    y_true,
    y_pred,
    class_names,
    filename,
    dpi=1500
):
    """
    Pastel blue–green confusion matrix with per-class recall bars.
    - Rows = reference (true) cauda features
    - Cols = generated (predicted) cauda features
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

    # Scale figure with number of classes
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
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.set_title("Generated cauda features", fontsize=9, pad=6)
    ax.set_ylabel("Reference cauda features", fontsize=9)

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
                    fontsize=6.5, color=color
                )

    # Right-side recall bars
    ax_bar = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_bar.barh(
        np.arange(n_classes),
        [1.0] * n_classes,
        height=0.2,
        color="#e5e7eb",
        edgecolor="none"
    )
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
            fontsize=6.5,
            color="#1e293b"
        )

    ax_bar.set_xlim(0, 1.05)
    ax_bar.yaxis.set_visible(False)
    ax_bar.set_xticks([0, 0.5, 1.0])
    ax_bar.set_xticklabels(["0", "0.5", "1"], fontsize=7, color="#1e293b")
    ax_bar.set_xlabel("Recall", fontsize=8, color="#1e293b")
    for spine in ["top", "right", "bottom", "left"]:
        spine_obj = ax_bar.spines[spine]
        spine_obj.set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.savefig(filename, bbox_inches="tight", dpi=dpi, transparent=True)
    plt.close(fig)


# ===================== LOAD & PREP ========================
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_IN)

# Parse GT and predicted raw phrases
df["cauda_gt_raw"]  = df[COL_GT].apply(extract_raw_cauda)
df["cauda_gen_raw"] = df[COL_PRED].apply(extract_raw_cauda)

filtered = df[
    (df["cauda_gt_raw"]  != "") &
    (df["cauda_gen_raw"] != "")
].copy()

# Clean labels (remove 'tubular', trailing 'curved', leading comma, etc.)
filtered["cauda_gt_label"]  = filtered["cauda_gt_raw"].apply(clean_feature_label)
filtered["cauda_gen_label"] = filtered["cauda_gen_raw"].apply(clean_feature_label)

# Drop rows that became empty after cleaning
filtered = filtered[
    (filtered["cauda_gt_label"]  != "") &
    (filtered["cauda_gen_label"] != "")
].copy()

# Crosstab (counts) on cleaned labels (for CSV)
cm_counts = pd.crosstab(
    filtered["cauda_gt_label"],
    filtered["cauda_gen_label"],
    dropna=False
)

# Order rows/cols by total counts for readability
row_order = cm_counts.sum(axis=1).sort_values(ascending=False).index.tolist()
col_order = cm_counts.sum(axis=0).sort_values(ascending=False).index.tolist()
cm_counts = cm_counts.loc[row_order, col_order]

# Row-normalized crosstab (for CSV output)
cm_norm = row_normalize(cm_counts)
cm_norm.to_csv(OUT_NORM_CSV, index=True)

# ===================== METRICS ======================
y_true = filtered["cauda_gt_label"].values
y_pred = filtered["cauda_gen_label"].values

# Class names = sorted union of labels present in true or predicted
class_names = sorted(list(set(y_true) | set(y_pred)))

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, labels=class_names,
                       average="macro", zero_division=0)
rec  = recall_score(y_true, y_pred, labels=class_names,
                    average="macro", zero_division=0)
f1   = f1_score(y_true, y_pred, labels=class_names,
                average="macro", zero_division=0)

print("\n=== Cauda Feature Metrics (Gemma) ===")
print(f"Top-1 Accuracy    : {acc:.4f}")
print(f"Precision (macro) : {prec:.4f}")
print(f"Recall (macro)    : {rec:.4f}")
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

# JSON summary
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

# ===================== PLOT =======================
plot_cauda_confusion_with_recall(
    y_true=y_true,
    y_pred=y_pred,
    class_names=class_names,
    filename=OUT_PNG
)

print(
    f"Saved:\n"
    f"- Normalized CM (crosstab, row-normalized): {OUT_NORM_CSV}\n"
    f"- Confusion matrix figure: {OUT_PNG}\n"
    f"- Classification report TXT: {OUT_REPORT_TXT}\n"
    f"- Classification report CSV: {OUT_REPORT_CSV}\n"
    f"- Summary metrics JSON: {OUT_SUMMARY_JSON}"
)
