#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import PowerNorm
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

# ========================= STYLE =========================
mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 1500,
    "font.family": "sans-serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

# ========================= INPUTS ========================
CSV_PATH = "/home/reshma/MORPHID/Plots/Fig_2/llama_output_filtered.csv" 
COL_TRUE = "actual_label"
COL_PRED = "predicted_label"

OUT_HM_PNG      = "/home/reshma/MORPHID/Plots/Fig_2/llama_confusion_heatmap_vibrant.png"
OUT_CHORD_PNG   = "/home/reshma/MORPHID/Plots/Fig_2/llama_confusion_chord_connect_only_vibrant.png"
OUT_ALLUVIAL    = "/home/reshma/MORPHID/Plots/Fig_2/llama_alluvial_vibrant.png"

# ========================= HELPERS =======================
def format_taxon_label(name: str) -> str:
    """
    Return a matplotlib-compatible label with italics for Genus + species.
    If the epithet is 'sp.' keep it upright.
    """
    parts = name.strip().split()
    if len(parts) == 2 and parts[1].startswith("sp"):
        return r"$\it{" + parts[0] + r"}$ " + parts[1]
    elif len(parts) >= 2:
        return r"$\it{" + parts[0] + r"}$ " + r"$\it{" + ' '.join(parts[1:]) + r"}$"
    else:
        return r"$\it{" + name + r"}$"

def vivid_palette(num):
    if num <= 20:
        return sns.color_palette("tab20", num)
    else:
        return sns.hls_palette(num, l=0.50, s=0.95)

def filter_topk_threshold(mat_norm, min_flow=0.0, top_k=None):
    edges = []
    for i in range(mat_norm.shape[0]):
        row = mat_norm[i].copy()
        idx = np.where(row >= min_flow)[0]
        if top_k is not None and len(idx) > 0:
            top_idx = np.argsort(row)[::-1][:top_k]
            idx = np.intersect1d(idx, top_idx)
        for j in idx:
            v = float(row[j])
            if v > 0:
                edges.append((i, j, v))
    return edges

def bezier_ribbon(x0, y0a, y0b, x1, y1a, y1b, curvature=0.5, color=(0,0,0), alpha=0.6, lw=0.5):
    cx0 = x0 + curvature * (x1 - x0)
    cx1 = x1 - curvature * (x1 - x0)
    verts = [
        (x0, y0b), (cx0, y0b), (cx1, y1b), (x1, y1b),
        (x1, y1a), (cx1, y1a), (cx0, y0a), (x0, y0a), (x0, y0b)
    ]
    codes = [
        Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY
    ]
    return PathPatch(Path(verts, codes), facecolor=color, edgecolor="white", lw=lw, alpha=alpha)

# ========================= LOAD ==========================
df = pd.read_csv(CSV_PATH)
y_true = df[COL_TRUE].astype(str)
y_pred = df[COL_PRED].astype(str)
labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
n = max(1, len(labels))

cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
cm_norm = np.nan_to_num(cm_norm)
cm_cnt  = confusion_matrix(y_true, y_pred, labels=labels)

cm_df_norm = pd.DataFrame(cm_norm, index=labels, columns=labels)
cm_df_cnt  = pd.DataFrame(cm_cnt,  index=labels, columns=labels)

# ========================= HEATMAP =======================
cmap_vibrant = plt.get_cmap("turbo")
norm = PowerNorm(gamma=0.55, vmin=1e-3, vmax=1.0)

fig, ax = plt.subplots(figsize=(max(6, n*0.4), max(6, n*0.4)))
sns.heatmap(cm_df_norm, ax=ax,
            cmap=cmap_vibrant, norm=norm, cbar=True,
            cbar_kws={"label": "Row-normalized score"},
            linewidths=0.8, linecolor="white", square=True)

ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
ax.set_xlabel("Predicted species", labelpad=4)
ax.set_ylabel("Reference species")

# format axis labels with italics
ax.set_xticklabels([format_taxon_label(t.get_text()) for t in ax.get_xticklabels()], rotation=90)
ax.set_yticklabels([format_taxon_label(t.get_text()) for t in ax.get_yticklabels()], rotation=0)

plt.savefig(OUT_HM_PNG, bbox_inches="tight", dpi=1500, transparent=True)
plt.close(fig)

# ========================= COLORS ========================
node_colors = vivid_palette(n)

# ========================= CHORD =========================
def plot_chord_ticks_labels_no_ring(mat_norm, labels, colors, out_png,
                                    min_flow=0.0, width_scale=12.0,
                                    tick_len=0.012, r_pad=0.025,
                                    label_fontsize=7, 
                                    #label_weight="bold",
                                    edge_alpha=0.7, curvature_pull=0.6):
    n = len(labels)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos = np.c_[np.cos(theta), np.sin(theta)]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal"); ax.axis("off")

    for k, (x, y) in enumerate(pos):
        ax.plot([(1.0 - tick_len)*x, (1.0 + tick_len)*x],
                [(1.0 - tick_len)*y, (1.0 + tick_len)*y],
                color=colors[k], lw=2)
        ang_deg = np.degrees(np.arctan2(y, x))
        lx, ly = (1.0 + tick_len + r_pad) * x, (1.0 + tick_len + r_pad) * y
        if -90 <= ang_deg <= 90:
            rot, ha = ang_deg, "left"
        else:
            rot, ha = ang_deg + 180, "right"
        ax.text(lx, ly, format_taxon_label(labels[k]),
                rotation=rot, rotation_mode="anchor",
                ha=ha, va="center",
                color=colors[k], fontsize=label_fontsize,
                #fontweight=label_weight, 
                clip_on=False)

    edges = filter_topk_threshold(mat_norm, min_flow=min_flow)
    for i, j, v in edges:
        p0, p1 = pos[i], pos[j]
        c0, c1 = p0 * curvature_pull, p1 * curvature_pull
        lw = max(0.2, width_scale * v)
        path = Path([p0, c0, c1, p1],
                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        ax.add_patch(PathPatch(path, facecolor="none",
                               edgecolor=colors[i], lw=lw,
                               alpha=edge_alpha, capstyle="round", joinstyle="round"))
    plt.savefig(out_png, bbox_inches="tight", dpi=1500, transparent=True); plt.close(fig)

# ========================= ALLUVIAL ======================
def plot_alluvial(cm_counts, labels, colors, out_png, min_flow_norm=0.02):
    n = len(labels); total = cm_counts.values.sum() or 1
    left_sizes  = cm_counts.sum(axis=1).values / total
    right_sizes = cm_counts.sum(axis=0).values / total
    left_starts  = np.r_[0, np.cumsum(left_sizes[:-1])]
    right_starts = np.r_[0, np.cumsum(right_sizes[:-1])]
    left_ptr, right_ptr = left_starts.copy(), right_starts.copy()

    fig, ax = plt.subplots(figsize=(12, max(6, 0.3*n+4)))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    xL, xR, bar_w = 0.08, 0.92, 0.015

    for i in range(n):
        ax.add_patch(Rectangle((xL-bar_w/2, left_starts[i]), bar_w, left_sizes[i],
                               facecolor=colors[i], edgecolor="white", lw=0.6))
        ax.text(xL-0.03, left_starts[i]+left_sizes[i]/2,
                format_taxon_label(labels[i]), ha="right", va="center",
                color=colors[i], fontsize=7, fontweight="bold")

    for j in range(n):
        ax.add_patch(Rectangle((xR-bar_w/2, right_starts[j]), bar_w, right_sizes[j],
                               facecolor=colors[j], edgecolor="white", lw=0.6))
        ax.text(xR+0.03, right_starts[j]+right_sizes[j]/2,
                format_taxon_label(labels[j]), ha="left", va="center",
                color=colors[j], fontsize=7, fontweight="bold")

    cm_norm_total = (cm_counts / total).values
    for i in range(n):
        for j, frac in enumerate(cm_norm_total[i]):
            if frac < min_flow_norm: continue
            y0a, y0b = left_ptr[i], left_ptr[i]+frac
            y1a, y1b = right_ptr[j], right_ptr[j]+frac
            left_ptr[i] += frac; right_ptr[j] += frac
            ax.add_patch(bezier_ribbon(xL+bar_w/2, y0a, y0b,
                                       xR-bar_w/2, y1a, y1b,
                                       curvature=0.5, color=colors[i],
                                       alpha=0.7, lw=0.6))
    plt.savefig(out_png, bbox_inches="tight", dpi=1500, transparent=True); plt.close(fig)

# ========================= RUN ===========================
if __name__ == "__main__":
    plot_chord_ticks_labels_no_ring(cm_df_norm.values, labels, node_colors, OUT_CHORD_PNG)
    plot_alluvial(cm_df_cnt, labels, node_colors, OUT_ALLUVIAL)
    print(f"Saved:\n  {OUT_HM_PNG}\n  {OUT_CHORD_PNG}\n  {OUT_ALLUVIAL}")
