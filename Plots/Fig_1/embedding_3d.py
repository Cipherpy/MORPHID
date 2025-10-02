#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------- paths ----------------
CSV_PATH = "/home/reshma/MORPHID/Plots/Fig_1/embedding_metrics_id_ood.csv"
OUT_PNG  = "/home/reshma/MORPHID/Plots/Fig_1/plots/Species6.png"
OUT_CSV  = "/home/reshma/MORPHID/Plots/Fig_1/plots/Species6.csv"

# --- User override (optional) ---
OOD_TARGET = "OOD_deepsea_Species6"
SHOW_POINTS = True  # show sample points

# ---------------- fonts (sans-serif, compact) ----------------

#plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans", "Helvetica"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 7

# ------------- helpers -------------
def detect_metric_cols(df):
    def pick(prefixes):
        for c in df.columns:
            lc = c.lower()
            if any(lc.startswith(p) for p in prefixes):
                return c
        return None
    pca_col  = pick(["pca", "pca_", "pca-res", "pcares", "pcaresidual"])
    knn_col  = pick(["knn", "knn_", "knnmean", "knn_mean"])
    maha_col = pick(["maha", "maha_", "mahal", "mahalan", "maha_distance", "mahalanobis"])
    if not pca_col or not knn_col or not maha_col:
        raise ValueError(
            f"Could not auto-detect metric columns. "
            f"Found pca={pca_col}, knn={knn_col}, maha={maha_col}."
        )
    return pca_col, knn_col, maha_col

def has_col(obj, names):
    cols = list(obj.columns) if hasattr(obj, "columns") else list(obj)
    for n in names:
        if n in cols:
            return n
    return None

def infer_label_from_path(path_str):
    try:
        p = Path(path_str)
        parent = p.parent.name
        if parent and parent.lower() != "ood":
            return parent
        name = p.stem
        tokens = re.split(r"[ _\-]+", name)
        for t in tokens:
            if t.lower() != "ood" and t.strip():
                return t
    except Exception:
        pass
    return None

def detect_is_ood(row):
    for col in ["is_ood", "ood", "OOD", "Is_OOD", "isOOD"]:
        if col in row and pd.notna(row[col]):
            val = row[col]
            if isinstance(val, (int, float)): return bool(val)
            if isinstance(val, str): return val.strip().lower() in ["1","true","yes","y","ood"]
            return bool(val)
    for col in ["file","filepath","path","filename","image","img_path"]:
        if col in row and isinstance(row[col], str) and "ood" in row[col].lower():
            return True
    return False

def extract_label(row):
    label_col = has_col(row.index, ["label","class","species","id_class","gt_label","true_label"])
    if label_col:
        val = row[label_col]
        if isinstance(val, str) and val.strip():
            return val.strip()
    for col in ["file","filepath","path","filename","image","img_path"]:
        if col in row and isinstance(row[col], str):
            lab = infer_label_from_path(row[col])
            if lab: return lab
    return None

def euclid(a, b):
    return float(np.linalg.norm(a - b))

def resolve_ood_target(ood_df, user_spec):
    if ood_df.empty: raise RuntimeError("No OOD rows detected.")
    labels_unique = sorted(map(str, ood_df["label"].unique()))
    if not user_spec:
        return ood_df["label"].value_counts().idxmax()
    exact = [l for l in labels_unique if l.lower() == user_spec.lower()]
    if exact: return exact[0]
    sub = [l for l in labels_unique if user_spec.lower() in l.lower()]
    if sub:
        counts = ood_df["label"].value_counts()
        return max(sub, key=lambda l: counts.get(l, 0))
    try:
        pat = re.compile(user_spec, re.IGNORECASE)
        reg = [l for l in labels_unique if pat.search(l)]
        if reg:
            counts = ood_df["label"].value_counts()
            return max(reg, key=lambda l: counts.get(l, 0))
    except re.error:
        pass
    sample = ", ".join(labels_unique[:30])
    raise RuntimeError(f"Could not find OOD label matching '{user_spec}'. "
                       f"Available examples (first 30): {sample}")

def small_positive(x, minv):
    x = float(x)
    return x if x > minv else minv

def estimate_radii(df_sub, cols, scale=1.25, min_radius=1e-2):
    if len(df_sub) >= 3:
        stds = df_sub[cols].std(ddof=1).values
        r = np.array([small_positive(s, min_radius) for s in stds]) * scale
    elif len(df_sub) == 2:
        dif = np.abs(df_sub[cols].iloc[0].values - df_sub[cols].iloc[1].values)
        r = np.array([small_positive(d, min_radius) for d in dif]) * 0.5 * scale
    else:
        r = np.array([min_radius, min_radius, min_radius])
    return r

def plot_ellipsoid(ax, center, radii, color, alpha=0.28):
    u = np.linspace(0, 2*np.pi, 72)
    v = np.linspace(0, np.pi, 36)
    x = radii[0] * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radii[1] * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    ax.plot_surface(
        x, y, z,
        rstride=1, cstride=1,
        color=color,
        edgecolor='none',
        linewidth=0.0,
        antialiased=True,
        shade=True,
        alpha=alpha
    )

# ------------- CLI args -------------
parser = argparse.ArgumentParser(description="Transparent 3D plot with ellipsoids and points (no lines/arrows).")
parser.add_argument("--ood-target", type=str, default=None,
                    help="Label (or substring/regex) of the OOD class to plot.")
args = parser.parse_args()

# ------------- load data -------------
df = pd.read_csv(CSV_PATH)
pca_col, knn_col, maha_col = detect_metric_cols(df)
metric_cols = [pca_col, knn_col, maha_col]

# Ensure some path-like column exists for label inference (if needed)
if not has_col(df, ["file","filepath","path","filename","image","img_path"]):
    df["filename"] = [f"row_{i}.jpg" for i in range(len(df))]

# Build is_ood and label columns if missing
if "is_ood" not in df.columns:
    df["is_ood"] = df.apply(detect_is_ood, axis=1)

if not has_col(df, ["label","class","species","id_class","gt_label","true_label"]):
    df["label"] = df.apply(extract_label, axis=1)
else:
    label_src = has_col(df, ["label","class","species","id_class","gt_label","true_label"])
    if label_src != "label":
        df["label"] = df[label_src]
df["label"] = df["label"].fillna("UNKNOWN").astype(str)

# ------------- choose OOD target -------------
ood_df = df[df["is_ood"] == True].copy()
if ood_df.empty:
    raise RuntimeError("No OOD rows detected. Ensure filenames or an 'is_ood' column indicate OOD samples.")
user_spec = args.ood_target or os.environ.get("OOD_TARGET") or OOD_TARGET
ood_target = resolve_ood_target(ood_df, user_spec)

# ------------- centroids -------------
ood_target_rows = ood_df[ood_df["label"] == ood_target].copy()
if ood_target_rows.empty:
    raise RuntimeError(f"Could not find OOD rows for target class '{ood_target}'.")
ood_centroid = ood_target_rows[metric_cols].mean().values

id_df = df[df["is_ood"] == False].copy()
id_classes = [c for c in id_df["label"].unique() if c not in [None,"","UNKNOWN"]]
centroids = {}
for cls in id_classes:
    sub = id_df[id_df["label"] == cls]
    if len(sub) >= 1:
        centroids[cls] = sub[metric_cols].mean().values

# Distances (not plotted)
dists = [(cls, euclid(ood_centroid, cen)) for cls, cen in centroids.items()]
dists.sort(key=lambda x: x[1])
nearest5 = [cls for cls, _ in dists[:5]]
if not nearest5:
    raise RuntimeError("Could not compute nearest ID classes (no ID centroids).")
nearest1_cls = nearest5[0]
nearest_other_list = [c for c in nearest5 if c != nearest1_cls]

# ------------- groups to plot -------------
df_id_nearest = id_df[id_df["label"] == nearest1_cls]
df_id_other   = id_df[id_df["label"].isin(nearest_other_list)]
plot_df = pd.concat([
    ood_target_rows.assign(group=lambda d: "OOD"),
    df_id_nearest.assign(group=lambda d: "ID_nearest"),
    df_id_other.assign(group=lambda d: "ID_other")
], ignore_index=True)

# Save plotted subset
plot_df_out = plot_df.copy()
plot_df_out.rename(columns={pca_col: "pca", knn_col: "knn", maha_col: "maha"}, inplace=True)
plot_df_out.to_csv(OUT_CSV, index=False)

# ------------- colors -------------
COL_OOD        = "#FF4DA6"   # magenta
COL_ID_NEAREST = "#4DA6FF"   # blue
COL_ID_OTHER   = "#BF80FF"   # violet

# ------------- ellipsoid params -------------
r_ood     = estimate_radii(ood_target_rows, metric_cols, scale=1.25, min_radius=1e-2)
r_nearest = estimate_radii(df_id_nearest, metric_cols, scale=1.25, min_radius=1e-2)
other_centroid = df_id_other[metric_cols].mean().values if not df_id_other.empty else df_id_nearest[metric_cols].mean().values
r_other   = estimate_radii(df_id_other if not df_id_other.empty else df_id_nearest,
                           metric_cols, scale=1.25, min_radius=1e-2)

# ------------- figure (TRANSPARENT) -------------
fig = plt.figure(figsize=(6.2, 5.2), dpi=300, facecolor="none")  # transparent fig
ax = fig.add_subplot(111, projection='3d', facecolor="none")     # transparent axes

# Transparent panes + subtle borders
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.set_facecolor((0,0,0,0))     # fully transparent
    pane.set_alpha(0)
    pane.set_edgecolor((0,0,0,0.25))  # light edges; set (0,0,0,0) to hide

# Light grid (will be visible over light backgrounds too)
ax.grid(True, linestyle=":", linewidth=0.5, color=(0,0,0,0.35))

# Axis labels/ticks
ax.tick_params(axis='both', which='major', labelsize=7, colors="black", length=3, width=0.7)
ax.set_xlabel("PCA residual", fontsize=7, labelpad=6, color="black")
ax.set_ylabel("kNN mean",     fontsize=7, labelpad=6, color="black")
ax.set_zlabel("")  # optional 2D label if you like:
ax.text2D(1.04, 0.5, "Mahalanobis distance", transform=ax.transAxes,
          rotation=90, va="center", ha="left", fontsize=7, color="black")

# View & aspect
ax.view_init(elev=18, azim=38)
ax.set_box_aspect((1, 1, 0.9))

# ------------- scatter points -------------
if SHOW_POINTS:
    for role, col, df_sub, size, alpha in [
        ("OOD", COL_OOD, ood_target_rows, 24, 0.95),
        ("ID_nearest", COL_ID_NEAREST, df_id_nearest, 16, 0.85),
        ("ID_other", COL_ID_OTHER, df_id_other, 14, 0.80)
    ]:
        if df_sub.empty: continue
        ax.scatter(df_sub[pca_col], df_sub[knn_col], df_sub[maha_col],
                   s=size, depthshade=False, facecolors=col,
                   edgecolors=(0,0,0,0.7), linewidths=0.3, alpha=alpha, label=role)

# ------------- ellipsoids -------------
def plot_ellipsoid(ax, center, radii, color, alpha=0.28):
    u = np.linspace(0, 2*np.pi, 72)
    v = np.linspace(0, np.pi, 36)
    x = radii[0] * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radii[1] * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    color=color, edgecolor='none', linewidth=0.0,
                    antialiased=True, shade=True, alpha=alpha)

plot_ellipsoid(ax, ood_centroid,   r_ood, COL_OOD, alpha=0.28)
plot_ellipsoid(ax, np.asarray(df_id_nearest[metric_cols].mean().values), r_nearest, COL_ID_NEAREST, alpha=0.28)
plot_ellipsoid(ax, other_centroid, r_other, COL_ID_OTHER, alpha=0.26)

# Helper to format species names (italicize genus + epithet, leave "sp." etc. roman)
def format_species(name: str) -> str:
    parts = name.split()
    if len(parts) >= 2 and parts[1].lower() not in ["sp.", "cf.", "aff."]:
        # Italicize first two words (binomial)
        return rf"$\it{{{parts[0]}\ {parts[1]}}}$" + (" " + " ".join(parts[2:]) if len(parts) > 2 else "")
    else:
        # Only genus italicized
        return rf"$\it{{{parts[0]}}}$" + (" " + " ".join(parts[1:]) if len(parts) > 1 else "")

# Legend (transparent)
legend_handles = [
    # Line2D([0],[0], marker='s', markersize=7, linestyle='None',
    #        markerfacecolor=COL_OOD, markeredgecolor=(0,0,0,0.7),
    #        label=r"OOD: $\it{Nemipterus}$ sp. (Shallow marine)"),

    Line2D([0],[0], marker='s', markersize=7, linestyle='None',
           markerfacecolor=COL_OOD, markeredgecolor=(0,0,0,0.7),
           label=r"OOD: species 6 (Deep sea)"),
    
    Line2D([0],[0], marker='s', markersize=7, linestyle='None',
           markerfacecolor=COL_ID_NEAREST, markeredgecolor=(0,0,0,0.7),
           label=f"ID (nearest): {format_species(nearest1_cls)}"),
    
    Line2D([0],[0], marker='s', markersize=7, linestyle='None',
           markerfacecolor=COL_ID_OTHER, markeredgecolor=(0,0,0,0.7),
           label="ID (proximal species, excluding nearest in k=5)")
]

leg = ax.legend(handles=legend_handles, loc='upper left',
                frameon=False, fontsize=7)

# Save with full transparency
plt.tight_layout(pad=0.5)
plt.subplots_adjust(right=0.90)
plt.savefig(OUT_PNG, dpi=1200, bbox_inches="tight",transparent=True)
print(f"Saved plot to: {OUT_PNG}")
print(f"Saved plotted subset CSV to: {OUT_CSV}")
