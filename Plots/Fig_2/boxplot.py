#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ------------------ PATHS ------------------
CSV_PATH  = "/home/reshma/MORPHID/Plots/Fig_2/caption_scores_llama.csv"
SAVE_PATH = "llama_boxplot.png"

# ------------------ STYLE (serif + size 7) ------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 6,
    "axes.titlesize": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.9,
    "figure.dpi": 150,
    "savefig.dpi": 1200,
})

# ------------------ LOAD DATA ------------------
df = pd.read_csv(CSV_PATH)
labcol = "actual_label"
score_cols = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L"]

df[labcol] = df[labcol].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
df[score_cols] = df[score_cols].apply(pd.to_numeric, errors="coerce")
df["avg_score"] = df[score_cols].mean(axis=1)

# ------------------ FILTER & ORDER ------------------
med = (df.groupby(labcol)["avg_score"].median().sort_values(ascending=False))
species_order_all = med[med > 0.50].index.tolist()
if len(species_order_all) == 0:
    raise ValueError("No species pass the median > 0.50 filter.")
df = df[df[labcol].isin(species_order_all)].copy()

# ------------------ DEFINE 4 PERFORMANCE BANDS ------------------
med_vals = med.loc[species_order_all].values
q75, q50, q25 = np.quantile(med_vals, [0.75, 0.50, 0.25])

def band_for(median_val):
    if median_val >= q75: return "High"
    if median_val >= q50: return "Mid-High"
    if median_val >= q25: return "Mid-Low"
    return "Low"

bands_in_order = ["High", "Mid-High", "Mid-Low", "Low"]
band_map = {sp: band_for(med.loc[sp]) for sp in species_order_all}

species_sorted = sorted(species_order_all, key=lambda s: med.loc[s], reverse=True)
band_blocks = {b: [s for s in species_sorted if band_map[s] == b] for b in bands_in_order}
species_order = [s for b in bands_in_order for s in band_blocks[b]]
n = len(species_order)

plot_df = df[[labcol, "avg_score"]].rename(columns={labcol: "species", "avg_score": "score"})
plot_df["species"] = pd.Categorical(plot_df["species"], categories=species_order, ordered=True)

# ------------------ COLOR SCHEME ------------------
# same color per section (no gradient)
band_colors = {
    "High":     "#84C784",   # green pastel
    "Mid-High": "#8CBCEB",   # blue pastel
    "Mid-Low":  "#E9B77B",   # orange pastel
    "Low":      "#E9A1A1"    # red pastel
}
palette_map = {s: band_colors[band_map[s]] for s in species_order}
palette_list = [palette_map[s] for s in species_order]

# section background tints (lighter)
section_bg = {
    "High":     (0.89, 0.96, 0.89),
    "Mid-High": (0.89, 0.94, 0.97),
    "Mid-Low":  (0.98, 0.94, 0.88),
    "Low":      (0.98, 0.91, 0.91)
}
SECTION_ALPHA = 0.26

# ------------------ STATS: mean ± 95% CI ------------------
def mean_ci95(a, reps=3000, seed=42):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    if a.size == 0: return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = a.mean()
    boots = rng.choice(a, (reps, a.size), replace=True).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return m, lo, hi

# ------------------ SPECIES LABEL FORMATTER (italic) ------------------
def format_species_label(text: str) -> str:
    t = text.strip()
    if not t:
        return ""
    parts = t.split(" ", 1)
    if len(parts) == 1:
        return rf"$\it{{{parts[0]}}}$"
    genus, rest = parts[0], parts[1].strip()
    if rest.lower().startswith("sp"):
        return rf"$\it{{{genus}}}$ {rest}"
    rest_parts = rest.split(" ", 1)
    if len(rest_parts) == 1:
        return rf"$\it{{{genus}}}$ " + rf"$\it{{{rest_parts[0]}}}$"
    else:
        first_epithet, tail = rest_parts[0], rest_parts[1]
        return rf"$\it{{{genus}}}$ " + rf"$\it{{{first_epithet}}}$ " + tail

formatted_xticks = [format_species_label(s) for s in species_order]

# ------------------ PLOT ------------------
fig_w = max(6, 0.34 * n)
fig, ax = plt.subplots(figsize=(fig_w, 4.9))

# background panels
current = 0
for b in bands_in_order:
    k = len(band_blocks[b])
    if k == 0:
        continue
    start, end = current, current + k - 1
    ax.axvspan(start - 0.5, end + 0.5, facecolor=section_bg[b], alpha=SECTION_ALPHA, zorder=0)
    current += k

# violins
sns.violinplot(
    data=plot_df, x="species", y="score",
    order=species_order, palette=palette_list,
    inner=None, cut=0, linewidth=0, saturation=1.0, ax=ax, zorder=3
)

# mean ± 95% CI
xpos = np.arange(n)
means, los, his = [], [], []
for s in species_order:
    vals = plot_df.loc[plot_df["species"] == s, "score"].values
    m, lo, hi = mean_ci95(vals)
    means.append(m); los.append(lo); his.append(hi)
means, los, his = np.array(means), np.array(los), np.array(his)

ax.errorbar(
    xpos, means, yerr=[means - los, his - means],
    fmt="none", elinewidth=1.1, capsize=2.4, ecolor="#1f1f1f", zorder=5
)
ax.scatter(xpos, means, s=14, c="#1f1f1f", zorder=6)

# section labels
label_face = {
    "High":     "#BEE6BE",
    "Mid-High": "#C4DDF3",
    "Mid-Low":  "#F2D3A9",
    "Low":      "#F3B8B8"
}
current = 0
ymax = 1.02
ax.set_ylim(0.50, ymax)
for b in bands_in_order:
    k = len(band_blocks[b])
    if k == 0: continue
    start, end = current, current + k - 1
    mid = (start + end) / 2.0
    ax.text(
        mid, ymax + 0.010, b,
        ha="center", va="bottom", fontsize=6.8, color="#1f2937", weight="bold",
        bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.08",
                  fc=label_face[b], ec="#444444", lw=0.4, alpha=0.95),
        transform=ax.get_xaxis_transform(), zorder=10
    )
    current += k

# axes
ax.set_yticks(np.arange(0.40, 1.05, 0.05))
ax.set_xlim(-0.6, n - 0.4)
ax.set_ylabel("Average automatic score", fontsize=7, fontfamily="sans-serif")
ax.set_xlabel("Species name", fontsize=7,fontfamily="sans-serif")
ax.set_xticks(xpos)
ax.set_xticklabels(formatted_xticks, rotation=35, ha="right", rotation_mode="anchor")

# inner black border only
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(1.0)

ax.set_facecolor("#FAFAFA")
ax.xaxis.grid(False)

plt.subplots_adjust(bottom=0.44, top=0.88, left=0.10, right=0.98)
plt.savefig(SAVE_PATH, bbox_inches="tight",transparent=True)
plt.show()
print(f"Saved: {SAVE_PATH}")
