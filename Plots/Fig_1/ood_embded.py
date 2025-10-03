# Interval Box Plot — Pastel & Vibrant, Wider Boxes (FIXED OVERLAP)
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, Patch
from matplotlib.patheffects import withStroke

mpl.rcParams.update({
    "font.family": "sans-serif",
    # "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 7,
    "axes.linewidth": 0.9
})

# -----------------------------
# Data
# -----------------------------
metrics = ["Mahalanobis distance", "PCA residual", "KNN mean"]
id_ranges  = [(31, 77),   (1.25, 3.20), (2.45, 4.99)]
ood_ranges = [(90, 227),  (4.6, 13.0),  (9.7, 34.0)]

# Species-6 overlap is only relevant to the Mahalanobis metric
OVERLAP_METRIC_NAME = "Mahalanobis distance"
species6_overlap_maha = (58, 77)

# -----------------------------
# Colors (pastel fills + vibrant edges)
# -----------------------------
C_ID_FILL  = "#A8DADC"   # pastel teal
C_OOD_FILL = "#F7A8B8"   # pastel pink
C_OV_FILL  = "#CDB4DB"   # pastel purple
C_EDGE     = "#2B2D42"   # deep slate
C_TXT      = "#1F2630"   # dark text

# -----------------------------
# Helpers
# -----------------------------
def draw_interval_box(ax, y, lo, hi, fill, height=0.40, edge=C_EDGE, lw=1.2, label=None, z=3):
    """Horizontal 'box' spanning [lo, hi] centered at y, with a midpoint line."""
    width = hi - lo
    rect = Rectangle((lo, y - height/2.0), width, height,
                     facecolor=fill, edgecolor=edge, lw=lw, zorder=z, alpha=0.95)
    ax.add_patch(rect)
    xm = (lo + hi) / 2.0
    ax.plot([xm, xm], [y - height/2.0 + 0.03, y + height/2.0 - 0.03],
            color=edge, lw=lw, zorder=z+1)
    if label is not None:
        rect.set_label(label)
    return rect

def annotate_gap(ax, y, id_hi, ood_lo):
    gap = ood_lo - id_hi
    if gap > 0:
        ax.annotate(f"gap = {gap:.2f}",
                    xy=((id_hi + ood_lo)/2.0, y - 0.52),
                    xytext=((id_hi + ood_lo)/2.0, y - 0.52),
                    ha="center", va="top", color=C_TXT, fontsize=5,
                    path_effects=[withStroke(linewidth=2.0, foreground="white")])
        ax.plot([id_hi, ood_lo], [y - 0.48, y - 0.48], lw=1.0, color=C_TXT)
        ax.plot([id_hi, id_hi], [y - 0.52, y - 0.44], lw=1.0, color=C_TXT)
        ax.plot([ood_lo, ood_lo], [y - 0.52, y - 0.44], lw=1.0, color=C_TXT)

# -----------------------------
# Figure
# -----------------------------
fig, ax = plt.subplots(figsize=(10.5, 5.6))

for i, m in enumerate(metrics):
    y_id, y_ood = i + 0.22, i - 0.22
    id_lo, id_hi   = id_ranges[i]
    ood_lo, ood_hi = ood_ranges[i]

    # Pastel boxes
    draw_interval_box(ax, y_id,  id_lo,  id_hi,  C_ID_FILL,  label="ID-test" if i == 0 else None)
    draw_interval_box(ax, y_ood, ood_lo, ood_hi, C_OOD_FILL, label="OOD"      if i == 0 else None)

    # Hatched Species-6 overlap ONLY for the Mahalanobis row
    if m == OVERLAP_METRIC_NAME:
        ov0 = max(species6_overlap_maha[0], id_lo)
        ov1 = min(species6_overlap_maha[1], id_hi)
        if ov1 > ov0:
            # make the band just a hair taller than the boxes so it peeks out cleanly
            ax.add_patch(Rectangle((ov0, i - 0.46), ov1 - ov0, 0.92,
                                   facecolor=C_OV_FILL, edgecolor=C_OV_FILL,
                                   hatch="///", alpha=0.25, lw=0.8, zorder=2))
            # ax.text((ov0 + ov1) / 2.0, i,
            #         "Species-6 overlap", ha="center", va="center",
            #         color=C_EDGE, fontsize=7, zorder=4)

    # Gap annotation (if any)
    annotate_gap(ax, i, id_hi, ood_lo)

# Cosmetics
ax.set_yticks(range(len(metrics)))
ax.set_yticklabels(metrics)
ax.set_xlabel("Metric value", color=C_TXT)
ax.set_title("ID vs OOD — Pastel Interval Box Plot (wider boxes)", color=C_TXT)

# Soft alternating bands
for j in range(len(metrics)):
    if j % 2 == 0:
        ax.axhspan(j - 0.5, j + 0.5, color="#FAFAFA", zorder=0)

ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)

# Legend (shows overlap even if not visible in some configs)
legend_items = [
    Patch(facecolor=C_ID_FILL,  edgecolor=C_EDGE, label="ID-test"),
    Patch(facecolor=C_OOD_FILL, edgecolor=C_EDGE, label="OOD"),
    Patch(facecolor=C_OV_FILL,  edgecolor=C_OV_FILL, hatch="///", alpha=0.25, label="Species 6 (Deep sea) overlap")
]
ax.legend(handles=legend_items, loc="upper right", frameon=False, ncol=3)

plt.tight_layout()
plt.savefig("interval_boxplot_ID_vs_OOD_pastel_wide.png",
            dpi=1200, transparent=True, bbox_inches="tight")
plt.show()
