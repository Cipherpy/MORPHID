# import matplotlib.pyplot as plt

# # -----------------------------
# # Data (as provided)
# # -----------------------------
# metrics = ["maha_min", "pca_residual", "knn_mean"]
# id_ranges = [(31, 77), (1.25, 3.20), (2.45, 4.99)]
# ood_ranges = [(90, 227), (4.6, 13), (9.7, 34)]

# # Species-specific overlap (for maha_min only)
# # Deepsea Species 6 observed around ~58–90 → overlap with ID is 58–77
# species6_overlap_maha = (58, 77)

# # -----------------------------
# # Helper: plot function
# # -----------------------------
# def plot_ranges(ax, metrics, id_ranges, ood_ranges, species6_overlap=None, normalized=False):
#     for i, metric in enumerate(metrics):
#         # ID range (upper)
#         ax.plot([id_ranges[i][0], id_ranges[i][1]], [i + 0.22, i + 0.22],
#                 lw=8, solid_capstyle="butt", color="blue", label="In-Distribution" if i == 0 else "")
#         # OOD range (lower)
#         ax.plot([ood_ranges[i][0], ood_ranges[i][1]], [i - 0.22, i - 0.22],
#                 lw=8, solid_capstyle="butt", color="red", label="Out-of-Distribution" if i == 0 else "")

#         # Species-specific overlap only for maha_min
#         if metric == "maha_min" and species6_overlap is not None:
#             ov_start = max(species6_overlap[0], id_ranges[i][0])
#             ov_end   = min(species6_overlap[1], id_ranges[i][1])
#             if ov_start < ov_end:
#                 ax.fill_betweenx([i - 0.3, i + 0.3], ov_start, ov_end,
#                                  alpha=0.25, hatch="///", color="purple")
#                 ax.text((ov_start + ov_end) / 2, i, "Species 6 overlap",
#                         ha="center", va="center", fontsize=9, color="purple")

#     ax.set_yticks(range(len(metrics)))
#     ax.set_yticklabels(metrics)
#     ax.set_xlabel("Normalized Metric Value (0–1)" if normalized else "Metric value")
#     ax.set_title("Normalized ID vs OOD Ranges with Species-Specific Overlap (maha_min)" if normalized
#                  else "ID vs OOD Ranges with Species-Specific Overlap (maha_min)")
#     ax.grid(axis="x", linestyle="--", alpha=0.5)
#     ax.legend(loc="upper right", frameon=False)

# # -----------------------------
# # 1) RAW FIGURE
# # -----------------------------
# fig_raw, ax_raw = plt.subplots(figsize=(8, 5))
# plot_ranges(ax_raw, metrics, id_ranges, ood_ranges,
#             species6_overlap=species6_overlap_maha, normalized=False)

# plt.tight_layout()
# plt.savefig("ID_vs_OOD_metrics_species6_overlap_RAW.png",
#             dpi=300, bbox_inches="tight")
# plt.close(fig_raw)

# # -----------------------------
# # 2) NORMALIZED FIGURE (per metric, min–max 0–1)
# # -----------------------------
# id_norm, ood_norm, overlap_norm = [], [], [None, None, None]

# for i, metric in enumerate(metrics):
#     min_val = min(id_ranges[i][0], ood_ranges[i][0])
#     max_val = max(id_ranges[i][1], ood_ranges[i][1])
#     span = max_val - min_val

#     # Normalize
#     id_norm.append(((id_ranges[i][0] - min_val) / span,
#                     (id_ranges[i][1] - min_val) / span))
#     ood_norm.append(((ood_ranges[i][0] - min_val) / span,
#                      (ood_ranges[i][1] - min_val) / span))

#     # Normalize overlap only for maha_min
#     if metric == "maha_min":
#         ov_start = max(species6_overlap_maha[0], id_ranges[i][0])
#         ov_end   = min(species6_overlap_maha[1], id_ranges[i][1])
#         if ov_start < ov_end:
#             overlap_norm[i] = ((ov_start - min_val) / span,
#                                (ov_end - min_val) / span)

# # Plot normalized
# fig_norm, ax_norm = plt.subplots(figsize=(8, 5))
# for i, metric in enumerate(metrics):
#     ax_norm.plot([id_norm[i][0], id_norm[i][1]], [i + 0.22, i + 0.22],
#                  lw=8, solid_capstyle="butt", color="blue",
#                  label="In-Distribution" if i == 0 else "")
#     ax_norm.plot([ood_norm[i][0], ood_norm[i][1]], [i - 0.22, i - 0.22],
#                  lw=8, solid_capstyle="butt", color="red",
#                  label="Out-of-Distribution" if i == 0 else "")

#     # Overlap shading if available
#     if overlap_norm[i] is not None:
#         ov_start, ov_end = overlap_norm[i]
#         ax_norm.fill_betweenx([i - 0.3, i + 0.3], ov_start, ov_end,
#                               alpha=0.25, hatch="///", color="purple")
#         ax_norm.text((ov_start + ov_end) / 2, i, "Species 6 overlap",
#                      ha="center", va="center", fontsize=9, color="purple")

# ax_norm.set_yticks(range(len(metrics)))
# ax_norm.set_yticklabels(metrics)
# ax_norm.set_xlabel("Normalized Metric Value (0–1)")
# ax_norm.set_title("Normalized ID vs OOD Ranges with Species-Specific Overlap (maha_min)")
# ax_norm.grid(axis="x", linestyle="--", alpha=0.5)
# ax_norm.legend(loc="upper right", frameon=False)

# plt.tight_layout()
# plt.savefig("ID_vs_OOD_metrics_species6_overlap_NORMALIZED.png",
#             dpi=300, bbox_inches="tight")
# plt.close(fig_norm)



# seaborn aesthetic version of your plot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# Global font settings
# -----------------------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["font.size"] = 8

# -----------------------------
# Data
# -----------------------------
metrics = ["maha_min", "pca_residual", "knn_mean"]
id_ranges = [(31, 77), (1.25, 3.20), (2.45, 4.99)]
ood_ranges = [(90, 227), (4.6, 13), (9.7, 34)]
species6_overlap_maha = (58, 77)

# -----------------------------
# Seaborn theme
# -----------------------------
sns.set_theme(style="white", context="paper")   # paper context for smaller fonts
c_id  = sns.color_palette("Blues", 6)[4]
c_ood = sns.color_palette("Reds", 6)[4]
c_ov  = sns.color_palette("Purples", 6)[3]

def _draw_range(ax, y, x0, x1, color, label=None):
    ax.plot([x0, x1], [y, y],
            lw=6, color=color, solid_capstyle="butt", label=label, zorder=2)
    ax.plot([x0, x1], [y, y],
            marker="|", ms=12, mec=color, mfc=color, lw=0, zorder=3)

# -----------------------------
# Helper: plot function
# -----------------------------
def plot_ranges_seaborn(ax, metrics, id_ranges, ood_ranges, species6_overlap=None,
                        normalized=False):
    for i, metric in enumerate(metrics):
        y_id  = i + 0.22
        y_ood = i - 0.22

        _draw_range(ax, y_id,  id_ranges[i][0],  id_ranges[i][1],
                    c_id,  label="ID-test" if i == 0 else None)
        _draw_range(ax, y_ood, ood_ranges[i][0], ood_ranges[i][1],
                    c_ood, label="OOD" if i == 0 else None)

        if metric == "maha_min" and species6_overlap is not None:
            ov_start = max(species6_overlap[0], id_ranges[i][0])
            ov_end   = min(species6_overlap[1], id_ranges[i][1])
            if ov_start < ov_end:
                ax.fill_betweenx([i - 0.32, i + 0.32], ov_start, ov_end,
                                 color=c_ov, alpha=0.25, hatch="///", edgecolor=c_ov,
                                 linewidth=0.0, zorder=1)
                ax.text((ov_start + ov_end) / 2, i,
                        "Species 6 (Deep sea) overlap",
                        ha="center", va="center", fontsize=12, color=c_ov, zorder=4)

    # axes cosmetics
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics,fontsize=12)
    ax.set_xlabel("Normalized metric value (0–1)" if normalized else "Metric value",fontsize=12)
    sns.despine(ax=ax, left=False, bottom=False)
    ax.legend(loc="upper right", frameon=False, fontsize=12)

# -----------------------------
# Draw & save
# -----------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.5))
plot_ranges_seaborn(ax, metrics, id_ranges, ood_ranges,
                    species6_overlap=species6_overlap_maha, normalized=False)

plt.tight_layout()
plt.savefig("ID_vs_OOD_metrics_species6_overlap_seaborn.png",
            dpi=300, bbox_inches="tight",transparent=True)
plt.show()
