import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Settings
# -------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 120,
    "savefig.dpi": 1200,
})

CSV_FILE = "/home/reshma/MORPHID/Plots/Fig_2/caption_scores_llama.csv" 
METRICS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L"]

# --------- taxonomic label formatter ----------
def compress_taxon_label(name: str) -> str:
    parts = name.strip().split()
    if len(parts) == 2 and parts[1].lower().startswith("sp"):
        return r"$\it{" + parts[0] + r"}$ " + parts[1]
    elif len(parts) >= 2:
        genus_abbrev = parts[0][0] + "."
        return r"$\it{" + genus_abbrev + r"}$ " + r"$\it{" + " ".join(parts[1:]) + r"}$"
    else:
        return r"$\it{" + name + r"}$"

# -------------------------
# Load & wrangle
# -------------------------
df = pd.read_csv(CSV_FILE)

long = df.melt(
    id_vars=["Image", "actual_label", "predicted_label"],
    value_vars=METRICS,
    var_name="Metric",
    value_name="Score"
).dropna(subset=["Score"])

species = sorted(long["actual_label"].astype(str).unique())
long["actual_label"] = pd.Categorical(long["actual_label"], categories=species, ordered=True)

label_map = {s: compress_taxon_label(s) for s in species}
long["pretty_label"] = long["actual_label"].map(label_map)

# -------------------------
# Color palette (vibrant but professional)
# -------------------------
# Okabe-Ito palette (8 distinct, colorblind-safe)
okabe_ito = ["#E69F00","#56B4E9","#009E73","#F0E442",
             "#0072B2","#D55E00","#CC79A7","#999999"]

# Use one color per metric (cycled if > palette length)
palette = {m: okabe_ito[i % len(okabe_ito)] for i, m in enumerate(METRICS)}

# -------------------------
# Plot boxplots
# -------------------------
sns.set_theme(style="whitegrid")

for metric in METRICS:
    sub = long[long["Metric"] == metric]
    fig_width = max(5, 0.4 * len(species))    
    fig_height = max(5, 5.5)                  
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Box color pastel, outline with darker edge
    sns.boxplot(
        data=sub, x="pretty_label", y="Score",
        ax=ax, fliersize=0, linewidth=0.8,
        boxprops=dict(facecolor=palette[metric], alpha=0.35, edgecolor=palette[metric]),
        whiskerprops=dict(color=palette[metric]),
        capprops=dict(color=palette[metric]),
        medianprops=dict(color="black", linewidth=1.0)
    )
    # Strip plot with same base color, darker & more opaque
    sns.stripplot(
        data=sub, x="pretty_label", y="Score",
        ax=ax, size=2, jitter=0.25, alpha=0.7, color=palette[metric], edgecolor="black", linewidth=0.2
    )
    
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    # ax.set_title(f"{metric}", fontweight="bold")
    ax.set_ylim(0, 1)
    
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha("center")
    
    plt.tight_layout()
    out_path = f"species_boxplot_{metric.replace('-', '').replace(' ', '_')}.png"
    plt.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"✅ Saved: {out_path}")
