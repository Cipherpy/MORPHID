import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# -------------------------
# Font settings (serif, size 7)
# -------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7
})

# -------------------------
# Load CSV
# -------------------------
CSV_FILE = "/home/reshma/MORPHID/Plots/Fig_2/Grayscale_gemma_captions_scores.csv"
df = pd.read_csv(CSV_FILE)

# -------------------------
# Group by species (actual_label) and compute mean scores
# -------------------------
metrics = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L"]
species_matrix = df.groupby("actual_label")[metrics].mean()

# -------------------------
# Save to CSV
# -------------------------
species_matrix.to_csv("species_scores_matrix.csv")
print("✅ Species-wise score matrix saved: species_scores_matrix.csv")

# -------------------------
# Heatmap visualization
# -------------------------
plt.figure(figsize=(10, 6))
ax = sns.heatmap(
    species_matrix,
    annot=True,
    cmap="Oranges",
    fmt=".3f",
    linewidths=0.5,
    linecolor="black",
    #cbar_kws={'label': 'Score'},
    cbar=False,  # 🔹 remove color bar
    annot_kws={"color": "black"}  # values inside plot in black
)

# 🔹 Format Y-axis tick labels properly
new_labels = []
for label in ax.get_yticklabels():
    text = label.get_text().strip()
    parts = text.split(" ", 1)  # split into genus and rest
    if len(parts) == 2 and parts[1].strip().startswith("sp"):
        # Case: Genus sp. → italicize only Genus
        new_text = r"$\it{" + parts[0] + r"}$ " + parts[1]
    elif len(parts) == 2:
        # Case: Genus species → italicize both, keep space
        new_text = r"$\it{" + parts[0] + r"}$ " + r"$\it{" + parts[1] + r"}$"
    else:
        # Single-word label (just in case) → italicize whole
        new_text = r"$\it{" + text + r"}$"
    new_labels.append(new_text)

ax.set_yticklabels(new_labels)

plt.xlabel("Metrics", fontsize=7)
plt.ylabel("Species name", fontsize=7)
plt.tight_layout()
plt.savefig("species_scores_heatmap.png", dpi=1200, transparent=True, bbox_inches="tight")
plt.show()
