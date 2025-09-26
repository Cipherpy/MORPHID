import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- TEXT EXTRACTION -------------------
def extract_raw_sulcus_acusticus(text):
    """Extract the exact raw text after 'Sulcus acusticus:' up to the next period or end."""
    if pd.isna(text):
        return ""
    m = re.search(r'anterior region:\s*([^\.]*)', str(text), flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""

# ---- load & prep
df = pd.read_csv("/home/reshma/Otolith/captioning/otolith/llama/llama_OOD_output_filtered.csv")

df["sulcus_acusticus_gt_raw"]  = df["Description"].apply(extract_raw_sulcus_acusticus)
df["sulcus_acusticus_gen_raw"] = df["generated_caption"].apply(extract_raw_sulcus_acusticus)

filtered = df[(df["sulcus_acusticus_gt_raw"] != "") & (df["sulcus_acusticus_gen_raw"] != "")]

# ------------------- CER CALCULATION -------------------
# Install if needed: pip install python-Levenshtein
import Levenshtein

def cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER).
    CER = (edit distance between reference and hypothesis) / length of reference
    """
    if len(reference) == 0:
        return float('inf')  # avoid division by zero
    distance = Levenshtein.distance(reference, hypothesis)
    return distance / len(reference)

# Apply CER row-wise
filtered["CER"] = filtered.apply(
    lambda row: cer(row["sulcus_acusticus_gt_raw"], row["sulcus_acusticus_gen_raw"]), axis=1
)

# ------------------- SUMMARY -------------------
print("Mean CER:", filtered["CER"].mean())
print("Median CER:", filtered["CER"].median())
print("Sample rows with CER:\n", filtered[["sulcus_acusticus_gt_raw", "sulcus_acusticus_gen_raw", "CER"]].head())

# ------------------- OPTIONAL: DISTRIBUTION PLOT -------------------
plt.figure(figsize=(6,4))
sns.histplot(filtered["CER"], bins=30, kde=True, color="royalblue")
plt.xlabel("Character Error Rate (CER)")
plt.ylabel("Count")
plt.title("Distribution of CER for Sulcus acusticus descriptions")
plt.tight_layout()
plt.show()
