import pandas as pd
import re

# -------------------------
# Helper: extract Shape up to first comma
# -------------------------
def extract_shape(text):
    """
    Extracts text after 'Shape:' until first comma or end.
    Returns '' if missing.
    """
    if pd.isna(text):
        return ""
    m = re.search(r"Shape:\s*([^\.]*)", str(text), flags=re.IGNORECASE)
    if not m:
        return ""

    s = m.group(1).strip()
    if "," in s:
        s = s.split(",", 1)[0].strip()
    return s


# -------------------------
# Load CSV
# -------------------------
CSV_IN = "/home/reshma/MORPHID/Plots/Fig_3/paired_captions_minimall_modi_gemma.csv"
df = pd.read_csv(CSV_IN)

# -------------------------
# Extract shapes
# -------------------------
df["actual_shape"]    = df["Description"].apply(extract_shape)
df["predicted_shape"] = df["generated_caption"].apply(extract_shape)

# -------------------------
# Keep only cases where predicted != actual (WRONG predictions)
# -------------------------
wrong = df[df["actual_shape"].str.lower() != df["predicted_shape"].str.lower()].copy()

# -------------------------
# Save clean CSV
# -------------------------
OUT_CSV = "shape_mismatch_cases_gemma.csv"
wrong[[
    "Image",
    "actual_shape",
    "predicted_shape",
    "generated_caption",
    "Description"
]].to_csv(OUT_CSV, index=False)

print(f"Saved → {OUT_CSV}")
print("Total wrong cases:", len(wrong))
