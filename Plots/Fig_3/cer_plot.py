#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Canonical plotting order
FEATURE_ORDER = [
    "Shape",
    "Ostium",
    "Cauda",
    "Sulcus acusticus",
    "Anterior region",
    "Posterior region",
]

# Try to find the CER column robustly
CER_COL_CAND = ["CER", "cer", "char_error_rate", "character_error_rate", "error"]
IMAGE_COL_CAND = ["Image", "image", "path", "file", "filename"]


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def load_one_csv(path):
    df = pd.read_csv(path)
    cer_col = pick_col(df, CER_COL_CAND)
    img_col = pick_col(df, IMAGE_COL_CAND)

    if cer_col is None:
        raise ValueError(f"Could not find CER column in {os.path.basename(path)}. Columns={list(df.columns)}")

    out = pd.DataFrame()
    out["image"] = df[img_col].astype(str) if img_col else np.arange(len(df)).astype(str)
    out["CER"] = pd.to_numeric(df[cer_col], errors="coerce")
    out = out.dropna(subset=["CER"]).copy()

    # CER typically 0..1, keep safe bounds
    out["CER"] = out["CER"].clip(0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot CER for 6 features × 2 models using an explicit mapping file.")
    ap.add_argument("--cer-dir", required=True, help="Folder containing the 12 CER CSVs")
    ap.add_argument("--map-csv", required=True, help="Mapping CSV with columns: file,model,feature")
    ap.add_argument("--out-png", default="cer_6features_2models_boxjitter.png")
    ap.add_argument("--out-long", default="cer_all_long.csv")
    ap.add_argument("--out-summ", default="cer_summary_by_model_feature.csv")
    args = ap.parse_args()

    cer_dir = args.cer_dir
    map_csv = args.map_csv

    m = pd.read_csv(map_csv)
    required_cols = {"file", "model", "feature"}
    if not required_cols.issubset(m.columns):
        raise ValueError(f"Mapping CSV must contain columns {required_cols}. Found: {list(m.columns)}")

    rows = []
    missing_files = []

    for _, r in m.iterrows():
        f = str(r["file"])
        model = str(r["model"])
        feature = str(r["feature"])

        path = f if os.path.isabs(f) else os.path.join(cer_dir, f)
        if not os.path.exists(path):
            missing_files.append(path)
            continue

        d = load_one_csv(path)
        d["model"] = model
        d["feature"] = feature
        d["source_file"] = os.path.basename(path)
        rows.append(d)

    if missing_files:
        print("\n[WARN] These files were listed in map but not found:")
        for p in missing_files:
            print("  -", p)

    if not rows:
        raise RuntimeError("No files were parsed. Check cer-dir and map-csv paths, and whether files exist.")

    long = pd.concat(rows, ignore_index=True)

    # enforce feature order
    long["feature"] = pd.Categorical(long["feature"], categories=FEATURE_ORDER, ordered=True)
    long = long.dropna(subset=["feature"]).sort_values(["feature", "model"]).reset_index(drop=True)

    # Save long table
    long.to_csv(args.out_long, index=False)

    # Summary stats
    summ = long.groupby(["model", "feature"]).agg(
        n=("CER", "count"),
        cer_mean=("CER", "mean"),
        cer_median=("CER", "median"),
        cer_std=("CER", "std"),
        cer_q25=("CER", lambda x: np.quantile(x, 0.25)),
        cer_q75=("CER", lambda x: np.quantile(x, 0.75)),
    ).reset_index()
    summ.to_csv(args.out_summ, index=False)

    # ---- Plot: grouped boxplot + jitter (single figure) ----
    features = [f for f in FEATURE_ORDER if f in long["feature"].astype(str).unique()]
    models = list(long["model"].unique())

    # keep a stable order: Gemma then LLaMA if present
    preferred = ["Gemma-3", "LLaMA-3.2 Vision"]
    models = [x for x in preferred if x in models] + [x for x in models if x not in preferred]

    data, positions = [], []
    gap = 0.8
    offset = 0.18
    width = 0.28

    x = 1.0
    xticks = []
    xticklabels = []

    for feat in features:
        xticks.append(x)
        xticklabels.append(feat)

        for mi, model in enumerate(models):
            vals = long[(long["feature"] == feat) & (long["model"] == model)]["CER"].values
            data.append(vals)
            positions.append(x + (-offset if mi == 0 else offset))

        x += (1 + gap)

    plt.figure(figsize=(10.5, 4.8), dpi=300)

    bp = plt.boxplot(
        data,
        positions=positions,
        widths=width,
        showfliers=False,
        patch_artist=True,
    )

    # distinguish models with hatch patterns (print-safe)
    for i, box in enumerate(bp["boxes"]):
        if i % 2 == 0:
            box.set_hatch("///")
        else:
            box.set_hatch("...")

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        px = np.full(len(vals), positions[i]) + rng.normal(0, 0.03, size=len(vals))
        plt.scatter(px, vals, s=10, alpha=0.35)

    plt.xticks(xticks, xticklabels, rotation=20, ha="right")
    plt.ylabel("Character error rate (CER)")
    plt.xlabel("Morphological feature")
    plt.ylim(0, 1)
    #plt.title("Feature-wise CER: Gemma-3 vs LLaMA-3.2 Vision")

    from matplotlib.patches import Patch
    legend_handles = []
    if len(models) >= 1:
        legend_handles.append(Patch(facecolor="white", edgecolor="black", hatch="///", label=models[0]))
    if len(models) >= 2:
        legend_handles.append(Patch(facecolor="white", edgecolor="black", hatch="...", label=models[1]))
    plt.legend(
    handles=legend_handles,
    loc="upper right",
    # bbox_to_anchor=(1.02, 1.0),
    # borderaxespad=0,
    frameon=True
)


    plt.tight_layout()
    plt.savefig(args.out_png, dpi=1200)
    plt.close()

    print("[OK] Saved plot:", args.out_png)
    print("[OK] Saved long table:", args.out_long)
    print("[OK] Saved summary:", args.out_summ)


if __name__ == "__main__":
    main()
