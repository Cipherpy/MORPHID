#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CER_COL_CAND = ["CER", "cer", "char_error_rate", "character_error_rate", "error"]
SPECIES_COL_CAND = ["actual_label", "Actual_label", "Actual_Label", "label", "true_label", "species"]

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def clean_species(x):
    s = str(x).strip()
    s = re.sub(r"(?i)^class\s*detected\s*:\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" ,;")

def load_one_csv(path):
    df = pd.read_csv(path)
    cer_col = pick_col(df, CER_COL_CAND)
    sp_col  = pick_col(df, SPECIES_COL_CAND)

    if cer_col is None or sp_col is None:
        raise ValueError(f"Missing CER/species column in {path}")

    out = pd.DataFrame({
        "species": df[sp_col].map(clean_species),
        "CER": pd.to_numeric(df[cer_col], errors="coerce")
    })

    out = out.dropna()
    out["CER"] = out["CER"].clip(0, 1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cer-dir", required=True)
    ap.add_argument("--map-csv", required=True)
    ap.add_argument("--out-scatter", default="species_scatter_coloured.png")
    args = ap.parse_args()

    m = pd.read_csv(args.map_csv)
    rows = []

    for _, r in m.iterrows():
        path = os.path.join(args.cer_dir, r["file"])
        d = load_one_csv(path)
        d["model"] = r["model"]
        rows.append(d)

    long = pd.concat(rows, ignore_index=True)

    # ---- Species × Model mean CER ----
    sp = (long.groupby(["species", "model"], as_index=False)
              .agg(cer_mean=("CER", "mean")))

    wide = sp.pivot(index="species", columns="model", values="cer_mean").dropna()

    models = list(wide.columns)
    if len(models) < 2:
        raise RuntimeError("Need at least two models")

    xcol, ycol = models[0], models[1]

    # ---- Colour map for species ----
    species = wide.index.tolist()
    cmap = plt.get_cmap("tab20")
    colors = {sp: cmap(i % 20) for i, sp in enumerate(species)}

    # ---- Scatter plot ----
    plt.figure(figsize=(6.5, 6.5), dpi=300)

    for sp_name in species:
        plt.scatter(
            wide.loc[sp_name, xcol],
            wide.loc[sp_name, ycol],
            s=45,
            color=colors[sp_name],
            edgecolor="black",
            linewidth=0.4
        )

    # y = x reference
    lo = min(wide[xcol].min(), wide[ycol].min())
    hi = max(wide[xcol].max(), wide[ycol].max())
    plt.plot([lo, hi], [lo, hi], "--", linewidth=1)

    plt.xlabel(f"Mean CER ({xcol})")
    plt.ylabel(f"Mean CER ({ycol})")
    #plt.title("Species-wise CER comparison")

    # ---- Legend with species names ----
    legend_handles = [
        Line2D([0], [0],
               marker="o",
               color="w",
               markerfacecolor=colors[sp],
               markeredgecolor="black",
               markersize=6,
               label=sp)
        for sp in species
    ]

    plt.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        fontsize=8,
        title="Species"
    )

    plt.tight_layout()
    plt.savefig(args.out_scatter, dpi=1200, bbox_inches="tight")
    plt.close()

    print("[OK] Saved:", args.out_scatter)

if __name__ == "__main__":
    main()
