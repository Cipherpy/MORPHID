import pandas as pd
import numpy as np
from typing import Dict, Iterable, Tuple, List, Optional

# ------------------ utils ------------------
def _clean_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan).dropna()

def _mean_ci(x: np.ndarray, n_boot: int = 2000, ci: float = 0.95, rng: np.random.Generator = None) -> Tuple[float, Tuple[float, float]]:
    """Bootstrap mean and CI."""
    if rng is None:
        rng = np.random.default_rng(42)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, (np.nan, np.nan)
    m = x.mean()
    bs = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(bs, [(1-ci)/2, 1-(1-ci)/2])
    return float(m), (float(lo), float(hi))

def _summarize_split(
    df_id: pd.DataFrame,
    df_ood: pd.DataFrame,
    metrics: Iterable[str],
    higher_is_better: Dict[str, bool],
    with_ci: bool = True,
    n_boot: int = 2000,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute summary for overall or grouped subsets.
    group_cols: None => overall; otherwise group by these columns (e.g., ['class'], ['model'], ['class','model'])
    """
    rng = np.random.default_rng(123)
    out = []

    if group_cols is None or len(group_cols) == 0:
        groups = [({}, (df_id, df_ood))]  # single group
    else:
        # Align groups present in either ID or OOD (outer join on groups)
        keys_id  = df_id[group_cols].drop_duplicates()
        keys_ood = df_ood[group_cols].drop_duplicates()
        keys = pd.concat([keys_id, keys_ood], ignore_index=True).drop_duplicates()

        groups = []
        for _, row in keys.iterrows():
            key = row.to_dict()
            mask_id  = (df_id[group_cols] == row.values).all(axis=1)
            mask_ood = (df_ood[group_cols] == row.values).all(axis=1)
            groups.append((key, (df_id[mask_id], df_ood[mask_ood])))

    for key, (g_id, g_ood) in groups:
        for m in metrics:
            s_id  = _clean_series(g_id[m]) if m in g_id else pd.Series([], dtype=float)
            s_ood = _clean_series(g_ood[m]) if m in g_ood else pd.Series([], dtype=float)

            if with_ci:
                mean_id,  ci_id  = _mean_ci(s_id.values,  n_boot=n_boot, rng=rng) if len(s_id)  else (np.nan, (np.nan, np.nan))
                mean_ood, ci_ood = _mean_ci(s_ood.values, n_boot=n_boot, rng=rng) if len(s_ood) else (np.nan, (np.nan, np.nan))
            else:
                mean_id  = float(s_id.mean())  if len(s_id)  else np.nan
                mean_ood = float(s_ood.mean()) if len(s_ood) else np.nan
                ci_id = ci_ood = (np.nan, np.nan)

            G = mean_id - mean_ood
            rho = (mean_ood / mean_id) if (mean_id is not np.nan and mean_id != 0) else np.nan

            row_out = {
                **key,
                "metric": m,
                "dir": "higher_is_better" if higher_is_better.get(m, True) else "lower_is_better",
                "mean_ID": mean_id,
                "mean_ID_CI_low": ci_id[0],
                "mean_ID_CI_high": ci_id[1],
                "mean_OOD": mean_ood,
                "mean_OOD_CI_low": ci_ood[0],
                "mean_OOD_CI_high": ci_ood[1],
                "gap_G": G,
                "stability_rho": rho
            }
            out.append(row_out)

    return pd.DataFrame(out)


# ------------------ main API ------------------
def load_id_ood_csv(
    id_csv_path: str,
    ood_csv_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load two Excel files (or specific sheets) into dataframes.
    """
    df_id  = pd.read_csv(id_csv_path)
    df_ood = pd.read_csv(ood_csv_path)
    return df_id, df_ood

def autodetect_metrics(df: pd.DataFrame, extra_keep: Optional[List[str]] = None) -> List[str]:
    """
    Auto-detect BLEU/ROUGE-like metric columns.
    Keeps columns matching prefixes: BLEU, BLEU_, BLEU- , ROUGE, ROUGE_
    """
    cols = []
    for c in df.columns:
        lc = str(c).lower()
        if lc.startswith("bleu") or lc.startswith("rouge"):
            cols.append(c)
    # Optional extras
    if extra_keep:
        for c in extra_keep:
            if c in df.columns and c not in cols:
                cols.append(c)
    return cols

def compute_ood_from_csv(
    id_csv_path: str,
    ood_csv_path: str,
    id_sheet: Optional[str] = None,
    ood_sheet: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    higher_is_better: Optional[Dict[str, bool]] = None,
    with_ci: bool = True,
    n_boot: int = 2000,
    group_cols_available: Optional[List[str]] = None,
):
    """
    Load ID/OOD Excel files and compute:
      - overall summary,
      - by-class summary (if 'class' col exists),
      - by-model summary (if 'model' col exists),
      - by (class × model) summary (if both exist).

    Returns a dict of DataFrames.
    """
    df_id, df_ood = load_id_ood_csv(id_csv_path, ood_csv_path)

    # Harmonize columns (outer-join style): keep only metrics existing in either
    if metrics is None:
        # detect from union of columns
        metrics_id  = autodetect_metrics(df_id)
        metrics_ood = autodetect_metrics(df_ood)
        metrics = sorted(list(set(metrics_id).union(metrics_ood)))
    if not metrics:
        raise ValueError("No metric columns detected/provided (expected BLEU*/ROUGE* columns).")

    # default directions (BLEU/ROUGE higher is better)
    if higher_is_better is None:
        higher_is_better = {m: True for m in metrics}

    # Identify grouping columns present (optional)
    if group_cols_available is None:
        group_cols_available = []
        for cand in ["actual_label", "species", "region", "model"]:
            if (cand in df_id.columns) or (cand in df_ood.columns):
                group_cols_available.append(cand)

    # Ensure missing group columns exist as NA to align grouping
    for gc in group_cols_available:
        if gc not in df_id.columns:
            df_id[gc] = pd.NA
        if gc not in df_ood.columns:
            df_ood[gc] = pd.NA

    # Overall
    overall = _summarize_split(
        df_id=df_id, df_ood=df_ood,
        metrics=metrics,
        higher_is_better=higher_is_better,
        with_ci=with_ci, n_boot=n_boot,
        group_cols=None
    )

    # By-class (if 'class' exists)
    by_class = pd.DataFrame()
    if "actual_label" in group_cols_available:
        by_class = _summarize_split(
            df_id, df_ood, metrics, higher_is_better,
            with_ci, n_boot, group_cols=["actual_label"]
        )

    # By-model (if 'model' exists)
    by_model = pd.DataFrame()
    if "model" in group_cols_available:
        by_model = _summarize_split(
            df_id, df_ood, metrics, higher_is_better,
            with_ci, n_boot, group_cols=["model"]
        )

    # By (class × model)
    by_class_model = pd.DataFrame()
    if "actual_label" in group_cols_available and "model" in group_cols_available:
        by_class_model = _summarize_split(
            df_id, df_ood, metrics, higher_is_better,
            with_ci, n_boot, group_cols=["actual_label", "model"]
        )

    return {
        "overall": overall.sort_values(["metric"]),
        "by_class": by_class.sort_values(["actual_label", "metric"]) if not by_class.empty else by_class,
        "by_model": by_model.sort_values(["model", "metric"]) if not by_model.empty else by_model,
        "by_class_model": by_class_model.sort_values(["actual_label", "model", "metric"]) if not by_class_model.empty else by_class_model,
    }


# ------------------ example usage ------------------
if __name__ == "__main__":
    # Example paths (edit these)
    ID_csv  = "/home/reshma/Otolith/captioning/otolith/caption_scores_all.csv"
    OOD_csv = "/home/reshma/Otolith/captioning/otolith/gemma_OOD_otolith_paired_captions_scores.csv"

    # If your metric columns are named like BLEU-1..4, ROUGE-L, this will auto-detect them.
    # If you want to be explicit, set metrics=[...].
    results = compute_ood_from_csv(
        ID_csv, OOD_csv,
        id_sheet=None, ood_sheet=None,           # set if you need a specific sheet name
        metrics=None,                             # auto-detect BLEU/ROUGE columns
        higher_is_better=None,                    # default: all higher-better
        with_ci=True, n_boot=2000,
        group_cols_available=None                 # auto-detect ['class','model',...]
    )

    print("\n=== Overall ===")
    print(results["overall"])

    if not results["by_class"].empty:
        print("\n=== By Class ===")
        print(results["by_class"].head())

    if not results["by_model"].empty:
        print("\n=== By Model ===")
        print(results["by_model"])

    if not results["by_class_model"].empty:
        print("\n=== By Class × Model ===")
        print(results["by_class_model"].head())
