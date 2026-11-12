"""Cluster bootstrap over overlap groups (D-22).

Images in an overlap group share tissue, so they are not independent samples. Resampling *groups* (with all their
images) instead of images gives honest CIs. Fiber-level F1 is recomputed from the summed TP/FP/FN of each
resample (micro-average), which is how F1 is reported everywhere in this project.
"""

import numpy as np
import pandas as pd


def micro_f1(df: pd.DataFrame, thr: str = "0.5") -> float:
    tp, fp, fn = df[f"tp@{thr}"].sum(), df[f"fp@{thr}"].sum(), df[f"fn@{thr}"].sum()
    return float(2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else float("nan")


def cluster_bootstrap(df: pd.DataFrame, stat, group_col: str = "group", n: int = 1000, seed: int = 0) -> dict:
    """``stat(df_resampled) -> float``. Returns point estimate, 95% percentile CI and half-width."""
    rng = np.random.default_rng(seed)
    groups = df[group_col].unique()
    idx = {g: np.flatnonzero(df[group_col].to_numpy() == g) for g in groups}
    vals = []
    for _ in range(n):
        pick = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        vals.append(stat(df.iloc[rows]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return {"est": float(stat(df)), "lo": float(lo), "hi": float(hi), "half_width": float((hi - lo) / 2)}


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, stat, group_col: str = "group", n: int = 1000,
                     seed: int = 0) -> dict:
    """CI of stat(b) − stat(a) with the *same* resampled groups for both (rows aligned by image name)."""
    a = a.set_index("name").sort_index()
    b = b.set_index("name").loc[a.index]
    rng = np.random.default_rng(seed)
    groups = a[group_col].unique()
    pos = {g: np.flatnonzero(a[group_col].to_numpy() == g) for g in groups}
    diffs = []
    for _ in range(n):
        pick = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([pos[g] for g in pick])
        diffs.append(stat(b.iloc[rows]) - stat(a.iloc[rows]))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"diff": float(stat(b) - stat(a)), "lo": float(lo), "hi": float(hi)}
