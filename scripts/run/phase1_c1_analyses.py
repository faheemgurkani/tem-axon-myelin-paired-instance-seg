"""Phase 1, T-16′: the remaining C1 analyses on the 53 dev-val images, for B1, Cellpose+B1 and AxonPath.

1. g-definition sensitivity: equivalent circle (primary, challenge), minor axis, max Feret, perimeter radius.
2. Recall-vs-g-MAE gaming curve: drop matched fibers (a) smallest predicted fiber first (a realistic "conservative
   model"), (b) worst |g error| first (adversarial bound).
3. PQ-style g-quality = F1@0.5 × mean_k (1 − min(|Δg_k| / s, 1)), s = 0.10 (primary; ≈ the AimSeg limits of
   agreement and the disease-effect range) and s = 0.05 (sensitivity). An adaptation of PQ (Kirillov 2019).
4. Bland–Altman of D_A, D_F (equivalent diameters, nm) and g; g-MAE and recall by GT *fiber* diameter quartile.
5. With and without oblique fibers (GT fiber eccentricity > 0.9).
6. HD95 variants per image: official (max of directed P95), mean of directed P95, and HD100 (max).
7. Rank stability of the three models across all switches.
CIs: cluster bootstrap over overlap groups (D-22). Per-fiber tables go to data/processed/phase1/ (git-ignored: per-fiber
GT derivatives); the summary to results/phase1/c1_analyses.json.

    .venv/bin/python scripts/run/phase1_c1_analyses.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops

from axon_myelin_seg.data.io import load_sample, read_mask
from axon_myelin_seg.evaluation.official import (
    PIXEL_NM,
    _boundary,
    apply_ignore,
    fiber_table,
    match,
    paired_fiber_map,
)
from axon_myelin_seg.evaluation.stats import cluster_bootstrap

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/phase1"
PROC = REPO / "data/processed/phase1"
MODELS = ["b1", "cellpose_b1", "axonpath"]
DEFS = ["eq", "minor", "feret", "perim"]


def _props(lab: np.ndarray, kind: str) -> dict[int, dict]:
    """Per-fiber geometry. kind='fiber' uses X ∪ X+1, kind='axon' uses even labels only."""
    m = paired_fiber_map(lab).copy()
    if kind == "axon":
        m[~((lab >= 2) & (lab % 2 == 0))] = 0
    out = {}
    for r in regionprops(m):
        out[int(r.label)] = {"area": r.area, "eq": r.equivalent_diameter_area, "minor": r.axis_minor_length,
                             "feret": r.feret_diameter_max, "perim": r.perimeter / np.pi,
                             "ecc": r.eccentricity}
    return out


def _hd_variants(p: np.ndarray, g: np.ndarray) -> dict:
    if not p.any() or not g.any():
        return {"official": np.nan, "mean_p95": np.nan, "hd100": np.nan}
    bp, bg = _boundary(p), _boundary(g)
    d_to_g = ndi.distance_transform_edt(~bg)[bp] * PIXEL_NM
    d_to_p = ndi.distance_transform_edt(~bp)[bg] * PIXEL_NM
    a, b = np.percentile(d_to_g, 95), np.percentile(d_to_p, 95)
    return {"official": max(a, b), "mean_p95": (a + b) / 2, "hd100": max(d_to_g.max(), d_to_p.max())}


def one(args):
    model, name = args
    _, gt = load_sample(name)
    pred = read_mask(REPO / "data/predictions" / model / f"{name}.tif")
    t = fiber_table(pred, gt)
    m = match(t, 0.5)
    p_ig, g_ig = apply_ignore(pred, gt)
    pf, pa = _props(p_ig, "fiber"), _props(p_ig, "axon")
    gf, ga = _props(g_ig, "fiber"), _props(g_ig, "axon")
    matched_gt = {kg for kg, _, _ in m}
    fib = []
    for kg, kp, iou in m:
        row = {"image": name, "gt_k": kg, "pred_k": kp, "iou": iou,
               "gt_DF_nm": gf[kg]["eq"] * PIXEL_NM, "pred_DF_nm": pf[kp]["eq"] * PIXEL_NM,
               "gt_DA_nm": ga.get(kg, {}).get("eq", 0) * PIXEL_NM, "pred_DA_nm": pa.get(kp, {}).get("eq", 0) * PIXEL_NM,
               "gt_ecc": gf[kg]["ecc"], "pred_area": pf[kp]["area"]}
        for d in DEFS:
            gd, pd_ = gf[kg][d], pf[kp][d]
            row[f"g_gt_{d}"] = ga.get(kg, {}).get(d, np.nan) / gd if gd else np.nan
            row[f"g_pred_{d}"] = pa.get(kp, {}).get(d, np.nan) / pd_ if pd_ else np.nan
        fib.append(row)
    gts = [{"image": name, "gt_k": k, "gt_DF_nm": v["eq"] * PIXEL_NM, "gt_ecc": v["ecc"], "matched": k in matched_gt}
           for k, v in gf.items() if k in ga]  # GT fibers with an axon (orphans can't match)
    img = {"image": name, "tp": len(m), "fp": len(t.pred_ids) - len(m), "fn": len(t.gt_ids) - len(m)}
    for comp, cond in (("axon", lambda x: (x >= 2) & (x % 2 == 0)), ("myelin", lambda x: (x >= 3) & (x % 2 == 1))):
        for k, v in _hd_variants(cond(p_ig), cond(g_ig)).items():
            img[f"hd_{comp}_{k}"] = v
    return model, fib, gts, img


def gq(f: pd.DataFrame, f1: float, s: float) -> float:
    err = (f.g_pred_eq - f.g_gt_eq).abs()
    return float(f1 * (1 - np.minimum(err / s, 1)).mean()) if len(f) else np.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    names = json.loads((REPO / "configs/splits/dev.json").read_text())["val"]
    groups = pd.read_csv(REPO / "results/day4/scores_b1.csv")[["name", "group"]].rename(columns={"name": "image"})
    with Pool(6) as pool:
        res = pool.map(one, [(m, n) for m in MODELS for n in names])
    S = {}
    curves = []
    for model in MODELS:
        fib = pd.DataFrame([r for mm, f, _, _ in res if mm == model for r in f])
        gts = pd.DataFrame([r for mm, _, g, _ in res if mm == model for r in g])
        img = pd.DataFrame([i for mm, _, _, i in res if mm == model])
        for d in (fib, gts, img):
            d["group"] = d.image.map(dict(zip(groups.image, groups.group)))
        fib.to_csv(PROC / f"c1_fibers_{model}.csv", index=False)
        gts.to_csv(PROC / f"c1_gtfibers_{model}.csv", index=False)
        img.to_csv(OUT / f"c1_images_{model}.csv", index=False)
        tp, fp, fn = img.tp.sum(), img.fp.sum(), img.fn.sum()
        f1, rec = 2 * tp / (2 * tp + fp + fn), tp / (tp + fn)
        s = {"f1@0.5": f1, "recall": rec, "n_matched": len(fib)}
        # 1. g definitions
        s["g_mae"] = {d: float((fib[f"g_pred_{d}"] - fib[f"g_gt_{d}"]).abs().mean()) for d in DEFS}
        s["g_bias"] = {d: float((fib[f"g_pred_{d}"] - fib[f"g_gt_{d}"]).mean()) for d in DEFS}
        s["g_mean_gt"] = {d: float(fib[f"g_gt_{d}"].mean()) for d in DEFS}
        s["g_mae_eq_ci"] = cluster_bootstrap(fib, lambda d: (d.g_pred_eq - d.g_gt_eq).abs().mean())
        # 3. g-quality
        s["g_quality"] = {str(sc): gq(fib, f1, sc) for sc in (0.10, 0.05)}
        # 4. Bland–Altman and fiber-diameter bins
        ba = {}
        for q, (a, b) in {"D_F": ("pred_DF_nm", "gt_DF_nm"), "D_A": ("pred_DA_nm", "gt_DA_nm"),
                          "g": ("g_pred_eq", "g_gt_eq")}.items():
            d = fib[a] - fib[b]
            ba[q] = {"bias": float(d.mean()), "loa_lo": float(d.mean() - 1.96 * d.std()),
                     "loa_hi": float(d.mean() + 1.96 * d.std())}
        s["bland_altman"] = ba
        qs = np.quantile(gts.gt_DF_nm, [0, .25, .5, .75, 1])
        gts["bin"] = pd.cut(gts.gt_DF_nm, qs, include_lowest=True)
        fib["bin"] = pd.cut(fib.gt_DF_nm, qs, include_lowest=True)
        s["by_fiber_diameter_quartile"] = {
            str(b): {"recall": float(gts[gts.bin == b].matched.mean()),
                     "g_mae": float((fib[fib.bin == b].g_pred_eq - fib[fib.bin == b].g_gt_eq).abs().mean())}
            for b in gts.bin.cat.categories}
        # 5. oblique
        ob_g, ob_f = gts.gt_ecc > 0.9, fib.gt_ecc > 0.9
        s["oblique"] = {"share_gt_oblique": float(ob_g.mean()),
                        "recall_without": float(gts[~ob_g].matched.mean()),
                        "g_mae_without": float((fib[~ob_f].g_pred_eq - fib[~ob_f].g_gt_eq).abs().mean()),
                        "g_mae_oblique_only": float((fib[ob_f].g_pred_eq - fib[ob_f].g_gt_eq).abs().mean())}
        # 6. HD95 variants
        s["hd_nm_mean"] = {c: float(img[c].mean()) for c in img.columns if c.startswith("hd_")}
        # 2. gaming curves
        n_gt = tp + fn
        err = (fib.g_pred_eq - fib.g_gt_eq).abs().to_numpy()
        for how, order in (("smallest_first", np.argsort(fib.pred_area.to_numpy())),
                           ("worst_err_first", np.argsort(-err))):
            for frac in np.linspace(0, 0.5, 11):
                keep = order[round(frac * len(order)):]
                curves.append({"model": model, "drop": how, "dropped_frac": frac,
                               "recall": len(keep) / n_gt, "g_mae": float(err[keep].mean())})
        S[model] = s
    # 7. rank stability (lower is better except f1, recall, g_quality)
    metrics = {f"g_mae_{d}": (lambda m, d=d: S[m]["g_mae"][d], False) for d in DEFS}
    metrics |= {f"hd_myelin_{v}": (lambda m, v=v: S[m]["hd_nm_mean"][f"hd_myelin_{v}"], False)
                for v in ("official", "mean_p95", "hd100")}
    metrics |= {"f1@0.5": (lambda m: S[m]["f1@0.5"], True), "recall": (lambda m: S[m]["recall"], True),
                "g_quality_0.1": (lambda m: S[m]["g_quality"]["0.1"], True),
                "g_mae_no_oblique": (lambda m: S[m]["oblique"]["g_mae_without"], False)}
    S["ranks"] = {k: [m for m, _ in sorted(((m, f(m)) for m in MODELS), key=lambda x: x[1], reverse=hi)]
                  for k, (f, hi) in metrics.items()}
    pd.DataFrame(curves).to_csv(OUT / "c1_gaming_curves.csv", index=False)
    (OUT / "c1_analyses.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps({m: {k: S[m][k] for k in ("f1@0.5", "g_mae", "g_quality", "oblique", "hd_nm_mean")}
                      for m in MODELS}, indent=1, default=float))
    print(json.dumps(S["ranks"], indent=1))


if __name__ == "__main__":
    main()
