"""Phase 1, T-25 (D-37): image fingerprint, an ANALYSIS-ONLY proxy for the missing condition metadata (A2).

Never a training input or network condition. Described as a proxy, not condition truth, until A2 is answered.

1. Tissue fingerprint per image, from LABELLED fibers with one pipeline for every dataset (physical units):
   fiber density (/100 µm², all fibers), myelin volume fraction, median / p90 g and share g > 0.85 (equivalent circle,
   non-border fibers), median fiber diameter (nm).
   Deliberately EXCLUDED: the unlabelled-candidate rate. It is H7's outcome, so clustering on it would make the H7
   stratification circular (16 Nov, logged in 07-plan §6). Acquisition features (myelin/background contrast, noise,
   sharpness) are kept apart and only described per cluster, to see whether clusters are acquisition, not tissue.
2. Cluster our 266 images (k-means on z-scores, k = 2..5 chosen by silhouette); stability = adjusted Rand index of a
   refit on overlap-group bootstrap resamples against the full fit. TEM1: the same features per image, averaged per
   mouse, clustered with k = 3; sanity check = ARI against the Phase 0b proxy clusters (results/phase0b/tem1_mouse_proxy.csv).
   Exploratory: TEM1 mice assigned to the nearest of OUR cluster centres (our standardisation).
3. H7 stratification (provisional, like H7 itself): per cluster, unlabelled candidates per 100 µm² background and per
   labelled fiber, and the provisional H7b count bias (real share 0.83, detector recall 0.65, as in
   phase1_h7_protocol_bias.py); overlap-group bootstrap CIs; rate ratios vs the largest cluster.
H8 and the H6 limitations use the saved cluster labels in Phase 2.

Outputs: results/phase1/t25_fingerprint_ours.csv, t25_fingerprint_tem1.csv, t25_summary.json.

    .venv/bin/python scripts/run/phase1_t25_fingerprint.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from axon_myelin_seg.data.external import astih_to_pairs

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/phase1"
TEM1 = REPO / "data/external/001436"
PX_OURS, PX_TEM1 = 2.152, 2.36  # nm / px
TISSUE = ["density_100um2", "mvf", "g_median", "g_p90", "share_g_085", "dfiber_median_nm"]
ACQ = ["contrast", "noise_sigma", "lap_var"]
REAL_SHARE, DET_RECALL = 0.83, 0.65  # provisional, as H7 (phase1_h7_protocol_bias.py)


def _summ(g: np.ndarray, dfib: np.ndarray) -> dict:
    return {"g_median": float(np.median(g)), "g_p90": float(np.percentile(g, 90)),
            "share_g_085": float((g > 0.85).mean()), "dfiber_median_nm": float(np.median(dfib))}


def ours() -> pd.DataFrame:
    f = pd.read_csv(REPO / "data/processed/eda/fibers.csv")
    im = pd.read_csv(REPO / "data/processed/eda/images.csv")
    deep = pd.read_csv(REPO / "data/processed/eda/deep_images.csv")
    comp = pd.read_csv(REPO / "results/phase0b/completeness_ours.csv")
    rows = []
    for name, d in f[~f.border].groupby("file"):
        rows.append({"image": name.removesuffix(".tif"), **_summ(d.g.to_numpy(), d.d_fiber_nm.to_numpy())})
    df = pd.DataFrame(rows)
    im["image"] = im.file.str.removesuffix(".tif")
    im["contrast"] = (im.int_bg - im.int_myelin) / im.int_std
    deep["image"] = deep.file.str.removesuffix(".tif")
    df = df.merge(im[["image", "frac_myelin", "contrast"]], on="image").merge(
        deep[["image", "noise_sigma", "lap_var"]], on="image").merge(comp, on="image")
    df["mvf"] = df.frac_myelin
    df["density_100um2"] = 100 * df.n_fibers / df.img_um2
    groups = json.loads((REPO / "configs/splits/overlap_groups.json").read_text())
    gmap = {m.removesuffix(".tif"): f"g{i}" for i, ms in enumerate(groups) for m in ms}
    df["group"] = df.image.map(lambda n: gmap.get(n, n))
    return df


def _tem1_image(p: Path) -> dict:
    subj = p.name.split("_")[0]
    d = TEM1 / "derivatives/labels" / subj / "micr"
    stem = p.name[:-4]
    ax = iio.imread(d / f"{stem}_seg-axon-manual.png")
    my = iio.imread(d / f"{stem}_seg-myelin-manual.png")
    pairs, _ = astih_to_pairs(ax, my, border_to_ignore=True)
    fib = np.where(pairs >= 2, pairs // 2, 0)
    ids, area = np.unique(fib[fib > 0], return_counts=True)
    axd = dict(zip(*np.unique(fib[(pairs >= 2) & (pairs % 2 == 0)], return_counts=True)))
    g = np.array([np.sqrt(axd.get(i, 0) / a) for i, a in zip(ids, area)])
    dfib = 2 * np.sqrt(area / np.pi) * PX_TEM1
    all_pairs = astih_to_pairs(ax, my)[0]
    img = iio.imread(p).astype(np.float32)
    img = img[..., 0] if img.ndim == 3 else img
    myl, bg = my > 0, (ax == 0) & (my == 0)
    return {"subject": subj, "image": p.name, "g": g, "dfib": dfib,
            "n_all": len(np.unique(all_pairs[all_pairs >= 2] // 2)), "um2": ax.size * (PX_TEM1 / 1000) ** 2,
            "myelin_px": int(myl.sum()), "px": ax.size,
            "contrast": float((np.median(img[bg]) - np.median(img[myl])) / img.std()) if myl.any() else np.nan}


def tem1() -> pd.DataFrame:
    with Pool(6) as p:
        rows = p.map(_tem1_image, sorted(TEM1.glob("sub-*/micr/*_TEM.png")))
    out = []
    for subj in sorted({r["subject"] for r in rows}):
        rr = [r for r in rows if r["subject"] == subj]
        g = np.concatenate([r["g"] for r in rr])
        dfib = np.concatenate([r["dfib"] for r in rr])
        out.append({"subject": subj, "images": len(rr), **_summ(g, dfib),
                    "density_100um2": 100 * sum(r["n_all"] for r in rr) / sum(r["um2"] for r in rr),
                    "mvf": sum(r["myelin_px"] for r in rr) / sum(r["px"] for r in rr),
                    "contrast": float(np.nanmean([r["contrast"] for r in rr]))})
    return pd.DataFrame(out)


def _cluster(z: np.ndarray, k: int, seed: int = 0) -> KMeans:
    return KMeans(n_clusters=k, n_init=50, random_state=seed).fit(z)


def _order(df: pd.DataFrame, lab: np.ndarray) -> np.ndarray:
    """Relabel clusters from most to least myelinated (MVF), so labels read the same across fits."""
    order = pd.Series(df.mvf.to_numpy()).groupby(lab).mean().sort_values(ascending=False).index.tolist()
    return np.array([order.index(x) for x in lab])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    S: dict = {"note": "T-25 fingerprint: analysis-only proxy (D-37); not condition truth until A2"}
    # --- ours
    df = ours()
    mu, sd = df[TISSUE].mean(), df[TISSUE].std()
    z = ((df[TISSUE] - mu) / sd).to_numpy()
    sil = {k: float(silhouette_score(z, _cluster(z, k).labels_)) for k in range(2, 6)}
    k = max(sil, key=sil.get)
    km = _cluster(z, k)
    df["cluster"] = _order(df, km.labels_)
    grp = sorted(set(df.group))
    aris = []
    for b in range(200):
        pick = rng.choice(grp, len(grp))
        idx = np.concatenate([np.flatnonzero(df.group.to_numpy() == x) for x in pick])
        kb = _cluster(z[idx], k, seed=b)
        aris.append(adjusted_rand_score(km.labels_, kb.predict(z)))
    S["ours"] = {"n_images": len(df), "silhouette": sil, "k": k,
                 "stability_ari_median": float(np.median(aris)), "stability_ari_p05": float(np.percentile(aris, 5)),
                 "cluster_sizes": df.cluster.value_counts().sort_index().to_dict(),
                 "cluster_means": df.groupby("cluster")[TISSUE + ACQ].mean().round(4).to_dict(orient="index")}
    # --- H7 stratification (provisional)

    def h7(d: pd.DataFrame) -> dict:
        c, n, bg = d.n_candidates.sum(), d.n_fibers.sum(), d.bg_um2.sum()
        real = REAL_SHARE * c / DET_RECALL
        return {"cand_per_100um2_bg": 100 * c / bg, "cand_per_labelled": c / n, "h7b_count_bias": real / (n + real)}

    ref = int(df.cluster.value_counts().idxmax())
    est = {c: h7(d) for c, d in df.groupby("cluster")}
    boots = {c: [] for c in est}
    rr = {c: [] for c in est if c != ref}
    for _ in range(1000):
        pick = rng.choice(grp, len(grp))
        bd = pd.concat([df[df.group == x] for x in pick])
        hb = {c: h7(d) for c, d in bd.groupby("cluster")}
        for c, lst in boots.items():
            if c in hb:
                lst.append(hb[c])
        for c, lst in rr.items():
            if c in hb and ref in hb and hb[ref]["cand_per_100um2_bg"] > 0:
                lst.append(hb[c]["cand_per_100um2_bg"] / hb[ref]["cand_per_100um2_bg"])
    S["h7_by_cluster"] = {
        str(c): {**{m: float(v) for m, v in est[c].items()},
                 **{f"{m}_ci95": [float(np.percentile([b[m] for b in boots[c]], q)) for q in (2.5, 97.5)]
                    for m in est[c]},
                 "n_images": int((df.cluster == c).sum()),
                 **({"rate_ratio_vs_ref": float(est[c]["cand_per_100um2_bg"] / est[ref]["cand_per_100um2_bg"]),
                     "rate_ratio_ci95": [float(np.percentile(rr[c], q)) for q in (2.5, 97.5)]} if c != ref else
                    {"reference": True})}
        for c in est}
    # --- TEM1
    t = tem1()
    zt = ((t[TISSUE] - t[TISSUE].mean()) / t[TISSUE].std()).to_numpy()
    t["cluster_t25"] = _order(t, _cluster(zt, 3).labels_)
    proxy = pd.read_csv(REPO / "results/phase0b/tem1_mouse_proxy.csv")[["subject", "cluster"]]
    t = t.merge(proxy.rename(columns={"cluster": "cluster_proxy"}), on="subject")
    zo = ((t[TISSUE] - mu) / sd).to_numpy()  # our standardisation -> our centres (exploratory)
    t["nearest_ours"] = [int(df.cluster[km.labels_ == km.predict(zo[i:i + 1])[0]].iloc[0]) for i in range(len(t))]
    S["tem1"] = {"n_mice": len(t), "ari_vs_phase0b_proxy": float(adjusted_rand_score(t.cluster_proxy, t.cluster_t25)),
                 "crosstab_t25_vs_proxy": pd.crosstab(t.cluster_t25, t.cluster_proxy).to_dict(),
                 "nearest_ours_counts": t.nearest_ours.value_counts().sort_index().to_dict(),
                 "test_mouse26": t[t.subject == "sub-nyuMouse26"][["cluster_t25", "cluster_proxy",
                                                                     "nearest_ours"]].to_dict(orient="records")}
    keep = ["image", "group", "cluster", *TISSUE, *ACQ, "n_candidates", "n_fibers", "bg_um2"]
    df[keep].to_csv(OUT / "t25_fingerprint_ours.csv", index=False)
    t.to_csv(OUT / "t25_fingerprint_tem1.csv", index=False)
    (OUT / "t25_summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=float))
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
