"""TEM1 has no per-mouse diet/timepoint metadata. Estimate each mouse's state from its GT (proxy, not truth).

Design (Jelescu 2016, Fig. 2A): 33 EM mice = 9 control, 12 Cpz6 (6 at wk 6 on cuprizone, 6 at wk 12 recovered),
12 Cpz12 (6 at wk 12 on cuprizone, 6 at wk 18 recovered). TEM1 holds 20 of them; which ones is not published.
Cuprizone lowers myelinated-fiber density and thins myelin (higher g); remyelinated fibers keep thin myelin.

Per mouse (pooled over its images): myelinated fibers per 100 µm² of image, myelin volume fraction (MVF),
median and 90th-percentile per-fiber g (equivalent circle, non-border fibers), share of fibers with g > 0.85.
Then a 3-cluster k-means on standardised features as an exploratory grouping. Output: results/phase0b/tem1_mouse_proxy.csv

    .venv/bin/python scripts/run/phase0b_tem1_condition_proxy.py
"""
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from axon_myelin_seg.data.external import astih_to_pairs

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/external/001436"
PX = 0.00236


def per_image(p: Path) -> dict:
    subj = p.name.split("_")[0]
    d = ROOT / "derivatives/labels" / subj / "micr"
    stem = p.name[:-4]
    ax = iio.imread(d / f"{stem}_seg-axon-manual.png")
    my = iio.imread(d / f"{stem}_seg-myelin-manual.png")
    pairs, _ = astih_to_pairs(ax, my, border_to_ignore=True)
    fib = pairs // 2
    fib[pairs < 2] = 0
    ids, area = np.unique(fib[fib > 0], return_counts=True)
    ax_ids, ax_area = np.unique(fib[(pairs >= 2) & (pairs % 2 == 0)], return_counts=True)
    axd = dict(zip(ax_ids, ax_area))
    g = np.array([np.sqrt(axd.get(i, 0) / a) for i, a in zip(ids, area)])
    _, all_pairs = None, astih_to_pairs(ax, my)[0]  # density counts every fiber, border ones included
    n_all = len(np.unique(all_pairs[all_pairs >= 2] // 2))
    return {"subject": subj, "image": p.name, "g": g, "n_fibers": n_all,
            "area_um2": ax.size * PX * PX, "myelin_px": int((my > 0).sum()), "px": ax.size}


def main() -> None:
    rows = [per_image(p) for p in sorted(ROOT.glob("sub-*/micr/*_TEM.png"))]
    out = []
    for subj in sorted({r["subject"] for r in rows}):
        rr = [r for r in rows if r["subject"] == subj]
        g = np.concatenate([r["g"] for r in rr])
        out.append({"subject": subj, "images": len(rr), "fibers_scored_g": len(g),
                    "fibers_per_100um2": 100 * sum(r["n_fibers"] for r in rr) / sum(r["area_um2"] for r in rr),
                    "mvf": sum(r["myelin_px"] for r in rr) / sum(r["px"] for r in rr),
                    "g_median": float(np.median(g)), "g_p90": float(np.percentile(g, 90)),
                    "share_g_gt_085": float((g > 0.85).mean()),
                    "test_subject": subj == "sub-nyuMouse26"})
    df = pd.DataFrame(out)
    feats = ["fibers_per_100um2", "mvf", "g_median", "share_g_gt_085"]
    z = (df[feats] - df[feats].mean()) / df[feats].std()
    df["cluster"] = KMeans(n_clusters=3, n_init=50, random_state=0).fit_predict(z)
    # order clusters from most to least myelinated (by MVF) for readability
    order = df.groupby("cluster").mvf.mean().sort_values(ascending=False).index.tolist()
    df["cluster"] = df.cluster.map({c: i for i, c in enumerate(order)})
    df = df.sort_values(["cluster", "mvf"], ascending=[True, False])
    df.to_csv(REPO / "results/phase0b/tem1_mouse_proxy.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print(df.groupby("cluster")[feats].mean().round(3), "\n", df.cluster.value_counts().sort_index())


if __name__ == "__main__":
    main()
