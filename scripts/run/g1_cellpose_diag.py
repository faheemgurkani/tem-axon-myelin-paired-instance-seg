"""T-30 (G1), exploratory, after the H11 verdict: what do CP-FT's missed GT fibers look like, compared with B1's?

On the 53 dev-split val images, per labelled GT fiber: equivalent-circle g, eccentricity, fiber diameter, whether
it touches GT label 1 or the image border (within 3 px), whether it lies in the protocol stratum of the benchmark
(g > 0.85 or g < 0.65 or ecc > 0.9), and whether CP-FT and B1 match it (IoU > 0.5). Also the share of CP-FT
fiber instances dropped by the containment pairing (from the inference log, if given).

Output (committed): results/paper/h11_diag.json.

    PYTHONPATH=src .venv/bin/python scripts/run/g1_cellpose_diag.py [--log <g1_dev.log>]
"""
from __future__ import annotations

import argparse
import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from skimage.measure import regionprops

from axon_myelin_seg.data.io import load_sample, read_mask
from axon_myelin_seg.evaluation.official import fiber_table, match, paired_fiber_map

REPO = Path(__file__).resolve().parents[2]


def one(name: str) -> list[dict]:
    _, gt = load_sample(name)
    preds = {m: read_mask(REPO / "data/predictions" / m / f"{name}.tif") for m in ("cellpose_ft", "b1")}
    matched = {}
    for m, p in preds.items():
        t = fiber_table(p, gt)
        matched[m] = {a for a, _, _ in match(t, 0.5)}
    g0 = np.where(gt == 1, 0, gt)
    fid = paired_fiber_map(g0)
    near1 = binary_dilation(gt == 1, iterations=3)
    border = np.zeros(gt.shape, bool)
    border[:3, :], border[-3:, :], border[:, :3], border[:, -3:] = True, True, True, True
    ax = np.where((g0 >= 2) & (g0 % 2 == 0), fid, 0)
    a_ax = np.bincount(ax.ravel(), minlength=fid.max() + 1)
    rows = []
    for p in regionprops(fid):
        k = p.label
        if a_ax[k] == 0:
            continue
        msk = fid[p.slice] == k
        g = float(np.sqrt(a_ax[k] / p.area))
        rows.append({"name": name, "k": k, "g": g, "ecc": float(p.eccentricity),
                     "d_nm": float(2 * np.sqrt(p.area / np.pi) * 2.152),
                     "near_label1": bool(near1[p.slice][msk].any()), "on_border": bool(border[p.slice][msk].any()),
                     "cp": k in matched["cellpose_ft"], "b1": k in matched["b1"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    names = json.loads((REPO / "configs/splits/dev.json").read_text())["val"]
    with Pool(6) as p:
        D = pd.DataFrame([r for rs in p.map(one, names) for r in rs])
    D["stratum"] = (D.g > 0.85) | (D.g < 0.65) | (D.ecc > 0.9)
    D["dq"] = pd.qcut(D.d_nm, 4, labels=["Q1 small", "Q2", "Q3", "Q4 large"])
    S = {"n_gt": len(D), "recall": {"cp_ft": float(D.cp.mean()), "b1": float(D.b1.mean())}}
    for col in ("stratum", "near_label1", "on_border", "dq"):
        S[f"recall_by_{col}"] = {str(k): {"n": int(len(d)), "cp_ft": float(d.cp.mean()), "b1": float(d.b1.mean())}
                                 for k, d in D.groupby(col, observed=True)}
    miss = D[~D.cp]
    S["cp_missed"] = {"n": len(miss), "share_near_label1_or_border": float((miss.near_label1 | miss.on_border).mean()),
                      "share_stratum": float(miss.stratum.mean()), "also_missed_by_b1": float((~miss.b1).mean()),
                      "median_d_nm": float(miss.d_nm.median()), "median_d_nm_all": float(D.d_nm.median())}
    S["all_share_near_label1_or_border"] = float((D.near_label1 | D.on_border).mean())
    if a.log:
        txt = Path(a.log).read_text()
        f = [(int(x), int(y)) for x, y in re.findall(r"fibers (\d+) axons \d+ paired (\d+)", txt)]
        S["pairing_dropped_share"] = 1 - sum(y for _, y in f) / max(1, sum(x for x, _ in f))
    (REPO / "results/paper/h11_diag.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
