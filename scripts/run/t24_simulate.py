"""T-24 step 1 (C3, H9; D-35 §7.6, D-42): simulated two-part omission on TEM1 training labels.

Per complete GT fiber (``astih_to_pairs``, native grid): g, eccentricity, stratum (g > 0.85 or g < 0.65 or
ecc > 0.9). Training mice only: stratum fibers deleted with p = 0.9, the rest with p = 0.07 (seed 0). A second
labelling draw on 136 random non-stratum training fibers gives r̂. Test mice keep full labels.

Outputs (git-ignored data/processed/t24/): fibers.csv; gt_pairs/<case>.png (uint16 pair map on the 2.152 grid);
labels_R1/<case>.png (semantic 0/1/2/3 on the 2.152 grid; training cases simulated, test cases full).
Committed summary: results/phase3b/t24_sim_summary.json (power check: deleted counts per kind).

    .venv/bin/python scripts/run/t24_simulate.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd
from skimage.measure import regionprops

from axon_myelin_seg.data.external import astih_to_pairs

REPO = Path(__file__).resolve().parents[2]
TEM1 = REPO / "data/external/001436"
INPUTS = REPO / "data/nnunet/phase2_inputs/tem1"
OUT = REPO / "data/processed/t24"
RES = REPO / "results/phase3b"
TEST_MICE = {f"sub-nyuMouse{n}" for n in (13, 14, 31, 32, 33, 34, 26, 27)}
P_STRATUM, P_RANDOM, R_TRUE, N_DRAW2, SEED = 0.9, 0.07, 0.93, 136, 0


def semantic(pairs: np.ndarray) -> np.ndarray:
    out = np.zeros(pairs.shape, np.uint8)
    out[(pairs >= 2) & (pairs % 2 == 0)] = 1
    out[(pairs >= 3) & (pairs % 2 == 1)] = 2
    out[pairs == 1] = 3
    return out


def fibers_of(r) -> tuple[np.ndarray, list[dict]]:
    d = TEM1 / "derivatives/labels" / r.subject / "micr"
    stem = r.source[:-4]
    pairs, _ = astih_to_pairs(iio.imread(d / f"{stem}_seg-axon-manual.png"),
                              iio.imread(d / f"{stem}_seg-myelin-manual.png"))
    fib = np.where(pairs >= 2, pairs // 2, 0)
    rows = []
    for p in regionprops(fib):
        k = p.label
        sub = pairs[p.slice]
        a, m = int((sub == 2 * k).sum()), int((sub == 2 * k + 1).sum())
        if a == 0 or m == 0:
            continue
        g = float(np.sqrt(a / (a + m)))
        ecc = float(p.eccentricity)
        rows.append({"case": r.case, "subject": r.subject, "test": r.subject in TEST_MICE, "k": int(k), "g": g,
                     "ecc": ecc, "area_px": int(a + m),
                     "stratum": bool(g > 0.85 or g < 0.65 or ecc > 0.9),
                     "s_thin": g > 0.85, "s_thick": g < 0.65, "s_oblique": ecc > 0.9})
    return pairs, rows


def one(r):
    pairs, rows = fibers_of(r)
    g2 = cv2.resize(pairs, (r.w_in, r.h_in), interpolation=cv2.INTER_NEAREST)
    iio.imwrite(OUT / "gt_pairs" / f"{r.case}.png", g2.astype(np.uint16))
    return rows


def main() -> None:
    for d in ("gt_pairs", "labels_R1"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(INPUTS / "manifest.csv")
    with Pool(6) as p:
        F = pd.DataFrame([x for rs in p.map(one, [SimpleNamespace(**x) for x in man.to_dict("records")]) for x in rs])
    rng = np.random.default_rng(SEED)
    F = F.sort_values(["case", "k"]).reset_index(drop=True)
    u = rng.random(len(F))
    tr = ~F.test
    F["deleted_stratum"] = tr & F.stratum & (u < P_STRATUM)
    F["deleted_random"] = tr & ~F.stratum & (u < P_RANDOM)
    F["deleted"] = F.deleted_stratum | F.deleted_random
    # second labelling draw on 136 random non-stratum training fibers
    pool = np.flatnonzero(tr & ~F.stratum)
    pick = rng.choice(pool, N_DRAW2, replace=False)
    F["draw2_member"] = False
    F.loc[pick, "draw2_member"] = True
    F["draw2_labelled"] = False
    F.loc[pick, "draw2_labelled"] = rng.random(N_DRAW2) < R_TRUE
    d2 = F[F.draw2_member]
    l1, l2 = ~d2.deleted, d2.draw2_labelled
    once = int((l1 | l2).sum())
    disc = int((l1 ^ l2).sum())
    pdisc = disc / once
    r_hat = float((2 - 2 * pdisc) / (2 - pdisc))  # solve pdisc = 2(1-r)/(2-r)
    F.to_csv(OUT / "fibers.csv", index=False)

    # labels: training cases simulated, test cases full
    for r in man.itertuples():
        pairs2 = iio.imread(OUT / "gt_pairs" / f"{r.case}.png")
        sem = semantic(pairs2)
        if r.subject not in TEST_MICE:
            gone = F.loc[(F.case == r.case) & F.deleted, "k"].to_numpy()
            if gone.size:
                fid = np.where(pairs2 >= 2, pairs2 // 2, 0)
                sem[np.isin(fid, gone)] = 0
        iio.imwrite(OUT / "labels_R1" / f"{r.case}.png", sem)

    t, te = F[tr], F[~tr]
    S = {"seed": SEED, "p_stratum": P_STRATUM, "p_random": P_RANDOM,
         "train": {"images": int(t.case.nunique()), "mice": int(t.subject.nunique()), "fibers": len(t),
                   "stratum": int(t.stratum.sum()), "thin": int(t.s_thin.sum()), "thick": int(t.s_thick.sum()),
                   "oblique": int(t.s_oblique.sum()), "deleted_stratum": int(t.deleted_stratum.sum()),
                   "deleted_random": int(t.deleted_random.sum()), "deleted_total": int(t.deleted.sum()),
                   "share_labels_removed": float(t.deleted.mean())},
         "test": {"images": int(te.case.nunique()), "mice": int(te.subject.nunique()), "fibers": len(te),
                  "known_positives_normal": int((~te.stratum).sum()), "stratum": int(te.stratum.sum()),
                  "thin": int(te.s_thin.sum()), "thick": int(te.s_thick.sum()), "oblique": int(te.s_oblique.sum())},
         "second_draw": {"n": N_DRAW2, "r_true": R_TRUE, "labelled_at_least_once": once, "discordant": disc,
                         "r_hat": r_hat}}
    (RES / "t24_sim_summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
