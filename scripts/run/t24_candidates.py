"""T-24 step 2 (C3; D-42): unlabelled-fiber candidates on the simulated TEM1 training labels, with features and t_k.

Per training case, on the 2.152 nm grid:
- B1 zero-shot instances (D-11 pairing of data/predictions/b1_ext/tem1/<case>.png) overlapping no remaining labelled
  fiber by >= 50% of their area, plus D-18 ring-candidate holes (`ring_candidates`, ds 4) on the simulated labels.
  A hole inside a B1 candidate is covered by it; a hole with none is dilated by 0.43 x its equivalent radius.
- Core footprint -> features from B1's semantics inside it (same pipeline for every candidate): equivalent g,
  outer diameter, myelin thickness, eccentricity, solidity, ring contrast, B1 support; t_k = mean B1 fiber
  probability (1 - p_bg; data/predictions/b1_ext_prob/tem1/<case>.npz) over the core.
- Truth from the complete GT (gt_pairs): best-IoU GT fiber >= 0.5 -> deleted_random ("missed normal"),
  deleted_stratum, labelled (still labelled), else not_fiber.
- Painted footprint = core dilated 2 px into simulated background only (later weights/ignore apply there).

Outputs (git-ignored data/processed/t24/): cands/<case>.png (uint16 painted ids), candidates.csv.

    .venv/bin/python scripts/run/t24_candidates.py [--cases tem1_0000 ...]
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops

from axon_myelin_seg.evaluation.taxonomy import ring_candidates
from axon_myelin_seg.postprocessing.pairing import semantic_to_paired_instances

REPO = Path(__file__).resolve().parents[2]
T24 = REPO / "data/processed/t24"
B1SEM = REPO / "data/predictions/b1_ext/tem1"
B1PROB = REPO / "data/predictions/b1_ext_prob/tem1"
IMG = REPO / "data/nnunet/phase2_inputs/tem1"
B8 = np.ones((3, 3), bool)


def footprint_features(p, s: np.ndarray, iv: np.ndarray, px_nm: float = 2.152) -> dict:
    """Same-pipeline fiber features (D-42) from a footprint's regionprops ``p`` and B1's semantics ``s`` / image
    values ``iv`` at its pixels. Used for candidates and, at scoring, for test GT fibers (hard-normal tercile)."""
    a, my = int((s == 1).sum()), int((s == 2).sum())
    ia, im = iv[s == 1], iv[s == 2]
    r_out, r_in = np.sqrt(p.area / np.pi), np.sqrt(a / np.pi)
    return {"g_eq": float(np.sqrt(a / (a + my))) if a and my else np.nan, "d_out_nm": 2 * r_out * px_nm,
            "myelin_nm": (r_out - r_in) * px_nm, "ecc": float(p.eccentricity), "solidity": float(p.solidity),
            "contrast": float(ia.mean() - im.mean()) if ia.size and im.size else np.nan,
            "b1_support": float((s > 0).mean())}


def one(case: str) -> list[dict]:
    lab = iio.imread(T24 / "labels_R1" / f"{case}.png")
    gtp = iio.imread(T24 / "gt_pairs" / f"{case}.png").astype(np.int64)
    img = iio.imread(IMG / f"{case}_0000.png")
    img = img if img.ndim == 2 else img[..., 0]
    sem = iio.imread(B1SEM / f"{case}.png")
    pfg = 1.0 - np.load(B1PROB / f"{case}.npz")["probabilities"][0, 0]
    assert lab.shape == gtp.shape == img.shape == sem.shape == pfg.shape, case
    F = pd.read_csv(T24 / "fibers.csv")
    F = F[F.case == case].set_index("k")
    labelled = (lab == 1) | (lab == 2)
    bg = lab == 0

    # simulated-label pair map (deleted fibers removed) for the D-18 detector's threshold and labelled filter
    gone = F.index[F.deleted].to_numpy()
    fid = np.where(gtp >= 2, gtp // 2, 0)
    sim_pairs = np.where(np.isin(fid, gone), 0, gtp)

    # (i) B1 instances not mostly on labelled fibers
    P = semantic_to_paired_instances(sem == 1, sem == 2).astype(np.int64)
    pf = np.where(P >= 2, P // 2, 0)
    tot = np.bincount(pf.ravel())
    onlab = np.bincount(pf[labelled].ravel(), minlength=tot.size)
    cand_ids = [k for k in range(1, tot.size) if tot[k] > 0 and onlab[k] < 0.5 * tot[k]]
    core = np.zeros(lab.shape, np.int32)
    src = {}
    for n, k in enumerate(cand_ids, start=1):
        core[pf == k] = n
        src[n] = "b1"
    # (ii) D-18 holes not covered by a B1 candidate
    holes = ring_candidates(img, sim_pairs, ds=4, px_nm=2.152)
    n = len(cand_ids)
    for p in regionprops(holes):
        hm = holes[p.slice] == p.label
        if (core[p.slice][hm] > 0).mean() > 0.5:
            continue
        r_eq = np.sqrt(p.area / np.pi)
        pad = int(np.ceil(0.43 * r_eq)) + 1
        y0, x0 = max(p.slice[0].start - pad, 0), max(p.slice[1].start - pad, 0)
        y1, x1 = min(p.slice[0].stop + pad, lab.shape[0]), min(p.slice[1].stop + pad, lab.shape[1])
        h = holes[y0:y1, x0:x1] == p.label
        d = ndi.binary_dilation(h, B8, iterations=max(round(0.43 * r_eq), 1))
        win = core[y0:y1, x0:x1]
        put = d & (win == 0) & ~labelled[y0:y1, x0:x1]
        if put.sum() < 50:
            continue
        n += 1
        win[put] = n
        src[n] = "d18"

    # painted footprint: core dilated 2 px into background only; first id wins on overlap
    painted = np.where(bg, core, 0)
    grow = ndi.grey_dilation(painted, footprint=B8, size=None)
    grow = ndi.grey_dilation(grow, footprint=B8)
    painted = np.where((painted == 0) & bg & (grow > 0), grow, painted)
    iio.imwrite(T24 / "cands" / f"{case}.png", painted.astype(np.uint16))

    gt_f = np.where(gtp >= 2, gtp // 2, 0)
    gt_area = np.bincount(gt_f.ravel())
    rows = []
    for p in regionprops(core, intensity_image=pfg):
        k = p.label
        sl = p.slice
        m = core[sl] == k
        feats = footprint_features(p, sem[sl][m], img[sl][m])
        # truth: best IoU GT fiber
        ov = np.bincount(gt_f[sl][m].ravel(), minlength=gt_area.size)
        ov[0] = 0
        kg = int(ov.argmax())
        iou = ov[kg] / (p.area + gt_area[kg] - ov[kg]) if kg else 0.0
        if iou >= 0.5 and kg in F.index:
            f = F.loc[kg]
            truth = ("deleted_stratum" if f.deleted_stratum else "deleted_random" if f.deleted_random
                     else "labelled")
            g_true, stratum = float(f.g), bool(f.stratum)
        else:
            truth, g_true, stratum = "not_fiber", np.nan, False
        rows.append({"case": case, "cid": k, "source": src[k], "area_px": int(p.area),
                     "t_raw": float(p.intensity_mean), **feats, "gt_k": kg if iou >= 0.5 else -1, "iou": float(iou),
                     "truth": truth, "stratum_true": stratum, "g_true": g_true,
                     "painted_px": int((painted == k).sum())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    (T24 / "cands").mkdir(parents=True, exist_ok=True)
    cases = a.cases or sorted(pd.read_csv(T24 / "fibers.csv").query("~test").case.unique())
    missing = [c for c in cases if not (B1PROB / f"{c}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} probability files missing, e.g. {missing[:3]}")
    with Pool(a.workers) as p:
        rows = [r for rs in p.map(one, cases) for r in rs]
    df = pd.DataFrame(rows)
    out = T24 / ("candidates.csv" if not a.cases else "candidates_subset.csv")
    df.to_csv(out, index=False)
    print(df.groupby(["source", "truth"]).size().unstack(fill_value=0))
    print(df.groupby("truth")[["t_raw", "g_eq", "ecc"]].median().round(3))


if __name__ == "__main__":
    main()
