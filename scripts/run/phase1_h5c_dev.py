"""Phase 1, H5c development (DEV IMAGES ONLY; D-29). Candidate post-processing rules that regroup B1's own semantic
pixels into fibers, scored against the pre-registered H5c targets: capture ≥ 50% of O-sep's g-MAE gain
(base 0.0160 → O-sep 0.0093 on dev) with ΔF1@0.5 ≥ −0.005 and Δrecall ≥ −0.005.

Variants (all keep B1's pixels; only V3 adds pixels):
  V0  current D-11 rule: watershed on −EDT(axon ∪ myelin), seeds = axon components.
  V1  nearest-axon: watershed on EDT(¬axon) (distance from axons) inside axon ∪ myelin.
  V2  axomyelin-unit prior: each foreground pixel goes to the axon minimising distance / axon equivalent radius
      (myelin thickness ∝ axon radius when g is near-constant; Gow 2025–26), within its connected component.
  V3  V2 + per-fiber ring closing (2 px) adding only background pixels inside that fiber's convex region.

Development only: the chosen rule is frozen in `postprocessing/pairing.py` before T-21 predictions are scored.

    .venv/bin/python scripts/run/phase1_h5c_dev.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from axon_myelin_seg.data.io import load_sample
from axon_myelin_seg.evaluation.official import fiber_table, g_ratio, match
from axon_myelin_seg.postprocessing.pairing import B8, semantic_to_paired_instances

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/phase1"


def _encode(axon_k: np.ndarray, my_k: np.ndarray, n: int) -> np.ndarray:
    has_ring = np.bincount(my_k.ravel(), minlength=n + 1) > 0
    has_axon = np.bincount(axon_k.ravel(), minlength=n + 1) > 0
    keep = has_ring & has_axon
    keep[0] = False
    new = np.zeros(n + 1, np.int64)
    new[keep] = np.arange(1, int(keep.sum()) + 1)
    out = np.zeros(axon_k.shape, np.uint16)
    a, m = new[axon_k], new[my_k]
    out[m > 0] = (2 * m[m > 0] + 1).astype(np.uint16)
    out[a > 0] = (2 * a[a > 0]).astype(np.uint16)
    return out


def _seeds(axon: np.ndarray, min_axon_px: int = 50):
    sm = ndi.binary_fill_holes(axon)
    seeds, n = ndi.label(sm, structure=B8)
    sizes = np.bincount(seeds.ravel())
    small = sizes < min_axon_px
    small[0] = False
    seeds[small[seeds]] = 0
    return sm, seeds, n


def v1(axon, myelin):
    myelin = myelin & ~axon
    sm, seeds, n = _seeds(axon)
    fg = sm | myelin
    regions = watershed(ndi.distance_transform_edt(seeds == 0), markers=seeds, mask=fg)
    return _encode(seeds, np.where(myelin & (seeds == 0), regions, 0), n)


def v2(axon, myelin, close_px: int = 0):
    myelin = myelin & ~axon
    sm, seeds, n = _seeds(axon)
    fg = sm | myelin
    comp, _ = ndi.label(fg, structure=B8)
    best = np.full(axon.shape, np.inf)
    lab = np.zeros(axon.shape, np.int32)
    for k, sl in enumerate(ndi.find_objects(seeds), start=1):
        if sl is None:
            continue
        m = seeds[sl] == k
        r = np.sqrt(m.sum() / np.pi)
        pad = int(np.ceil(4 * r)) + 5  # myelin reach: g ≥ 0.2 within 4 r
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, axon.shape[0])
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, axon.shape[1])
        win = (slice(y0, y1), slice(x0, x1))
        sk = seeds[win] == k
        cid = np.unique(comp[win][sk])
        same = np.isin(comp[win], cid[cid > 0])
        d = ndi.distance_transform_edt(~sk) / max(r, 1.0)
        upd = same & (d < best[win])
        best[win][upd] = d[upd]
        lab[win][upd] = k
    my_k = np.where(myelin & (seeds == 0), lab, 0)
    if close_px:
        bg = ~fg
        for k, sl in enumerate(ndi.find_objects(my_k), start=1):
            if sl is None:
                continue
            ys = slice(max(sl[0].start - close_px, 0), sl[0].stop + close_px)
            xs = slice(max(sl[1].start - close_px, 0), sl[1].stop + close_px)
            ring = my_k[ys, xs] == k
            closed = ndi.binary_closing(ring, structure=B8, iterations=close_px)
            add = closed & ~ring & bg[ys, xs] & (lab[ys, xs] == k)
            my_k[ys, xs][add] = k
    return _encode(seeds, my_k, n)


VARIANTS = {
    "V0": lambda a, m: semantic_to_paired_instances(a, m),
    "V1": v1,
    "V2": lambda a, m: v2(a, m),
    "V3": lambda a, m: v2(a, m, close_px=2),
}


def one(name):
    _, gt = load_sample(name)
    sem = iio.imread(REPO / "data/predictions/b1_sem" / f"{name}.png")
    rows = []
    for v, f in VARIANTS.items():
        pred = f(sem == 1, sem == 2)
        t = fiber_table(pred, gt)
        m = match(t, 0.5)
        err = [abs(g_ratio(t.pred_axon_area[kp], t.pred_area[kp]) - g_ratio(t.gt_axon_area[kg], t.gt_area[kg]))
               for kg, kp, _ in m]
        rows.append({"name": name, "variant": v, "tp": len(m), "fp": len(t.pred_ids) - len(m),
                     "fn": len(t.gt_ids) - len(m), "g_abs_err_sum": float(np.sum(err)), "n_matched": len(m)})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = json.loads((REPO / "configs/splits/dev.json").read_text())["val"]
    with Pool(6) as p:
        rows = [r for rs in p.map(one, names) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "h5c_dev_images.csv", index=False)
    osep_g = 0.0093
    summ = {}
    for v, d in df.groupby("variant"):
        tp, fp, fn = d.tp.sum(), d.fp.sum(), d.fn.sum()
        g = d.g_abs_err_sum.sum() / d.n_matched.sum()
        summ[v] = {"f1": 2 * tp / (2 * tp + fp + fn), "recall": tp / (tp + fn), "g_mae": g}
    base_g = summ["V0"]["g_mae"]
    for x in summ.values():
        x["share_of_osep_gain"] = (base_g - x["g_mae"]) / (base_g - osep_g)
        x["dF1"] = x["f1"] - summ["V0"]["f1"]
        x["dRecall"] = x["recall"] - summ["V0"]["recall"]
    (OUT / "h5c_dev_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(pd.DataFrame(summ).T.round(4).to_string())


if __name__ == "__main__":
    main()
